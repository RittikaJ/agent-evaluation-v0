"""Run & Evaluate: run agent on a question, show tool calls live, compute all 3 layers, upload to Langfuse."""

import json
import os
import time
from datetime import datetime

import anthropic
import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse

from agent import run_agent
from evaluate import (
    exact_match,
    f1_score,
    paragraph_retrieval_scores,
    action_order_score,
    search_efficiency,
    trajectory_composite,
    judge_reasoning,
    load_rubric,
)

load_dotenv()

st.set_page_config(page_title="Run & Evaluate", page_icon="🤖", layout="wide")
st.title("🤖 Run & Evaluate")
st.markdown(
    "Run the agent on a question, then score it across all evaluation layers — "
    "including plan quality, tool efficacy, failure diagnosis, and cost tracking."
)

st.divider()


@st.cache_data
def load_dataset():
    with open("dataset_local.json") as f:
        return json.load(f)


@st.cache_resource
def get_langfuse():
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )


dataset = load_dataset()
langfuse = get_langfuse()

# ── Helper functions ───────────────────────────────────────────────────────────

# Claude Sonnet pricing: $3/M input, $15/M output  →  ~$6.60/M avg (70/30 mix)
_CLAUDE_AVG_PRICE_PER_M = 6.60


def estimate_cost_usd(token_usage: int) -> float:
    """Estimate total API cost in USD."""
    return (token_usage / 1_000_000) * _CLAUDE_AVG_PRICE_PER_M


def classify_failure_mode(all_scores: dict, judge_scores: dict) -> dict:
    """Classify the dominant failure mode from composite scores."""
    answer_f1 = all_scores["answer_f1"]
    retrieval_f1 = all_scores["retrieval_f1"]

    if answer_f1 >= 0.7:
        return {
            "mode": "success",
            "emoji": "✅",
            "label": "Success",
            "color": "green",
            "description": "Agent answered correctly with sufficient evidence retrieval.",
        }

    retrieval_ok = retrieval_f1 >= 0.5
    search_ok = judge_scores.get("search_strategy", 1) >= 3
    grounded = judge_scores.get("groundedness", 1) >= 3
    coherent = judge_scores.get("reasoning_coherence", 1) >= 3

    if not retrieval_ok and not search_ok:
        return {
            "mode": "retrieval",
            "emoji": "🔍",
            "label": "Retrieval Failure",
            "color": "red",
            "description": (
                "Agent failed to retrieve required evidence. "
                "Search queries were poorly targeted or too broad."
            ),
        }
    elif retrieval_ok and not grounded:
        return {
            "mode": "synthesis",
            "emoji": "⚗️",
            "label": "Synthesis Failure",
            "color": "orange",
            "description": (
                "Agent retrieved relevant evidence but failed to synthesize "
                "a correct final answer from it."
            ),
        }
    elif not coherent:
        return {
            "mode": "planning",
            "emoji": "🗺️",
            "label": "Planning / Reasoning Failure",
            "color": "orange",
            "description": (
                "Agent had partial evidence but the reasoning chain was "
                "incomplete or logically broken."
            ),
        }
    else:
        return {
            "mode": "partial",
            "emoji": "⚠️",
            "label": "Partial Failure",
            "color": "yellow",
            "description": "Agent partially succeeded but the answer was incomplete or slightly off.",
        }


def compute_tool_efficacy(trajectory: list, answer: str) -> list:
    """Tag each read_paragraph call: did the retrieved content contribute to the answer?"""
    answer_lower = answer.lower()
    answer_tokens = set(w for w in answer_lower.split() if len(w) > 3)
    results = []
    for i, step in enumerate(trajectory):
        if step["tool"] == "read_paragraph":
            title = step["input"].get("title", "")
            output_text = str(step["output"])

            # Title keyword overlap
            title_hit = any(w in answer_lower for w in title.lower().split() if len(w) > 3)

            # Content token overlap (lightweight heuristic)
            para_tokens = set(w.lower() for w in output_text.split() if len(w) > 4)
            content_overlap = 0.0
            if para_tokens:
                overlap_count = len(para_tokens & answer_tokens)
                content_overlap = overlap_count / min(len(para_tokens), 30)

            contributed = title_hit or content_overlap > 0.05
            results.append({
                "step": i + 1,
                "title": title,
                "contributed": contributed,
                "content_overlap": round(content_overlap, 3),
            })
    return results


def judge_plan_quality(question: str, trajectory: list) -> dict:
    """LLM judge: Plan Quality, Plan Adherence, Step Efficiency (each 1–5)."""
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    steps_text = ""
    for i, step in enumerate(trajectory):
        if step["tool"] == "search_paragraphs":
            steps_text += f"Step {i+1}: SEARCH  query='{step['input'].get('query', '')}'\n"
            preview = str(step["output"])[:200].replace("\n", " ")
            steps_text += f"  -> Results preview: {preview}...\n\n"
        else:
            steps_text += f"Step {i+1}: READ  title='{step['input'].get('title', '')}'\n\n"

    if not steps_text:
        steps_text = "(No tool calls were made)"

    prompt = f"""You are an evaluation judge for a multi-hop QA agent.
Score the agent's trajectory on THREE dimensions.

## Question
{question}

## Agent Tool Calls
{steps_text}

## Dimensions

**Plan Quality (1-5)** — Did the agent decompose the question into targeted sub-queries?
- 5: Excellent decomposition; each search targets a distinct hop
- 4: Good with minor gaps
- 3: Reasonable but some searches are unfocused or partially redundant
- 2: Poor: searches too broad or miss key entities
- 1: No decomposition; copied full question or made irrelevant queries

**Plan Adherence (1-5)** — Did the agent follow through on its search plan systematically?
- 5: Every sub-goal was pursued to completion; no abandoned threads
- 4: Mostly followed through; one minor gap
- 3: Partial follow-through; one sub-goal abandoned
- 2: Frequently deviated; plan abandoned mid-way
- 1: No coherent plan followed

**Step Efficiency (1-5)** — Were calls necessary and non-redundant?
- 5: Every step was unique and necessary
- 4: Mostly efficient; one unnecessary step
- 3: Some redundant or ineffective calls
- 2: Several wasted steps (duplicate queries or irrelevant reads)
- 1: Mostly redundant or wasted steps

Respond with ONLY valid JSON (no markdown):
{{"plan_quality": N, "plan_adherence": N, "step_efficiency": N, "reasoning": "1-2 sentence explanation"}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        result = json.loads(text)
        return {
            "plan_quality": max(1, min(5, int(result.get("plan_quality", 1)))),
            "plan_adherence": max(1, min(5, int(result.get("plan_adherence", 1)))),
            "step_efficiency": max(1, min(5, int(result.get("step_efficiency", 1)))),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "plan_quality": 1,
            "plan_adherence": 1,
            "step_efficiency": 1,
            "reasoning": f"(Judge error: {e})",
        }


def decompose_subgoals(question: str, trajectory: list) -> dict:
    """LLM decomposes the question into sub-goals and checks which were addressed."""
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    queries = [
        step["input"].get("query", "")
        for step in trajectory
        if step["tool"] == "search_paragraphs"
    ]
    read_titles = [
        step["input"].get("title", "")
        for step in trajectory
        if step["tool"] == "read_paragraph"
    ]

    prompt = f"""For the multi-hop question below, identify 2-3 sub-goals the agent must accomplish.
Then check which sub-goals were addressed by the agent's actual searches and reads.

## Question
{question}

## Agent's Search Queries
{json.dumps(queries, indent=2)}

## Agent's Read Titles
{json.dumps(read_titles, indent=2)}

Respond with ONLY valid JSON (no markdown):
{{
  "subgoals": [
    {{"id": 1, "description": "short description", "addressed": true, "evidence": "which query/title addressed it"}}
  ],
  "coverage_pct": 0
}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return json.loads(text)
    except Exception:
        return {"subgoals": [], "coverage_pct": 0}


def judge_subgoal_quality(question: str, subgoals: list, trajectory: list) -> dict:
    """LLM judge: score the decomposed sub-questions on completeness and logical order (1–5 each).

    Completeness — do the identified sub-goals cover all information needed to answer?
    Logical Order — are sub-goals pursued in a logically valid sequence (A→B→answer)?
    """
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    queries = [
        s["input"].get("query", "")
        for s in trajectory
        if s["tool"] == "search_paragraphs"
    ]
    subgoals_text = (
        json.dumps(subgoals, indent=2) if subgoals else "No sub-goals were identified."
    )

    prompt = f"""You are evaluating how well an AI agent decomposed a multi-hop question into sub-questions.

## Question
{question}

## Identified Sub-goals
{subgoals_text}

## Agent's Search Queries (in execution order)
{json.dumps(queries, indent=2)}

Score on TWO dimensions (1–5 each):

**Completeness (1-5)** — Do the sub-goals cover ALL information needed to answer the question?
- 5: All necessary sub-goals identified and addressed; nothing important missing
- 4: Almost complete; one minor aspect uncovered
- 3: Partially complete; one major sub-goal missing or only partially addressed
- 2: Significant gaps; multiple important sub-goals missing
- 1: Superficial decomposition; most of the question is unaddressed

**Logical Order (1-5)** — Are sub-goals pursued in a logically valid order for multi-hop reasoning?
- 5: Perfect chain order; each step builds on the previous (A→B→answer for bridge questions)
- 4: Mostly ordered; one minor dependency slightly out of sequence
- 3: Partially ordered; some steps out of logical sequence but roughly sensible
- 2: Largely disordered; key dependencies not respected
- 1: Random order or no logical flow at all

Respond with ONLY valid JSON (no markdown):
{{"completeness": N, "logical_order": N, "reasoning": "1-2 sentence explanation"}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        result = json.loads(text)
        return {
            "completeness": max(1, min(5, int(result.get("completeness", 1)))),
            "logical_order": max(1, min(5, int(result.get("logical_order", 1)))),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "completeness": 1,
            "logical_order": 1,
            "reasoning": f"(Judge error: {e})",
        }


# ── Question Selector ─────────────────────────────────────────────────────────

options = []
for i, item in enumerate(dataset):
    qtype = item["metadata"]["type"]
    q = item["input"]["question"][:80]
    options.append(f"[{i}] ({qtype}) {q}")

default_idx = st.session_state.get("selected_idx", 0)
selected = st.selectbox("Select a question", options, index=default_idx)
idx = int(selected.split("]")[0].replace("[", ""))
item = dataset[idx]

# Question card
with st.container(border=True):
    st.markdown(f"### {item['input']['question']}")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Gold Answer**: `{item['expected_output']['answer']}`")
    col2.markdown(f"**Type**: `{item['metadata']['type']}`")
    col3.markdown(
        f"**Expected Paragraphs**: "
        f"{', '.join(item['expected_output']['trajectory']['expected_paragraphs'])}"
    )


# ── Display function ───────────────────────────────────────────────────────────

def display_results(result, item, all_scores, judge_scores, plan_scores,
                    subgoal_data, subgoal_quality, tool_efficacy, trace_id, latency_secs):
    """Display the full results for a completed agent run."""
    gold_answer = item["expected_output"]["answer"]
    gold_trajectory = item["expected_output"]["trajectory"]
    reasoning_score = sum(s / 5.0 for s in judge_scores.values()) / 3.0
    composite = (
        0.4 * all_scores["answer_f1"]
        + 0.35 * all_scores["trajectory_score"]
        + 0.25 * reasoning_score
    )
    failure = classify_failure_mode(all_scores, judge_scores)
    cost = estimate_cost_usd(result["token_usage"])

    # ══════════════════════════════════════════════════════════════════════
    # Composite Score Header
    # ══════════════════════════════════════════════════════════════════════

    st.divider()

    with st.container(border=True):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("📝 Answer (40%)", f"{all_scores['answer_f1'] * 100:.1f}%")
        "Token-level F1 vs gold answer"
        with col2:
            st.metric("\U0001f50d Trajectory (35%)", f"{all_scores['trajectory_score'] * 100:.1f}%")
            st.caption("Retrieval + order + efficiency")
        with col3:
            st.metric("\U0001f9e0 Reasoning (25%)", f"{reasoning_score * 100:.1f}%")
            st.caption("LLM judge: 3 rubric dimensions")
        with col4:
            st.metric("🎯 Composite", f"{composite * 100:.1f}%")
            st.caption("Weighted overall score")
        with col5:
            st.metric("⏱️ Latency", f"{latency_secs:.1f}s")
            st.caption("Agent wall-clock time")
        with col6:
            st.metric("💰 Est. Cost", f"${cost:.4f}")
            st.caption(f"{result['token_usage']:,} tokens")

    # ══════════════════════════════════════════════════════════════════════
    # Failure Mode Analysis  (NEW)
    # ══════════════════════════════════════════════════════════════════════

    st.header("🔬 Failure Mode Analysis")
    st.caption(
        "Classifies the dominant reason for any shortfall — retrieval failure, "
        "synthesis failure, or planning/reasoning failure."
    )

    if failure["mode"] == "success":
        st.success(f"{failure['emoji']} **{failure['label']}** — {failure['description']}")
    elif failure["color"] == "red":
        st.error(f"{failure['emoji']} **{failure['label']}** — {failure['description']}")
    elif failure["color"] in ("orange", "yellow"):
        st.warning(f"{failure['emoji']} **{failure['label']}** — {failure['description']}")
    else:
        st.info(f"{failure['emoji']} **{failure['label']}** — {failure['description']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        ret_pct = all_scores["retrieval_f1"] * 100
        st.metric(
            "Retrieval Health", f"{ret_pct:.0f}%",
            delta="✅ OK" if ret_pct >= 50 else "❌ Failed",
            delta_color="normal" if ret_pct >= 50 else "inverse",
        )
        st.caption("Was the right evidence found?")
    with col2:
        coh_pct = judge_scores.get("reasoning_coherence", 1) / 5.0 * 100
        st.metric(
            "Reasoning Health", f"{coh_pct:.0f}%",
            delta="✅ OK" if coh_pct >= 60 else "❌ Weak",
            delta_color="normal" if coh_pct >= 60 else "inverse",
        )
        st.caption("Was the reasoning chain coherent?")
    with col3:
        grd_pct = judge_scores.get("groundedness", 1) / 5.0 * 100
        st.metric(
            "Groundedness Health", f"{grd_pct:.0f}%",
            delta="✅ OK" if grd_pct >= 60 else "❌ Weak",
            delta_color="normal" if grd_pct >= 60 else "inverse",
        )
        st.caption("Is the answer backed by evidence?")

    # ══════════════════════════════════════════════════════════════════════
    # Answer Comparison
    # ══════════════════════════════════════════════════════════════════════

    st.header("📝 Answer Comparison")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Agent Answer**")
            st.markdown(result["answer"])
    with col2:
        with st.container(border=True):
            st.markdown("**Gold Answer**")
            st.markdown(f"**{gold_answer}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1", f"{all_scores['answer_f1'] * 100:.1f}%")
        st.caption("Token overlap between agent and gold answer")
    with col2:
        st.metric("Exact Match", "✅ Yes" if all_scores["answer_em"] > 0 else "❌ No")
        st.caption("Do the normalized answers match exactly?")

    if all_scores["answer_f1"] < 0.3:
        st.warning(
            "⚠️ Low F1 — the agent likely gave a verbose answer. "
            "The gold answer is usually 1–3 words."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Plan Quality Assessment  (NEW)
    # ══════════════════════════════════════════════════════════════════════

    st.header("🗺️ Plan Quality Assessment")
    st.caption(
        "LLM judge evaluates how well the agent decomposed the question "
        "and executed a coherent search plan — beyond simple F1 metrics."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric("Plan Quality", f"{plan_scores['plan_quality']}/5")
            st.caption("Did the agent decompose the question into targeted sub-queries?")
    with col2:
        with st.container(border=True):
            st.metric("Plan Adherence", f"{plan_scores['plan_adherence']}/5")
            st.caption("Did the agent follow through on its search plan systematically?")
    with col3:
        with st.container(border=True):
            st.metric("Step Efficiency (LLM)", f"{plan_scores['step_efficiency']}/5")
            st.caption("Were each step necessary and non-redundant?")

    if plan_scores.get("reasoning"):
        st.info(f"💬 **Judge rationale**: {plan_scores['reasoning']}")

    # Sub-goal decomposition
    if subgoal_data and subgoal_data.get("subgoals"):
        st.subheader("🎯 Sub-goal Coverage")
        coverage = subgoal_data.get("coverage_pct", 0)
        st.caption(
            "The question was automatically decomposed into sub-goals. "
            f"The agent addressed **{coverage}%** of them."
        )
        st.progress(coverage / 100, text=f"{coverage}% sub-goals addressed")
        for sg in subgoal_data["subgoals"]:
            icon = "✅" if sg.get("addressed") else "❌"
            body = f"{icon} **Sub-goal {sg['id']}**: {sg['description']}"
            if sg.get("evidence"):
                body += f"\n\n  _{sg['evidence']}_"
            if sg.get("addressed"):
                st.success(body)
            else:
                st.error(body)

        # LLM quality scores for the sub-goal decomposition
        if subgoal_quality:
            st.markdown("**Sub-goal Decomposition Quality (LLM Judge)**")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.metric("Completeness", f"{subgoal_quality['completeness']}/5")
                    st.caption("Do the sub-goals cover all aspects needed to answer?")
            with col2:
                with st.container(border=True):
                    st.metric("Logical Order", f"{subgoal_quality['logical_order']}/5")
                    st.caption("Are sub-goals ordered correctly for multi-hop reasoning?")
            if subgoal_quality.get("reasoning"):
                st.info(f"💬 **Judge rationale**: {subgoal_quality['reasoning']}")

    # ══════════════════════════════════════════════════════════════════════
    # Tool-Use Efficacy  (NEW)
    # ══════════════════════════════════════════════════════════════════════

    st.header("🛠️ Tool-Use Efficacy")
    st.caption(
        "Did each `read_paragraph` call actually contribute to the final answer? "
        "Wasted reads indicate over-retrieval or poor evidence synthesis."
    )

    if not tool_efficacy:
        st.info("No `read_paragraph` calls were made — agent answered from search results only.")
    else:
        contributed_count = sum(1 for e in tool_efficacy if e["contributed"])
        total_reads = len(tool_efficacy)
        eff_pct = contributed_count / total_reads * 100 if total_reads else 0

        st.progress(
            eff_pct / 100,
            text=f"Efficacy: {contributed_count}/{total_reads} reads contributed "
                 f"to the answer ({eff_pct:.0f}%)",
        )
        for e in tool_efficacy:
            if e["contributed"]:
                st.success(f"✅ Step {e['step']}: `{e['title']}` — **contributed to answer**")
            else:
                st.error(
                    f"❌ Step {e['step']}: `{e['title']}` — "
                    f"not reflected in final answer (overlap={e['content_overlap']:.2f})"
                )

    # ══════════════════════════════════════════════════════════════════════
    # Two-Level Evaluation  (NEW)
    # ══════════════════════════════════════════════════════════════════════

    st.header("📊 Two-Level Evaluation")
    st.caption(
        "A comprehensive view of both **turn-level** (per step) and "
        "**end-to-end** (overall task) performance."
    )

    tab_turn, tab_e2e = st.tabs(["🔄 Turn-Level (per step)", "🏁 End-to-End (overall)"])

    with tab_turn:
        st.caption(
            "Per-step assessment: was each tool call well-formed, "
            "targeted, and did it return useful output?"
        )
        if not result["trajectory"]:
            st.warning("No tool calls were made.")
        else:
            for i, step in enumerate(result["trajectory"]):
                tool = step["tool"]
                inp = step["input"]
                output = str(step["output"])

                if tool == "search_paragraphs":
                    query = inp.get("query", "")
                    full_q = item["input"]["question"].lower().strip()
                    is_copy = query.lower().strip() == full_q
                    is_empty = "No paragraphs found" in output

                    if is_copy:
                        badge, issue = "⚠️", " — query = full question (anti-pattern)"
                    elif is_empty:
                        badge, issue = "❌", " — no results returned"
                    else:
                        badge, issue = "✅", ""

                    with st.expander(
                        f"{badge} Step {i+1} — SEARCH `{query}`{issue}", expanded=False
                    ):
                        if is_copy:
                            st.warning(
                                "Anti-pattern detected: the query is a verbatim copy of the "
                                "question. Use focused sub-queries that target specific entities."
                            )
                        if is_empty:
                            st.warning("No results — try a simpler or different query term.")
                        st.code(output[:600], language=None)

                else:  # read_paragraph
                    title = inp.get("title", "")
                    not_found = "No paragraph found" in output
                    eff_match = next(
                        (e for e in tool_efficacy if e["step"] == i + 1), None
                    )
                    cited = eff_match["contributed"] if eff_match else False

                    if not_found:
                        badge, note = "❌", " — paragraph not found"
                    elif cited:
                        badge, note = "✅", " — cited in answer"
                    else:
                        badge, note = "⚠️", " — not reflected in answer"

                    with st.expander(
                        f"{badge} Step {i+1} — READ `{title}`{note}", expanded=False
                    ):
                        if not_found:
                            st.error(
                                "Paragraph title not found. "
                                "Use the exact title returned by search results."
                            )
                        elif not cited:
                            st.warning(
                                "This paragraph was read but its content is not reflected "
                                "in the final answer — potential synthesis gap."
                            )
                        else:
                            st.success("Content from this paragraph contributed to the final answer.")
                        st.code(output[:600], language=None)

    with tab_e2e:
        st.caption("Overall task completion across the three evaluation pillars.")
        col1, col2, col3 = st.columns(3)
        with col1:
            goal_done = all_scores["answer_f1"] >= 0.7
            st.metric(
                "Goal Completion",
                "✅ Achieved" if goal_done else "❌ Not Achieved",
            )
            st.caption("answer_f1 ≥ 0.7 = task complete")
        with col2:
            progress_rate = (
                0.4 * all_scores["answer_f1"]
                + 0.3 * all_scores["retrieval_f1"]
                + 0.3 * (judge_scores.get("reasoning_coherence", 1) / 5.0)
            )
            st.metric("Progress Rate", f"{progress_rate * 100:.1f}%")
            st.caption("Partial credit: answer + evidence + reasoning")
        with col3:
            milestone_pct = subgoal_data.get("coverage_pct", 0) if subgoal_data else 0
            st.metric("Sub-goal Coverage", f"{milestone_pct}%")
            st.caption("Fraction of identified sub-goals addressed")

        # Visual summary bar
        st.markdown("---")
        st.markdown("**Score breakdown across layers:**")
        cols = st.columns([2, 1])
        with cols[0]:
            bar_data = {
                "Answer F1": all_scores["answer_f1"],
                "Retrieval F1": all_scores["retrieval_f1"],
                "Action Order": all_scores["action_order"],
                "Efficiency": all_scores["efficiency"],
                "Groundedness": judge_scores.get("groundedness", 1) / 5.0,
                "Reasoning": judge_scores.get("reasoning_coherence", 1) / 5.0,
                "Search Strategy": judge_scores.get("search_strategy", 1) / 5.0,
                "Plan Quality": plan_scores["plan_quality"] / 5.0,
                "Plan Adherence": plan_scores["plan_adherence"] / 5.0,
            }
            import pandas as pd
            df = pd.DataFrame(
                {"Score": list(bar_data.values())},
                index=list(bar_data.keys()),
            )
            st.bar_chart(df, horizontal=True)

    # ══════════════════════════════════════════════════════════════════════
    # Agent Trajectory (enhanced with efficacy badges)
    # ══════════════════════════════════════════════════════════════════════

    st.header("🔄 Agent Trajectory")
    st.caption("Full tool-call log. `read_paragraph` steps are tagged with citation status.")

    if not result["trajectory"]:
        st.warning("The agent made no tool calls.")
    else:
        for i, step in enumerate(result["trajectory"]):
            tool = step["tool"]
            inp = step["input"]
            output = str(step["output"])

            if tool == "search_paragraphs":
                icon = "🔎"
                label = f"Search: `{inp.get('query', '')}`"
            else:
                icon = "📖"
                eff_match = next((e for e in tool_efficacy if e["step"] == i + 1), None)
                cited_badge = (
                    " 🟢 cited" if (eff_match and eff_match["contributed"]) else " 🔴 not cited"
                )
                label = f"Read: `{inp.get('title', '')}`{cited_badge}"

            with st.expander(f"{icon} Step {i+1} — {label}", expanded=False):
                st.code(output, language=None)

    # ══════════════════════════════════════════════════════════════════════
    # Trajectory Evaluation (existing, enhanced)
    # ══════════════════════════════════════════════════════════════════════

    st.header("📈 Trajectory Evaluation")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Retrieval F1", f"{all_scores['retrieval_f1'] * 100:.0f}%")
        st.caption("Found the right paragraphs?")
    with col2:
        st.metric("Action Order (LCS)", f"{all_scores['action_order'] * 100:.0f}%")
        st.caption("Followed the right tool sequence?")
    with col3:
        st.metric("Efficiency (det.)", f"{all_scores['efficiency'] * 100:.0f}%")
        st.caption("Used the minimum calls needed?")
    with col4:
        st.metric("Trajectory Score", f"{all_scores['trajectory_score'] * 100:.0f}%")
        st.caption("40% retrieval + 40% order + 20% efficiency")

    st.subheader("Evidence Retrieval Check")
    gold_titles = set(gold_trajectory["expected_paragraphs"])
    retrieved_titles = set()
    for step in result["trajectory"]:
        if step["tool"] == "read_paragraph":
            retrieved_titles.add(step["input"].get("title", ""))
        elif step["tool"] == "search_paragraphs":
            try:
                results_parsed = (
                    json.loads(step["output"])
                    if isinstance(step["output"], str)
                    else step["output"]
                )
                if isinstance(results_parsed, list):
                    for r in results_parsed:
                        if isinstance(r, dict) and "title" in r:
                            retrieved_titles.add(r["title"])
            except (json.JSONDecodeError, TypeError):
                pass

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Required paragraphs**")
        for t in gold_titles:
            if t in retrieved_titles:
                st.success(f"✅ {t}")
            else:
                st.error(f"❌ {t}")
    with col2:
        st.markdown("**Action sequence comparison**")
        gold_actions = [a["tool"].replace("_", " ").title() for a in gold_trajectory["actions"]]
        agent_actions = [s["tool"].replace("_", " ").title() for s in result["trajectory"]]
        st.markdown(f"**Gold**: {' → '.join(gold_actions)}")
        st.markdown(f"**Agent**: {' → '.join(agent_actions) if agent_actions else '(none)'}")

    # ══════════════════════════════════════════════════════════════════════
    # Reasoning Evaluation (existing)
    # ══════════════════════════════════════════════════════════════════════

    st.header("🧠 Reasoning Evaluation (LLM Judge)")

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric("Groundedness", f"{judge_scores['groundedness']}/5")
            st.caption("Is the answer backed by retrieved evidence?")
    with col2:
        with st.container(border=True):
            st.metric("Reasoning Coherence", f"{judge_scores['reasoning_coherence']}/5")
            st.caption("Does the multi-hop chain make sense?")
    with col3:
        with st.container(border=True):
            st.metric("Search Strategy", f"{judge_scores['search_strategy']}/5")
            st.caption("Were search queries well-chosen?")

    # ══════════════════════════════════════════════════════════════════════
    # Cost & Latency  (NEW)
    # ══════════════════════════════════════════════════════════════════════

    st.header("⚡ Cost & Latency")
    st.caption(
        "Production-readiness metrics. Latency and cost are critical for "
        "enterprise deployment at scale."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Wall-clock Time", f"{latency_secs:.2f}s")
        st.caption("Total agent execution time")
    with col2:
        st.metric("Total Tokens", f"{result['token_usage']:,}")
        st.caption("Input + output tokens used")
    with col3:
        steps = max(len(result["trajectory"]), 1)
        st.metric("Tokens / Step", f"{result['token_usage'] // steps:,}")
        st.caption("Avg tokens per tool call")
    with col4:
        st.metric("Estimated Cost", f"${cost:.4f}")
        st.caption("Claude Sonnet @ ~$6.60/M avg")

    monthly_queries = st.slider(
        "📊 Monthly query volume (for cost projection)",
        min_value=100, max_value=100_000, value=1_000, step=100,
    )
    monthly_cost = monthly_queries * cost
    st.info(
        f"💡 At **{monthly_queries:,} queries/month** → "
        f"estimated **${monthly_cost:.2f}/month** in API costs"
    )

    st.divider()
    st.caption(
        f"Tokens: {result['token_usage']} | Latency: {latency_secs:.1f}s | "
        f"Trace: `{trace_id}` | All scores uploaded to Langfuse."
    )


# ── Run Agent ─────────────────────────────────────────────────────────────────

if st.button("▶️ Run Agent & Evaluate", type="primary", use_container_width=True):
    gold_answer = item["expected_output"]["answer"]
    gold_trajectory = item["expected_output"]["trajectory"]

    with st.status("Running agent...", expanded=True) as status:
        st.write("🤖 Sending question to agent...")

        # Create Langfuse trace
        experiment_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        lf_dataset = langfuse.get_dataset("rittika-hotpotqa-10")

        lf_item = None
        for li in lf_dataset.items:
            if li.id == item["id"]:
                lf_item = li
                break

        if lf_item is None:
            st.error(f"Could not find Langfuse dataset item with id {item['id']}")
            st.stop()

        # ── Run agent (timed) ──
        agent_start = time.time()
        with langfuse.start_as_current_observation(name=experiment_name) as span:
            result = run_agent(item["input"]["question"], item["input"]["context"])
            span.update(
                input=item["input"]["question"],
                output=result["answer"],
                metadata={
                    "trajectory_length": len(result["trajectory"]),
                    "token_usage": result["token_usage"],
                },
            )
            trace_id = span.trace_id
            try:
                langfuse.api.dataset_run_items.create(
                    run_name=experiment_name,
                    dataset_item_id=lf_item.id,
                    trace_id=trace_id,
                    observation_id=span.id,
                )
            except Exception:
                pass  # Dataset run linking is optional for demo runs
        latency_secs = time.time() - agent_start

        st.write(f"✅ Agent made **{len(result['trajectory'])} tool calls** in {latency_secs:.1f}s:")
        for i, step in enumerate(result["trajectory"]):
            tool = step["tool"]
            if tool == "search_paragraphs":
                st.write(f"  🔎 {i+1}. Search: `{step['input'].get('query', '')}`")
            else:
                st.write(f"  📖 {i+1}. Read: `{step['input'].get('title', '')}`")
        st.write(
            f"💬 Agent answered: **{result['answer'][:150]}"
            f"{'...' if len(result['answer']) > 150 else ''}**"
        )

        st.write("---")
        st.write("📊 Computing Layer 1 & 2 scores...")

        # ── Layer 1 ──
        answer_em = exact_match(result["answer"], gold_answer)
        answer_f1 = f1_score(result["answer"], gold_answer)

        # ── Layer 2 ──
        retrieval = paragraph_retrieval_scores(result["trajectory"], gold_trajectory)
        action_score = action_order_score(result["trajectory"], gold_trajectory)
        efficiency = search_efficiency(result["trajectory"], gold_trajectory)
        traj_score = trajectory_composite(retrieval["retrieval_f1"], action_score, efficiency)

        # ── Layer 3: existing rubrics ──
        st.write("🧠 Running LLM judge (3 rubric calls)...")
        rubrics = {
            "groundedness": load_rubric("groundedness"),
            "reasoning_coherence": load_rubric("reasoning_coherence"),
            "search_strategy": load_rubric("search_strategy"),
        }
        judge_scores = {}
        for name, rubric_text in rubrics.items():
            st.write(f"  Judging `{name}`...")
            judge_scores[name] = judge_reasoning(
                item["input"]["question"],
                result["answer"],
                gold_answer,
                result["trajectory"],
                name,
                rubric_text,
            )

        # ── Plan Quality Judge (new) ──
        st.write("🗺️ Running plan quality judge...")
        plan_scores = judge_plan_quality(
            item["input"]["question"], result["trajectory"]
        )

        # ── Sub-goal Decomposition (new) ──
        st.write("🎯 Decomposing sub-goals...")
        subgoal_data = decompose_subgoals(
            item["input"]["question"], result["trajectory"]
        )

        # ── Sub-goal Quality Judge (new) ──
        st.write("📐 Judging sub-goal completeness & logical order...")
        subgoal_quality = judge_subgoal_quality(
            item["input"]["question"],
            subgoal_data.get("subgoals", []),
            result["trajectory"],
        )

        # ── Tool Efficacy (new) ──
        tool_efficacy = compute_tool_efficacy(result["trajectory"], result["answer"])

        # ── Compute composite ──
        reasoning_score_val = sum(s / 5.0 for s in judge_scores.values()) / 3.0
        composite_val = (
            0.4 * answer_f1 + 0.35 * traj_score + 0.25 * reasoning_score_val
        )

        # ── Upload to Langfuse ──
        st.write("📤 Uploading scores to Langfuse...")

        all_scores = {
            "answer_f1": answer_f1,
            "answer_em": answer_em,
            "paragraph_precision": retrieval["paragraph_precision"],
            "paragraph_recall": retrieval["paragraph_recall"],
            "retrieval_f1": retrieval["retrieval_f1"],
            "action_order": action_score,
            "efficiency": efficiency,
            "trajectory_score": traj_score,
        }

        # Tool efficacy rate (0-1)
        eff_rate = (
            sum(1 for e in tool_efficacy if e["contributed"]) / len(tool_efficacy)
            if tool_efficacy else 0.0
        )

        langfuse_scores = {
            "Answer Accuracy": answer_f1,
            "Evidence Quality": traj_score,
            "Groundedness": judge_scores["groundedness"],
            "Reasoning Quality": judge_scores["reasoning_coherence"],
            "Search Quality": judge_scores["search_strategy"],
            "Overall Score": composite_val,
            # Plan quality scores
            "Plan Quality": plan_scores["plan_quality"],
            "Plan Adherence": plan_scores["plan_adherence"],
            "Step Efficiency LLM": plan_scores["step_efficiency"],
            # Sub-goal quality scores (new)
            "Subgoal Completeness": subgoal_quality["completeness"],
            "Subgoal Logical Order": subgoal_quality["logical_order"],
            # Tool efficacy rate (new)
            "Tool Efficacy Rate": eff_rate,
        }

        for score_name, score_val in langfuse_scores.items():
            langfuse.create_score(
                name=score_name,
                value=score_val,
                trace_id=trace_id,
                data_type="NUMERIC",
            )
        langfuse.flush()

        status.update(
            label="✅ Done — agent run scored and uploaded to Langfuse",
            state="complete",
        )

    # Store in session state
    st.session_state.agent_result = result
    st.session_state.trace_id = trace_id
    st.session_state.selected_idx = idx
    st.session_state.all_scores = all_scores
    st.session_state.judge_scores = judge_scores
    st.session_state.plan_scores = plan_scores
    st.session_state.subgoal_data = subgoal_data
    st.session_state.subgoal_quality = subgoal_quality
    st.session_state.tool_efficacy = tool_efficacy
    st.session_state.latency_secs = latency_secs

    display_results(
        result, item, all_scores, judge_scores, plan_scores,
        subgoal_data, subgoal_quality, tool_efficacy, trace_id, latency_secs,
    )

# ── Show Previous Run (persisted across tab switches) ─────────────────────────

elif "agent_result" in st.session_state and "all_scores" in st.session_state:
    prev_idx = st.session_state.get("selected_idx", 0)
    prev_item = dataset[prev_idx]

    if prev_idx != idx:
        st.info(
            f"Showing results from previous run on question [{prev_idx}]. "
            "Select that question or click 'Run Agent & Evaluate' to run on this one."
        )

    display_results(
        st.session_state.agent_result,
        prev_item,
        st.session_state.all_scores,
        st.session_state.judge_scores,
        st.session_state.get("plan_scores", {"plan_quality": 0, "plan_adherence": 0, "step_efficiency": 0, "reasoning": ""}),
        st.session_state.get("subgoal_data", {"subgoals": [], "coverage_pct": 0}),
        st.session_state.get("subgoal_quality", {"completeness": 0, "logical_order": 0, "reasoning": ""}),
        st.session_state.get("tool_efficacy", []),
        st.session_state.trace_id,
        st.session_state.get("latency_secs", 0.0),
    )
else:
    st.info("Select a question and click '▶️ Run Agent & Evaluate' to start.")
