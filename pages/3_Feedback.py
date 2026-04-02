"""
Feedback page: structured human-in-the-loop evaluation.

Provides:
  - 6 dimension ratings (1-5 sliders) replacing binary thumbs up/down
  - Multi-select failure taxonomy (retrieval / synthesis / planning / efficiency)
  - Correction field if the agent's answer was wrong
  - Free-text reviewer notes
  - Step-level annotation (helpful / unhelpful / harmful per tool call)
  - Comparison view: human ratings vs LLM judge scores side-by-side
  - Full upload to Langfuse (7 named scores + structured comment)
  - Quick thumbs up/down still available for fast review
"""

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

st.set_page_config(page_title="Feedback", page_icon="👤", layout="wide")
st.title("👤 Human-in-the-Loop Feedback")
st.markdown(
    "Review the agent's run, rate each dimension, tag failure categories, "
    "and optionally correct the answer. All ratings sync to Langfuse."
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

# ── Guard: require a completed run ────────────────────────────────────────────

if "agent_result" not in st.session_state or "trace_id" not in st.session_state:
    st.warning(
        "⚠️ No agent run found in this session. "
        "Go to **Run & Evaluate**, run a question, then return here."
    )
    st.stop()

# Pull everything from session state
result = st.session_state.agent_result
trace_id = st.session_state.trace_id
idx = st.session_state.get("selected_idx", 0)
item = dataset[idx]
all_scores = st.session_state.get("all_scores", {})
judge_scores = st.session_state.get("judge_scores", {})
plan_scores = st.session_state.get("plan_scores", {})
tool_efficacy = st.session_state.get("tool_efficacy", [])
subgoal_data = st.session_state.get("subgoal_data", {})
latency_secs = st.session_state.get("latency_secs", 0.0)

gold_answer = item["expected_output"]["answer"]
gold_trajectory = item["expected_output"]["trajectory"]

# Derive composite if we have scores
reasoning_score_norm = (
    sum(s / 5.0 for s in judge_scores.values()) / 3.0 if judge_scores else 0.0
)
composite = (
    0.4 * all_scores.get("answer_f1", 0)
    + 0.35 * all_scores.get("trajectory_score", 0)
    + 0.25 * reasoning_score_norm
) if all_scores else 0.0

# Simple failure mode label (mirrors classify_failure_mode in page 2)
def _failure_label(a_scores, j_scores):
    f1 = a_scores.get("answer_f1", 0)
    if f1 >= 0.7:
        return "success", "✅ Success"
    if a_scores.get("retrieval_f1", 0) < 0.5 and j_scores.get("search_strategy", 1) < 3:
        return "retrieval", "🔍 Retrieval Failure"
    if a_scores.get("retrieval_f1", 0) >= 0.5 and j_scores.get("groundedness", 1) < 3:
        return "synthesis", "⚗️ Synthesis Failure"
    if j_scores.get("reasoning_coherence", 1) < 3:
        return "planning", "🗺️ Planning Failure"
    return "partial", "⚠️ Partial Failure"

failure_key, failure_label = _failure_label(all_scores, judge_scores)

# ── Section 1: Run Context ─────────────────────────────────────────────────────

st.subheader("Run Summary")

with st.container(border=True):
    st.markdown(f"**{item['input']['question']}**")
    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.markdown(f"**Type**: `{item['metadata']['type']}`")
    meta2.markdown(f"**Level**: `{item['metadata'].get('level', '?')}`")
    meta3.markdown(f"**Trace**: `{trace_id[:24]}...`")
    if all_scores:
        meta4.markdown(f"**Auto diagnosis**: {failure_label}")

# Answer side-by-side
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**Agent Answer**")
        ans_f1 = all_scores.get("answer_f1", 0)
        badge = "✅" if ans_f1 >= 0.7 else ("⚠️" if ans_f1 >= 0.3 else "❌")
        st.markdown(f"{badge} {result['answer']}")
        if all_scores:
            st.caption(f"F1: {ans_f1*100:.1f}%  |  EM: {'Yes' if all_scores.get('answer_em', 0) > 0 else 'No'}")
with col2:
    with st.container(border=True):
        st.markdown("**Gold Answer**")
        st.markdown(f"**{gold_answer}**")

# Automated scores reference bar (if available)
if all_scores:
    st.subheader("Automated Scores (for reference)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Answer F1", f"{all_scores.get('answer_f1', 0)*100:.1f}%")
    c2.metric("Retrieval F1", f"{all_scores.get('retrieval_f1', 0)*100:.1f}%")
    c3.metric("Groundedness", f"{judge_scores.get('groundedness', '—')}/5")
    c4.metric("Reasoning", f"{judge_scores.get('reasoning_coherence', '—')}/5")
    c5.metric("Search", f"{judge_scores.get('search_strategy', '—')}/5")
    c6.metric("Composite", f"{composite*100:.1f}%")

    if plan_scores:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Plan Quality", f"{plan_scores.get('plan_quality', '—')}/5")
        p2.metric("Plan Adherence", f"{plan_scores.get('plan_adherence', '—')}/5")
        p3.metric("Step Efficiency", f"{plan_scores.get('step_efficiency', '—')}/5")
        p4.metric("Latency", f"{latency_secs:.1f}s")

# Trajectory view with efficacy badges
with st.expander("Agent Trajectory", expanded=False):
    if not result["trajectory"]:
        st.warning("No tool calls were made.")
    else:
        for i, step in enumerate(result["trajectory"]):
            tool = step["tool"]
            inp = step["input"]
            if tool == "search_paragraphs":
                st.markdown(f"🔎 **Step {i+1}** SEARCH: `{inp.get('query', '')}`")
            else:
                eff = next((e for e in tool_efficacy if e["step"] == i + 1), None)
                badge = "🟢 cited" if (eff and eff["contributed"]) else "🔴 not cited"
                st.markdown(f"📖 **Step {i+1}** READ: `{inp.get('title', '')}` — {badge}")

# Sub-goals (if available)
if subgoal_data and subgoal_data.get("subgoals"):
    with st.expander(
        f"Sub-goal Coverage: {subgoal_data.get('coverage_pct', 0)}%", expanded=False
    ):
        for sg in subgoal_data["subgoals"]:
            icon = "✅" if sg.get("addressed") else "❌"
            st.markdown(f"{icon} **Sub-goal {sg['id']}**: {sg['description']}")
            if sg.get("evidence"):
                st.caption(sg["evidence"])

st.divider()

# ── Section 2: Dimension Ratings ──────────────────────────────────────────────

st.header("📊 Dimension Ratings")
st.caption(
    "Rate each dimension 1–5. These are uploaded as named scores in Langfuse "
    "alongside (not replacing) the automated scores."
)

_labels = {
    1: "1 — Very poor",
    2: "2 — Poor",
    3: "3 — Acceptable",
    4: "4 — Good",
    5: "5 — Excellent",
}

# Suggest starting values from automated scores (map 0-1 or 1-5 to slider default)
def _suggest(val_01, fallback=3):
    if val_01 is None:
        return fallback
    return max(1, min(5, round(val_01 * 5)))

def _suggest_from_judge(val_15, fallback=3):
    if not val_15:
        return fallback
    return max(1, min(5, int(val_15)))

col1, col2, col3 = st.columns(3)

with col1:
    default_overall = _suggest(composite, fallback=3)
    rating_overall = st.select_slider(
        "Overall Quality",
        options=[1, 2, 3, 4, 5],
        value=default_overall,
        format_func=lambda x: _labels[x],
    )
    st.caption("Your holistic impression of this agent run")

    default_answer = _suggest(all_scores.get("answer_f1"), fallback=3)
    rating_answer = st.select_slider(
        "Answer Correctness",
        options=[1, 2, 3, 4, 5],
        value=default_answer,
        format_func=lambda x: _labels[x],
    )
    st.caption("Is the final answer factually correct and concise?")

with col2:
    default_search = _suggest_from_judge(judge_scores.get("search_strategy"), fallback=3)
    rating_search = st.select_slider(
        "Search Strategy",
        options=[1, 2, 3, 4, 5],
        value=default_search,
        format_func=lambda x: _labels[x],
    )
    st.caption("Were search queries well-targeted and decomposed?")

    default_evidence = _suggest_from_judge(judge_scores.get("groundedness"), fallback=3)
    rating_evidence = st.select_slider(
        "Evidence Use (Groundedness)",
        options=[1, 2, 3, 4, 5],
        value=default_evidence,
        format_func=lambda x: _labels[x],
    )
    st.caption("Did the agent's answer reflect what it retrieved?")

with col3:
    default_reasoning = _suggest_from_judge(judge_scores.get("reasoning_coherence"), fallback=3)
    rating_reasoning = st.select_slider(
        "Reasoning Chain",
        options=[1, 2, 3, 4, 5],
        value=default_reasoning,
        format_func=lambda x: _labels[x],
    )
    st.caption("Was the multi-hop reasoning logical and complete?")

    default_efficiency = _suggest(all_scores.get("efficiency"), fallback=3)
    rating_efficiency = st.select_slider(
        "Efficiency",
        options=[1, 2, 3, 4, 5],
        value=default_efficiency,
        format_func=lambda x: _labels[x],
    )
    st.caption("Did the agent avoid unnecessary or redundant steps?")

# ── Human vs LLM Comparison chart ─────────────────────────────────────────────

if all_scores and judge_scores:
    st.subheader("Human vs LLM Judge Comparison")
    st.caption(
        "Blue = LLM judge (normalised to 0–1). Red = your ratings (normalised to 0–1). "
        "Disagreements highlight where the automated judge may need recalibration."
    )
    compare_data = {
        "Answer": (all_scores.get("answer_f1", 0), rating_answer / 5.0),
        "Search Strategy": (judge_scores.get("search_strategy", 1) / 5.0, rating_search / 5.0),
        "Groundedness": (judge_scores.get("groundedness", 1) / 5.0, rating_evidence / 5.0),
        "Reasoning": (judge_scores.get("reasoning_coherence", 1) / 5.0, rating_reasoning / 5.0),
        "Efficiency": (all_scores.get("efficiency", 0), rating_efficiency / 5.0),
        "Overall": (composite, rating_overall / 5.0),
    }
    df_compare = pd.DataFrame(
        {
            "LLM Judge": [v[0] for v in compare_data.values()],
            "Human Rating": [v[1] for v in compare_data.values()],
        },
        index=list(compare_data.keys()),
    )
    st.bar_chart(df_compare, horizontal=True, color=["#4a90d9", "#e05c5c"])

    # Agreement summary
    disagreements = [
        k for k, (llm, human) in compare_data.items() if abs(llm - human) > 0.3
    ]
    if disagreements:
        st.warning(
            f"⚠️ Notable disagreement between human and LLM judge on: "
            f"**{', '.join(disagreements)}**. "
            "Consider whether the rubric needs adjustment for these dimensions."
        )
    else:
        st.success("✅ Human ratings broadly agree with the LLM judge on all dimensions.")

st.divider()

# ── Section 3: Failure Classification ─────────────────────────────────────────

st.header("🔬 Failure Classification")
st.caption(
    "Tag every failure type that applies. This builds the failure distribution "
    "needed to prioritise what to fix next."
)

FAILURE_OPTIONS = [
    "✅ No failure — agent succeeded",
    "🔍 Retrieved wrong paragraphs (irrelevant search results)",
    "🔍 Missed a required paragraph entirely",
    "🔍 Search query was too broad / copied full question",
    "⚗️ Had correct evidence but final answer was still wrong",
    "⚗️ Answer was verbose — buried the correct token",
    "⚗️ Answer hallucinated content not in retrieved evidence",
    "🗺️ Multi-hop reasoning chain was broken or incomplete",
    "🗺️ Agent abandoned a sub-goal mid-trajectory",
    "⚡ Too many redundant or duplicate tool calls",
    "❓ Other (explain in notes)",
]

# Pre-select based on automated failure mode
preselect = []
if failure_key == "success":
    preselect = ["✅ No failure — agent succeeded"]
elif failure_key == "retrieval":
    preselect = ["🔍 Missed a required paragraph entirely"]
elif failure_key == "synthesis":
    preselect = ["⚗️ Had correct evidence but final answer was still wrong"]
elif failure_key == "planning":
    preselect = ["🗺️ Multi-hop reasoning chain was broken or incomplete"]

selected_failures = st.multiselect(
    "What went wrong? (select all that apply)",
    options=FAILURE_OPTIONS,
    default=preselect,
    help="Pre-filled from the automated failure mode classifier. Adjust as needed.",
)

if "✅ No failure — agent succeeded" in selected_failures and len(selected_failures) > 1:
    st.warning("You selected both 'No failure' and failure categories. Please review your selection.")

st.divider()

# ── Section 4: Correction & Notes ─────────────────────────────────────────────

st.header("✏️ Correction & Notes")

col1, col2 = st.columns(2)
with col1:
    corrected_answer = st.text_input(
        "Corrected answer (if agent was wrong)",
        value="",
        placeholder=f"Gold: {gold_answer}",
        help="If you know the correct answer, enter it here. Stored in the Langfuse comment.",
    )
    if corrected_answer:
        st.caption(
            f"Gold: `{gold_answer}` | Agent: `{result['answer'][:60]}` "
            f"| Your correction: `{corrected_answer}`"
        )

with col2:
    quality_note = st.text_area(
        "Reviewer notes (optional)",
        value="",
        placeholder=(
            "e.g. 'The agent found paragraph A correctly but failed to extract "
            "the founding year from paragraph B...'"
        ),
        height=110,
    )

# Step-level annotation
with st.expander("📍 Step-Level Annotation (optional)", expanded=False):
    st.caption(
        "Mark individual tool calls as helpful, unhelpful, or harmful. "
        "This creates a fine-grained record of which steps caused the failure."
    )
    step_annotations = {}
    if not result["trajectory"]:
        st.info("No tool calls to annotate.")
    else:
        for i, step in enumerate(result["trajectory"]):
            tool = step["tool"]
            inp = step["input"]
            key_val = inp.get("query", inp.get("title", ""))
            col_a, col_b = st.columns([2, 3])
            with col_a:
                icon = "🔎" if tool == "search_paragraphs" else "📖"
                st.markdown(f"{icon} **Step {i+1}**: `{key_val[:60]}`")
            with col_b:
                annotation = st.radio(
                    f"step_{i+1}_label",
                    options=["neutral ➖", "helpful ✅", "unhelpful ⚠️", "harmful ❌"],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"step_ann_{i}",
                )
            step_annotations[i + 1] = annotation

st.divider()

# ── Section 5: Submit ─────────────────────────────────────────────────────────

st.header("📤 Submit Feedback")

# Preview
with st.expander("Preview what will be uploaded to Langfuse", expanded=False):
    preview_scores = {
        "Human Overall (1-5)": rating_overall,
        "Human Answer Quality (1-5)": rating_answer,
        "Human Search Quality (1-5)": rating_search,
        "Human Evidence Quality (1-5)": rating_evidence,
        "Human Reasoning Quality (1-5)": rating_reasoning,
        "Human Efficiency (1-5)": rating_efficiency,
        "Human Rating (0-1)": round(rating_overall / 5.0, 2),
    }
    for name, val in preview_scores.items():
        st.markdown(f"- **{name}**: `{val}`")
    if selected_failures:
        st.markdown(f"- **Failure tags**: {selected_failures}")
    if corrected_answer:
        st.markdown(f"- **Correction**: `{corrected_answer}`")
    if quality_note:
        st.markdown(f"- **Notes**: {quality_note[:100]}...")

# Buttons
btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1])
with btn_col1:
    submit_full = st.button(
        "📤 Submit Full Feedback", type="primary", use_container_width=True
    )
with btn_col2:
    quick_good = st.button("👍 Quick Good", use_container_width=True)
with btn_col3:
    quick_bad = st.button("👎 Quick Bad", use_container_width=True)

# ── Quick submit handlers ──────────────────────────────────────────────────────

if quick_good:
    langfuse.create_score(
        name="Human Rating",
        value=1.0,
        trace_id=trace_id,
        comment="Quick thumbs up from feedback UI",
        data_type="NUMERIC",
    )
    langfuse.flush()
    st.success("👍 Quick thumbs-up sent to Langfuse.")
    st.caption(f"Trace: `{trace_id}`")

if quick_bad:
    langfuse.create_score(
        name="Human Rating",
        value=0.0,
        trace_id=trace_id,
        comment="Quick thumbs down from feedback UI",
        data_type="NUMERIC",
    )
    langfuse.flush()
    st.error("👎 Quick thumbs-down sent to Langfuse.")
    st.caption(f"Trace: `{trace_id}`")

# ── Full submit handler ────────────────────────────────────────────────────────

if submit_full:
    # Build structured comment
    comment_parts = []
    if selected_failures:
        comment_parts.append("Failures: [" + " | ".join(selected_failures) + "]")
    if corrected_answer:
        comment_parts.append(f"Correction: '{corrected_answer}'")
    if quality_note:
        comment_parts.append(f"Notes: {quality_note.strip()}")
    non_neutral_steps = {
        k: v for k, v in step_annotations.items() if "neutral" not in v
    }
    if non_neutral_steps:
        step_str = "; ".join(f"step {k}={v}" for k, v in non_neutral_steps.items())
        comment_parts.append(f"Step annotations: [{step_str}]")
    if not comment_parts:
        comment_parts = ["No additional notes"]

    full_comment = " | ".join(comment_parts)

    # Upload 7 named scores
    upload_scores = [
        ("Human Overall", float(rating_overall)),
        ("Human Answer Quality", float(rating_answer)),
        ("Human Search Quality", float(rating_search)),
        ("Human Evidence Quality", float(rating_evidence)),
        ("Human Reasoning Quality", float(rating_reasoning)),
        ("Human Efficiency", float(rating_efficiency)),
        ("Human Rating", round(rating_overall / 5.0, 2)),  # backward-compat 0-1
    ]

    with st.spinner("Uploading to Langfuse..."):
        for score_name, score_val in upload_scores:
            langfuse.create_score(
                name=score_name,
                value=score_val,
                trace_id=trace_id,
                comment=full_comment[:1000],
                data_type="NUMERIC",
            )
        langfuse.flush()

    st.success(f"✅ {len(upload_scores)} scores submitted to Langfuse!")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("**Scores uploaded:**")
        for name, val in upload_scores:
            st.markdown(f"- `{name}`: **{val}**")
    with res_col2:
        st.markdown("**Comment (attached to all scores):**")
        st.code(full_comment[:500], language=None)

    if corrected_answer and corrected_answer.lower().strip() != gold_answer.lower().strip():
        st.info(
            f"📝 Note: your correction `{corrected_answer}` differs from the gold answer "
            f"`{gold_answer}`. This has been logged in the comment."
        )

    # Store feedback summary in session state for cross-page continuity
    st.session_state.last_feedback = {
        "trace_id": trace_id,
        "question_idx": idx,
        "overall": rating_overall,
        "failures": selected_failures,
        "correction": corrected_answer,
    }

st.divider()
st.caption(
    f"Reviewing trace `{trace_id}` | "
    f"Question [{idx}]: {item['input']['question'][:60]}..."
)
