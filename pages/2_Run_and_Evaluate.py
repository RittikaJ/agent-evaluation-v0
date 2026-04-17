"""Run & Evaluate: run agent on a question, show tool calls live, compute all 3 layers, upload to Langfuse."""

import json
import os
from datetime import datetime

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
st.caption("Pick a question, run the agent live, and see how it scores across answer accuracy, trajectory quality, and reasoning.")

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
    col3.markdown(f"**Expected Paragraphs**: {', '.join(item['expected_output']['trajectory']['expected_paragraphs'])}")


def display_results(result, item, all_scores, judge_scores, trace_id):
    """Display the full results for a completed agent run."""
    gold_answer = item["expected_output"]["answer"]
    gold_trajectory = item["expected_output"]["trajectory"]
    reasoning_score = sum(s / 5.0 for s in judge_scores.values()) / 3.0
    composite = 0.4 * all_scores["answer_f1"] + 0.35 * all_scores["trajectory_score"] + 0.25 * reasoning_score

    # ══════════════════════════════════════════════════════════════════════
    # Composite at the top
    # ══════════════════════════════════════════════════════════════════════

    st.divider()

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Answer (40%)", f"{all_scores['answer_f1'] * 100:.1f}%")
            st.caption("How correct is the answer?")
        with col2:
            st.metric("🔍 Trajectory (35%)", f"{all_scores['trajectory_score'] * 100:.1f}%")
            st.caption("Did it find the right evidence?")
        with col3:
            st.metric("🧠 Reasoning (25%)", f"{reasoning_score * 100:.1f}%")
            st.caption("Does the reasoning make sense?")
        with col4:
            st.metric("🎯 Composite", f"{composite * 100:.1f}%")
            st.caption("Overall weighted score")

    # ══════════════════════════════════════════════════════════════════════
    # Answer comparison
    # ══════════════════════════════════════════════════════════════════════

    st.header("Answer Comparison")

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
            "Low F1 — the agent likely gave a verbose answer. "
            "The gold answer is usually 1-3 words."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Agent Trajectory
    # ══════════════════════════════════════════════════════════════════════

    st.header("Agent Trajectory")

    if not result["trajectory"]:
        st.warning("The agent made no tool calls.")
    else:
        for i, step in enumerate(result["trajectory"]):
            tool = step["tool"]
            inp = step["input"]
            output = str(step["output"])

            # Color-code by tool type
            if tool == "search_paragraphs":
                icon = "🔎"
                label = f"Search: `{inp.get('query', '')}`"
            else:
                icon = "📖"
                label = f"Read: `{inp.get('title', '')}`"

            with st.expander(f"{icon} Step {i+1} — {label}", expanded=True):
                st.code(output, language=None)

    # ══════════════════════════════════════════════════════════════════════
    # Trajectory Evaluation
    # ══════════════════════════════════════════════════════════════════════

    st.header("Trajectory Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Retrieval F1", f"{all_scores['retrieval_f1'] * 100:.0f}%")
        st.caption("Did it find the right paragraphs?")

    with col2:
        st.metric("Action Order (LCS)", f"{all_scores['action_order'] * 100:.0f}%")
        st.caption("Did it follow the right tool sequence?")

    with col3:
        st.metric("Efficiency", f"{all_scores['efficiency'] * 100:.0f}%")
        st.caption("Did it use the minimum tool calls needed?")

    with col4:
        st.metric("Trajectory Score", f"{all_scores['trajectory_score'] * 100:.0f}%")
        st.caption("Weighted combo: 40% retrieval + 40% order + 20% efficiency")

    # Evidence retrieval check
    st.subheader("Evidence Retrieval")

    gold_titles = set(gold_trajectory["expected_paragraphs"])
    retrieved_titles = set()
    for step in result["trajectory"]:
        if step["tool"] == "read_paragraph":
            retrieved_titles.add(step["input"].get("title", ""))
        elif step["tool"] == "search_paragraphs":
            try:
                results_parsed = json.loads(step["output"]) if isinstance(step["output"], str) else step["output"]
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
    # Reasoning Scores
    # ══════════════════════════════════════════════════════════════════════

    st.header("Reasoning Evaluation (LLM Judge)")

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

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.caption(f"Tokens: {result['token_usage']} | Trace: `{trace_id}` | Scores uploaded to Langfuse.")
    with col_right:
        st.page_link("pages/3_Feedback.py", label="Rate this response →", icon="👍")


# ── Run Agent ─────────────────────────────────────────────────────────────────

if st.button("Run Agent & Evaluate", type="primary", use_container_width=True):
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

        with lf_item.run(
            run_name=experiment_name,
            run_metadata={"source": "demo", "question_type": item["metadata"]["type"]},
        ) as span:
            result = run_agent(item["input"]["question"], item["input"]["context"])
            trace_id = span.trace_id

            # Show tool calls inside the status
            st.write(f"✅ Agent made **{len(result['trajectory'])} tool calls**:")
            for i, step in enumerate(result["trajectory"]):
                tool = step["tool"]
                if tool == "search_paragraphs":
                    st.write(f"  🔎 {i+1}. Search: `{step['input'].get('query', '')}`")
                else:
                    st.write(f"  📖 {i+1}. Read: `{step['input'].get('title', '')}`")
            st.write(f"💬 Agent answered: **{result['answer']}**")

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

            # ── Layer 3 ── (inside the span so judge generations nest under this trace)
            st.write("🧠 Running LLM judge (3 API calls)...")
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

            span.update(
                input=item["input"]["question"],
                output=result["answer"],
                metadata={
                    "trajectory_length": len(result["trajectory"]),
                    "token_usage": result["token_usage"],
                },
            )

        # ── Compute composite ──
        reasoning_score_val = sum(s / 5.0 for s in judge_scores.values()) / 3.0
        composite_val = 0.4 * answer_f1 + 0.35 * traj_score + 0.25 * reasoning_score_val

        # ── Upload to Langfuse (7 clean scores) ──
        st.write("📤 Uploading scores to Langfuse...")

        # All scores kept locally for display
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

        # Only upload business-friendly scores to Langfuse
        langfuse_scores = {
            "Answer Accuracy": answer_f1,
            "Evidence Quality": traj_score,
            "Groundedness": judge_scores["groundedness"],
            "Reasoning Quality": judge_scores["reasoning_coherence"],
            "Search Quality": judge_scores["search_strategy"],
            "Overall Score": composite_val,
        }

        for score_name, score_val in langfuse_scores.items():
            langfuse.create_score(
                name=score_name,
                value=score_val,
                trace_id=trace_id,
                data_type="NUMERIC",
            )
        langfuse.flush()

        status.update(label="✅ Done — agent run scored and uploaded to Langfuse", state="complete")

    # Store in session state so results persist across tab switches
    st.session_state.agent_result = result
    st.session_state.trace_id = trace_id
    st.session_state.selected_idx = idx
    st.session_state.all_scores = all_scores
    st.session_state.judge_scores = judge_scores

    display_results(result, item, all_scores, judge_scores, trace_id)

# ── Show Previous Run (persisted across tab switches) ─────────────────────────

elif "agent_result" in st.session_state and "all_scores" in st.session_state:
    prev_idx = st.session_state.get("selected_idx", 0)
    prev_item = dataset[prev_idx]

    if prev_idx != idx:
        st.info(f"Showing results from previous run on question [{prev_idx}]. Select that question or click 'Run Agent & Evaluate' to run on this one.")

    display_results(
        st.session_state.agent_result,
        prev_item,
        st.session_state.all_scores,
        st.session_state.judge_scores,
        st.session_state.trace_id,
    )
else:
    st.info("👆 Select a question above and click **Run Agent & Evaluate** to see results here.")
