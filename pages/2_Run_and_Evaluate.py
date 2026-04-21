"""Run & Evaluate (Deterministic): run agent on a question, compute Answer F1, Retrieval F1,
Action Order, and Efficiency — the 4 deterministic metrics."""

import json
import os
import time
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
)

load_dotenv()

st.set_page_config(page_title="Run & Evaluate", page_icon="🤖", layout="wide")
st.title("🤖 Run & Evaluate (Deterministic Metrics)")
st.markdown(
    "Run the agent on a question and score it on the **4 deterministic metrics**: "
    "Answer F1, Retrieval F1, Action Order, and Efficiency. "
    "For the full 9-metric evaluation, use **LLM-as-Judge Run & Evaluate** in the sidebar."
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

def display_results(result, item, all_scores, trace_id):
    """Display the results: Answer Comparison + Trajectory Evaluation (4 deterministic metrics)."""
    gold_answer = item["expected_output"]["answer"]
    gold_trajectory = item["expected_output"]["trajectory"]

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Deterministic Metrics Overview
    # ══════════════════════════════════════════════════════════════════════

    st.header("📊 Deterministic Metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📝 Answer F1", f"{all_scores['answer_f1'] * 100:.1f}%")
        st.caption("Token-level F1 vs gold answer")
    with c2:
        st.metric("🔍 Retrieval F1", f"{all_scores['retrieval_f1'] * 100:.1f}%")
        st.caption("Found the right paragraphs?")
    with c3:
        st.metric("🔀 Action Order", f"{all_scores['action_order'] * 100:.1f}%")
        st.caption("Correct tool call sequence (LCS)")
    with c4:
        st.metric("⚡ Efficiency", f"{all_scores['efficiency'] * 100:.1f}%")
        st.caption("Minimum calls needed?")

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
    # Agent Trajectory
    # ══════════════════════════════════════════════════════════════════════

    st.header("🔄 Agent Trajectory")
    st.caption("Full tool-call log.")

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
                label = f"Read: `{inp.get('title', '')}`"

            with st.expander(f"{icon} Step {i+1} — {label}", expanded=False):
                st.code(output, language=None)

    # ══════════════════════════════════════════════════════════════════════
    # Trajectory Evaluation
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

    st.divider()
    st.caption(f"Trace: `{trace_id}` | Scores uploaded to Langfuse.")


# ── Run Agent ─────────────────────────────────────────────────────────────────

if st.button("▶️ Run Agent & Evaluate", type="primary", use_container_width=True):
    gold_answer = item["expected_output"]["answer"]
    gold_trajectory = item["expected_output"]["trajectory"]

    with st.status("Running agent...", expanded=True) as status:
        st.write("🤖 Sending question to agent...")

        # Create Langfuse trace
        experiment_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        lf_dataset = langfuse.get_dataset("DeepEval-hotpotqa-20")

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
        st.write("📊 Computing scores...")

        # ── Layer 1 ──
        answer_em = exact_match(result["answer"], gold_answer)
        answer_f1 = f1_score(result["answer"], gold_answer)

        # ── Layer 2 ──
        retrieval = paragraph_retrieval_scores(result["trajectory"], gold_trajectory)
        action_score = action_order_score(result["trajectory"], gold_trajectory)
        efficiency = search_efficiency(result["trajectory"], gold_trajectory)
        traj_score = trajectory_composite(retrieval["retrieval_f1"], action_score, efficiency)

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

        langfuse_scores = {
            "Answer Accuracy": answer_f1,
            "Evidence Quality": traj_score,
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

    display_results(result, item, all_scores, trace_id)

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
        st.session_state.trace_id,
    )
else:
    st.info("Select a question and click '▶️ Run Agent & Evaluate' to start.")
