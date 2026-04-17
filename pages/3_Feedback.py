"""Feedback page: rate agent responses with thumbs up/down, sent to Langfuse."""

import json
import os

import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

st.set_page_config(page_title="Feedback", page_icon="👍", layout="wide")
st.title("👍 Give Feedback")


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

# ── Check for agent run ──────────────────────────────────────────────────────

if "agent_result" not in st.session_state or "trace_id" not in st.session_state:
    st.warning("No agent run found. Go to **Run & Evaluate** first.")
    st.page_link("pages/2_Run_and_Evaluate.py", label="Go to Run & Evaluate →", icon="🤖")
    st.stop()

result = st.session_state.agent_result
trace_id = st.session_state.trace_id
idx = st.session_state.get("selected_idx", 0)
item = dataset[idx]
all_scores = st.session_state.get("all_scores", {})
judge_scores = st.session_state.get("judge_scores", {})

# ── Run Summary (compact) ───────────────────────────────────────────────────

with st.container(border=True):
    st.markdown(f"**Question**: {item['input']['question']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Agent**: {result['answer']}")
    with col2:
        st.markdown(f"**Gold**: {item['expected_output']['answer']}")
    with col3:
        if all_scores:
            reasoning_score = sum(s / 5.0 for s in judge_scores.values()) / 3.0 if judge_scores else 0
            composite = 0.4 * all_scores.get("answer_f1", 0) + 0.35 * all_scores.get("trajectory_score", 0) + 0.25 * reasoning_score
            st.metric("Composite Score", f"{composite * 100:.1f}%")

    with st.expander("Agent Trajectory", expanded=False):
        for i, step in enumerate(result["trajectory"]):
            tool = step["tool"]
            icon = "🔎" if tool == "search_paragraphs" else "📖"
            inp_str = step["input"].get("query", "") or step["input"].get("title", "")
            st.markdown(f"{icon} {i+1}. `{tool}` — {inp_str}")

# ── Feedback ─────────────────────────────────────────────────────────────────

st.markdown("### Rate this response")

comment = st.text_input("Optional comment", placeholder="e.g. 'Good reasoning but answer too verbose'")

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Good response", type="primary", use_container_width=True):
        langfuse.create_score(
            name="Human Rating",
            value=1.0,
            trace_id=trace_id,
            comment=comment or "Thumbs up from demo UI",
            data_type="NUMERIC",
        )
        langfuse.flush()
        st.success("Thumbs up sent to Langfuse!")

with col2:
    if st.button("👎 Bad response", use_container_width=True):
        langfuse.create_score(
            name="Human Rating",
            value=0.0,
            trace_id=trace_id,
            comment=comment or "Thumbs down from demo UI",
            data_type="NUMERIC",
        )
        langfuse.flush()
        st.success("Thumbs down sent to Langfuse!")

st.caption(f"Trace: `{trace_id}`")
