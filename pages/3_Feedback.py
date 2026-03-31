"""Feedback page: rate agent responses with thumbs up/down, sent to Langfuse."""

import json
import os

import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

st.set_page_config(page_title="Feedback", page_icon="👍", layout="wide")
st.title("Give Feedback")


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
    st.warning("No agent run found. Go to **Run & Evaluate** page first.")
    st.stop()

result = st.session_state.agent_result
trace_id = st.session_state.trace_id
idx = st.session_state.get("selected_idx", 0)
item = dataset[idx]

# ── Show the run ──────────────────────────────────────────────────────────────

st.markdown(f"**Question**: {item['input']['question']}")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Agent Answer**: {result['answer']}")
with col2:
    st.markdown(f"**Gold Answer**: {item['expected_output']['answer']}")

with st.expander("Agent Trajectory", expanded=False):
    for i, step in enumerate(result["trajectory"]):
        st.markdown(f"{i+1}. `{step['tool']}({json.dumps(step['input'])})`")

# ── Feedback Buttons ──────────────────────────────────────────────────────────

st.header("Rate this response")

col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Good response", type="primary", use_container_width=True):
        langfuse.create_score(
            name="Human Rating",
            value=1.0,
            trace_id=trace_id,
            comment="Thumbs up from demo UI",
            data_type="NUMERIC",
        )
        langfuse.flush()
        st.success("Feedback submitted: thumbs up! Score sent to Langfuse.")
        st.markdown("**Next**: Go to **Run & Evaluate** to try another question, or check the Langfuse dashboard to see this feedback score on the trace.")

with col2:
    if st.button("👎 Bad response", use_container_width=True):
        langfuse.create_score(
            name="Human Rating",
            value=0.0,
            trace_id=trace_id,
            comment="Thumbs down from demo UI",
            data_type="NUMERIC",
        )
        langfuse.flush()
        st.success("Feedback submitted: thumbs down! Score sent to Langfuse.")
        st.markdown("**Next**: Go to **Run & Evaluate** to try another question, or check the Langfuse dashboard to see this feedback score on the trace.")

st.caption(f"Feedback linked to trace: `{trace_id}`")
