"""Dataset Explorer: browse questions, gold answers, gold trajectories, context paragraphs."""

import json
import os

import streamlit as st

st.set_page_config(page_title="Dataset Explorer", page_icon="📚", layout="wide")
st.title("📚 Dataset Explorer")


@st.cache_data
def load_dataset():
    # Include file mtime in cache key so edits to the file bust the cache
    mtime = os.path.getmtime("dataset_local.json")
    with open("dataset_local.json") as f:
        return json.load(f), mtime


dataset, _mtime = load_dataset()

# ── Dataset Summary ───────────────────────────────────────────────────────────

with st.container(border=True):
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    n_bridge = sum(1 for d in dataset if d["metadata"]["type"] == "bridge")
    n_comparison = len(dataset) - n_bridge
    n_easy = sum(1 for d in dataset if d["metadata"]["level"] == "easy")
    n_medium = sum(1 for d in dataset if d["metadata"]["level"] == "medium")
    n_hard = sum(1 for d in dataset if d["metadata"]["level"] == "hard")
    s1.metric("Total Questions", len(dataset))
    s2.metric("Bridge", n_bridge)
    s3.metric("Comparison", n_comparison)
    s4.metric("Easy", n_easy)
    s5.metric("Medium", n_medium)
    s6.metric("Hard", n_hard)

# ── Question Selector ─────────────────────────────────────────────────────────

options = []
for i, item in enumerate(dataset):
    qtype = item["metadata"]["type"]
    level = item["metadata"]["level"]
    q = item["input"]["question"][:70]
    options.append(f"[{i}] ({qtype}/{level}) {q}")

selected = st.selectbox("Select a question", options)
idx = int(selected.split("]")[0].replace("[", ""))
item = dataset[idx]
st.session_state.selected_idx = idx

# ── Question & Gold Answer ────────────────────────────────────────────────────

st.header("Question")
st.markdown(f"**{item['input']['question']}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Gold Answer", item["expected_output"]["answer"])
with col2:
    st.metric("Type", item["metadata"]["type"].capitalize())
with col3:
    st.metric("Difficulty", item["metadata"]["level"].capitalize())

# ── Gold Trajectory ───────────────────────────────────────────────────────────

st.header("Gold Trajectory")

traj = item["expected_output"]["trajectory"]
st.markdown(f"**Reasoning Type**: {traj['reasoning_type']}")
st.markdown(f"**Strategy**: {traj['reasoning_description']}")
st.markdown(f"**Expected Paragraphs**: {', '.join(traj['expected_paragraphs'])}")

st.subheader("Ideal Action Sequence")
for i, action in enumerate(traj["actions"]):
    tool = action["tool"]
    inp = json.dumps(action["input"])
    title = action.get("expected_result_title", "")
    suffix = f" &rarr; expects: *{title}*" if title else ""
    st.markdown(f"{i+1}. `{tool}({inp})`{suffix}")

# ── Context Paragraphs ────────────────────────────────────────────────────────

st.header("Context Paragraphs")
n_ctx = len(item["input"]["context"])
gold_titles = set(traj["expected_paragraphs"])
n_gold = sum(1 for t, _ in item["input"]["context"] if t in gold_titles)
st.caption(
    f"{n_ctx} paragraphs provided to the agent. "
    f"{n_gold} are gold (relevant), {n_ctx - n_gold} are distractors."
)

for title, sentences in item["input"]["context"]:
    is_gold = title in gold_titles
    label = f"{'⭐ ' if is_gold else ''}{title}"
    with st.expander(label, expanded=is_gold):
        for j, sent in enumerate(sentences):
            st.markdown(f"{j}. {sent}")
