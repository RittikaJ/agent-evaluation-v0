"""Dataset Explorer: browse questions, gold answers, gold trajectories, context paragraphs."""

import json

import streamlit as st

st.set_page_config(page_title="Dataset Explorer", page_icon="📚", layout="wide")
st.title("Dataset Explorer")


@st.cache_data
def load_dataset():
    with open("dataset_local.json") as f:
        return json.load(f)


dataset = load_dataset()

# ── Question Selector ─────────────────────────────────────────────────────────

options = []
for i, item in enumerate(dataset):
    qtype = item["metadata"]["type"]
    q = item["input"]["question"][:80]
    options.append(f"[{i}] ({qtype}) {q}")

selected = st.selectbox("Select a question", options)
idx = int(selected.split("]")[0].replace("[", ""))
item = dataset[idx]
st.session_state.selected_idx = idx

# ── Question & Gold Answer ────────────────────────────────────────────────────

st.header("Question")
st.markdown(f"**{item['input']['question']}**")

col1, col2 = st.columns(2)
with col1:
    st.metric("Gold Answer", item["expected_output"]["answer"])
with col2:
    st.metric("Type", item["metadata"]["type"].capitalize())

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
st.caption("10 paragraphs provided to the agent. Gold paragraphs are highlighted.")

gold_titles = set(traj["expected_paragraphs"])

for title, sentences in item["input"]["context"]:
    is_gold = title in gold_titles
    label = f"{'⭐ ' if is_gold else ''}{title}"
    with st.expander(label, expanded=is_gold):
        for j, sent in enumerate(sentences):
            st.markdown(f"{j}. {sent}")
