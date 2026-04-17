"""Dataset Explorer: browse questions, gold answers, gold trajectories, context paragraphs."""

import json

import streamlit as st

st.set_page_config(page_title="Dataset Explorer", page_icon="📚", layout="wide")
st.title("📚 Dataset Explorer")
st.caption("Browse the 10 HotpotQA questions, their gold answers, expected tool sequences, and context paragraphs.")


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

selected = st.selectbox("Select a question", options, help="Pick a question to explore its details below.")
idx = int(selected.split("]")[0].replace("[", ""))
item = dataset[idx]
st.session_state.selected_idx = idx

# ── Question Card ────────────────────────────────────────────────────────────

with st.container(border=True):
    st.markdown(f"### {item['input']['question']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gold Answer", item["expected_output"]["answer"])
    with col2:
        st.metric("Type", item["metadata"]["type"].capitalize())
    with col3:
        traj = item["expected_output"]["trajectory"]
        st.metric("Tool Calls Expected", len(traj["actions"]))

# ── Quick Action ─────────────────────────────────────────────────────────────

st.page_link(
    "pages/2_Run_and_Evaluate.py",
    label="Run the agent on this question →",
    icon="🤖",
)

# ── Gold Trajectory ───────────────────────────────────────────────────────────

st.header("Gold Trajectory")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"**Reasoning Type**: `{traj['reasoning_type']}`")
    st.markdown(f"**Expected Paragraphs**: {', '.join(traj['expected_paragraphs'])}")

with col2:
    st.markdown(f"**Strategy**: {traj['reasoning_description']}")

st.subheader("Ideal Action Sequence")
for i, action in enumerate(traj["actions"]):
    tool = action["tool"]
    inp = json.dumps(action["input"])
    title = action.get("expected_result_title", "")
    icon = "🔎" if tool == "search_paragraphs" else "📖"
    suffix = f" → expects: *{title}*" if title else ""
    st.markdown(f"{icon} **Step {i+1}**: `{tool}({inp})`{suffix}")

# ── Context Paragraphs ────────────────────────────────────────────────────────

st.header("Context Paragraphs")
st.caption("10 paragraphs provided to the agent. Gold paragraphs are highlighted.")

gold_titles = set(traj["expected_paragraphs"])

# Show gold paragraphs first, then distractors
gold_paragraphs = [(t, s) for t, s in item["input"]["context"] if t in gold_titles]
distractor_paragraphs = [(t, s) for t, s in item["input"]["context"] if t not in gold_titles]

if gold_paragraphs:
    st.subheader(f"Gold Paragraphs ({len(gold_paragraphs)})")
    for title, sentences in gold_paragraphs:
        with st.expander(f"⭐ {title}", expanded=True):
            st.markdown(" ".join(sentences))

if distractor_paragraphs:
    st.subheader(f"Distractor Paragraphs ({len(distractor_paragraphs)})")
    for title, sentences in distractor_paragraphs:
        with st.expander(title, expanded=False):
            st.markdown(" ".join(sentences))
