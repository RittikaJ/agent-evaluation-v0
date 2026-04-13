"""
Failure Analysis: batch-run the agent and quantify failure mode distribution.

Answers: of all cases where the agent failed, what fraction are
  • Retrieval failures  — couldn't find the right evidence
  • Synthesis failures  — found evidence but drew the wrong conclusion
  • Planning failures   — evidence found but reasoning chain was broken
  • Partial failures    — answer almost correct / off by a small amount
"""

import json
import time
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import run_agent
from evaluate import (
    f1_score,
    paragraph_retrieval_scores,
    action_order_score,
    search_efficiency,
    trajectory_composite,
    judge_reasoning,
    load_rubric,
)

load_dotenv()

st.set_page_config(page_title="Failure Analysis", page_icon="🔬", layout="wide")
st.title("🔬 Failure Mode Distribution")
st.markdown(
    "Run the agent on all dataset questions, classify every failure, and quantify "
    "the distribution — retrieval vs synthesis vs planning. "
    "Answers: *why* does the agent fail, and how often for each root cause?"
)
st.divider()


@st.cache_data
def load_dataset():
    with open("dataset_local.json") as f:
        return json.load(f)


_FAILURE_MODES = {
    "success":   {"emoji": "✅", "label": "Success"},
    "retrieval": {"emoji": "🔍", "label": "Retrieval Failure"},
    "synthesis": {"emoji": "⚗️", "label": "Synthesis Failure"},
    "planning":  {"emoji": "🗺️", "label": "Planning Failure"},
    "partial":   {"emoji": "⚠️", "label": "Partial Failure"},
}


def classify_failure(answer_f1, retrieval_f1, groundedness=None, reasoning_coh=None):
    if answer_f1 >= 0.70:
        return "success"
    retrieval_ok = retrieval_f1 >= 0.50
    if groundedness is not None and reasoning_coh is not None:
        grounded = groundedness >= 3
        coherent = reasoning_coh >= 3
        if not retrieval_ok:
            return "retrieval"
        elif retrieval_ok and not grounded:
            return "synthesis"
        elif not coherent:
            return "planning"
        else:
            return "partial"
    else:
        if not retrieval_ok:
            return "retrieval"
        elif answer_f1 < 0.30:
            return "synthesis"
        else:
            return "partial"


dataset = load_dataset()

with st.sidebar:
    st.header("⚙️ Options")
    run_judge = st.toggle(
        "Run LLM judge (slower, more accurate)",
        value=False,
        help="Adds 2 judge calls per question for groundedness + reasoning coherence.",
    )
    st.caption(f"Dataset: {len(dataset)} questions")

if st.button("▶️ Run Failure Analysis", type="primary", use_container_width=True):
    rubrics = {}
    if run_judge:
        rubrics = {
            "groundedness": load_rubric("groundedness"),
            "reasoning_coherence": load_rubric("reasoning_coherence"),
        }

    rows = []
    progress = st.progress(0, text="Starting…")
    log = st.empty()

    for i, item in enumerate(dataset):
        question = item["input"]["question"]
        context = item["input"]["context"]
        gold_answer = item["expected_output"]["answer"]
        gold_traj = item["expected_output"]["trajectory"]
        qtype = item["metadata"].get("type", "?")
        level = item["metadata"].get("level", "?")

        progress.progress(i / len(dataset), text=f"Running [{i+1}/{len(dataset)}]…")
        log.markdown(f"⏳ `{question[:70]}...`")

        t0 = time.time()
        result = run_agent(question, context)
        latency = time.time() - t0

        answer_f1 = f1_score(result["answer"], gold_answer)
        retrieval = paragraph_retrieval_scores(result["trajectory"], gold_traj)
        action_score = action_order_score(result["trajectory"], gold_traj)
        efficiency = search_efficiency(result["trajectory"], gold_traj)
        traj_score = trajectory_composite(retrieval["retrieval_f1"], action_score, efficiency)

        groundedness = None
        reasoning_coh = None
        if run_judge:
            groundedness = judge_reasoning(
                question, result["answer"], gold_answer,
                result["trajectory"], "groundedness", rubrics["groundedness"],
            )
            reasoning_coh = judge_reasoning(
                question, result["answer"], gold_answer,
                result["trajectory"], "reasoning_coherence", rubrics["reasoning_coherence"],
            )

        failure_mode = classify_failure(
            answer_f1, retrieval["retrieval_f1"], groundedness, reasoning_coh
        )

        rows.append({
            "id": item["id"],
            "question": question[:80],
            "gold_answer": gold_answer,
            "agent_answer": result["answer"][:60],
            "type": qtype,
            "level": level,
            "answer_f1": answer_f1,
            "retrieval_f1": retrieval["retrieval_f1"],
            "trajectory_score": traj_score,
            "groundedness": groundedness,
            "reasoning_coherence": reasoning_coh,
            "failure_mode": failure_mode,
            "n_steps": len(result["trajectory"]),
            "latency_s": round(latency, 2),
        })

    progress.progress(1.0, text="✅ Done")
    log.empty()
    st.session_state.failure_rows = rows
    st.success(f"Evaluated {len(rows)} questions.")

if "failure_rows" not in st.session_state:
    st.info("Click **▶️ Run Failure Analysis** to evaluate all questions.")
    st.stop()

rows = st.session_state.failure_rows
n = len(rows)
mode_counts = Counter(r["failure_mode"] for r in rows)
failures_only = [r for r in rows if r["failure_mode"] != "success"]
n_fail = len(failures_only)
avg_f1 = sum(r["answer_f1"] for r in rows) / n
fail_modes = ["retrieval", "synthesis", "planning", "partial"]

# ── Summary ───────────────────────────────────────────────────────────────────
st.header("📊 Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Questions", n)
c2.metric("Successes", mode_counts.get("success", 0),
          delta=f"{mode_counts.get('success', 0)/n*100:.0f}%")
c3.metric("Failures", n_fail, delta=f"{n_fail/n*100:.0f}%", delta_color="inverse")
c4.metric("Avg Answer F1", f"{avg_f1:.3f}")

# ── Distribution ──────────────────────────────────────────────────────────────
st.header("🥧 Failure Mode Distribution")
st.caption(
    "Among **failing** cases only — what is the dominant root cause? "
    "This directly guides where to invest improvement effort."
)

if n_fail == 0:
    st.success("🎉 No failures — all questions answered correctly!")
else:
    cols = st.columns(4)
    for idx, mode in enumerate(fail_modes):
        cnt = mode_counts.get(mode, 0)
        pct = cnt / n_fail * 100 if n_fail else 0
        info = _FAILURE_MODES[mode]
        with cols[idx]:
            with st.container(border=True):
                st.metric(f"{info['emoji']} {info['label']}", f"{cnt}/{n_fail}",
                          delta=f"{pct:.0f}% of failures")

    dist_data = {
        f"{_FAILURE_MODES[m]['emoji']} {_FAILURE_MODES[m]['label']}": mode_counts.get(m, 0)
        for m in fail_modes if mode_counts.get(m, 0) > 0
    }
    if dist_data:
        st.bar_chart(
            pd.DataFrame({"Failure Count": list(dist_data.values())},
                         index=list(dist_data.keys())),
            horizontal=True,
        )

    # Dominant insight
    dominant = max(fail_modes, key=lambda m: mode_counts.get(m, 0))
    dom_pct = mode_counts.get(dominant, 0) / n_fail * 100

    st.markdown("---")
    insights = {
        "retrieval": (
            st.error,
            f"**🔍 Dominant failure: Retrieval ({dom_pct:.0f}% of failures)**  \n"
            "The agent cannot consistently find the required evidence paragraphs. "
            "Improve search query decomposition — break multi-hop questions into "
            "targeted entity-level queries."
        ),
        "synthesis": (
            st.warning,
            f"**⚗️ Dominant failure: Synthesis ({dom_pct:.0f}% of failures)**  \n"
            "The agent retrieves relevant paragraphs but fails to produce a correct "
            "concise answer. Improve final answer extraction in the system prompt."
        ),
        "planning": (
            st.warning,
            f"**🗺️ Dominant failure: Planning ({dom_pct:.0f}% of failures)**  \n"
            "The agent has partial evidence but the multi-hop chain is broken. "
            "Explicitly instruct the agent to use hop-1 output to form the hop-2 query."
        ),
        "partial": (
            st.info,
            f"**⚠️ Dominant failure: Partial ({dom_pct:.0f}% of failures)**  \n"
            "Answers are almost correct but slightly off. Focus on conciseness and "
            "answer normalization."
        ),
    }
    fn, msg = insights.get(dominant, (st.info, "No dominant failure pattern."))
    fn(msg)

# ── Breakdown by type / level ─────────────────────────────────────────────────
st.header("🔍 Breakdown by Type & Difficulty")
col1, col2 = st.columns(2)

with col1:
    st.subheader("By Question Type")
    for qtype in sorted({r["type"] for r in rows}):
        tr = [r for r in rows if r["type"] == qtype]
        tf = [r for r in tr if r["failure_mode"] != "success"]
        tm = Counter(r["failure_mode"] for r in tf)
        with st.container(border=True):
            st.markdown(f"**{qtype.title()}** — {len(tr)} Qs, {len(tf)} failures")
            for mode in fail_modes:
                cnt = tm.get(mode, 0)
                if cnt:
                    info = _FAILURE_MODES[mode]
                    st.markdown(f"  {info['emoji']} {info['label']}: **{cnt}** ({cnt/len(tf)*100:.0f}%)")

with col2:
    st.subheader("By Difficulty Level")
    for level in ["easy", "medium", "hard"]:
        lr = [r for r in rows if r["level"] == level]
        if not lr:
            continue
        lf = [r for r in lr if r["failure_mode"] != "success"]
        lm = Counter(r["failure_mode"] for r in lf)
        with st.container(border=True):
            st.markdown(
                f"**{level.title()}** — {len(lr)} Qs, "
                f"{len(lf)} failures ({len(lf)/len(lr)*100:.0f}%)"
            )
            for mode in fail_modes:
                cnt = lm.get(mode, 0)
                if cnt and lf:
                    info = _FAILURE_MODES[mode]
                    st.markdown(f"  {info['emoji']} {info['label']}: **{cnt}** ({cnt/len(lf)*100:.0f}%)")

# ── Per-question table ────────────────────────────────────────────────────────
st.header("📋 Per-Question Results")
filter_opt = st.selectbox(
    "Filter by failure mode",
    ["All"] + [f"{v['emoji']} {v['label']}" for v in _FAILURE_MODES.values()],
)

disp = rows
if filter_opt != "All":
    for mk, mv in _FAILURE_MODES.items():
        if f"{mv['emoji']} {mv['label']}" == filter_opt:
            disp = [r for r in rows if r["failure_mode"] == mk]
            break

table = []
for r in disp:
    info = _FAILURE_MODES.get(r["failure_mode"], _FAILURE_MODES["partial"])
    table.append({
        "Mode": f"{info['emoji']} {info['label']}",
        "Question": r["question"],
        "Gold": r["gold_answer"],
        "Agent": r["agent_answer"],
        "F1": f"{r['answer_f1']:.2f}",
        "Ret F1": f"{r['retrieval_f1']:.2f}",
        "Traj": f"{r['trajectory_score']:.2f}",
        "Type": r["type"],
        "Level": r["level"],
        "Steps": r["n_steps"],
    })

if table:
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# ── Score distribution chart ──────────────────────────────────────────────────
st.header("📈 Score Distribution Across All Questions")
score_df = pd.DataFrame({
    "Answer F1":    [r["answer_f1"] for r in rows],
    "Retrieval F1": [r["retrieval_f1"] for r in rows],
    "Traj Score":   [r["trajectory_score"] for r in rows],
})
st.bar_chart(score_df)
st.caption(
    f"Avg Answer F1: {avg_f1:.3f}  |  "
    f"Avg Retrieval F1: {sum(r['retrieval_f1'] for r in rows)/n:.3f}  |  "
    f"Avg Trajectory: {sum(r['trajectory_score'] for r in rows)/n:.3f}"
)




