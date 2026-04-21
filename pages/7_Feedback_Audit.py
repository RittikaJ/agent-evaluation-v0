"""
Feedback Audit: surface human vs LLM-judge disagreements to guide rubric recalibration.

This page:
  - Loads all human and LLM judge scores from Langfuse
  - Shows aggregate agreement/disagreement per dimension
  - Highlights rubric dimensions with notable gaps (>0.3 normalized)
  - Surfaces specific cases with largest disagreements for manual review
  - Generates recalibration recommendations
  - Provides a priority queue for human audit with CSV download
"""

import json
import os
from collections import defaultdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

st.set_page_config(page_title="Feedback Audit", page_icon="🔍", layout="wide")
st.title("🔍 Feedback Audit & Rubric Recalibration")
st.markdown(
    "Aggregate human feedback vs LLM-judge scores to identify rubric calibration issues. "
    "Highlights dimensions where human reviewers disagree with the automated judge, "
    "and generates actionable recalibration recommendations."
)
st.divider()


@st.cache_resource
def get_langfuse():
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )


@st.cache_data
def load_dataset():
    try:
        with open("dataset_local.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


langfuse = get_langfuse()
dataset = load_dataset()

# Build a question lookup by trace_id (best-effort from dataset)
_question_by_id = {}
for item in dataset:
    _question_by_id[item["id"]] = item["input"]["question"]


# ── Score dimension mapping ────────────────────────────────────────────────────

# Maps (human_score_name, llm_score_name, label, normalization)
DIMENSION_MAP = [
    {
        "label": "Answer Quality",
        "human_name": "Human Answer Quality",
        "llm_name": "Answer Accuracy",
        "human_scale": 5,   # human scores 1-5, normalize /5
        "llm_scale": 1,     # LLM score already 0-1
    },
    {
        "label": "Search Strategy",
        "human_name": "Human Search Quality",
        "llm_name": "Search Quality",
        "human_scale": 5,
        "llm_scale": 5,     # LLM judge scores 1-5
    },
    {
        "label": "Evidence / Groundedness",
        "human_name": "Human Evidence Quality",
        "llm_name": "Groundedness",
        "human_scale": 5,
        "llm_scale": 5,
    },
    {
        "label": "Reasoning Quality",
        "human_name": "Human Reasoning Quality",
        "llm_name": "Reasoning Quality",
        "human_scale": 5,
        "llm_scale": 5,
    },
    {
        "label": "Overall",
        "human_name": "Human Overall",
        "llm_name": "Overall Score",
        "human_scale": 5,
        "llm_scale": 1,     # Overall Score is 0-1
    },
]


# ── Fetch scores from Langfuse ─────────────────────────────────────────────────

def fetch_all_scores():
    """Fetch all human and LLM scores from Langfuse, grouped by trace_id."""
    # Gather all score names we care about
    human_names = {d["human_name"] for d in DIMENSION_MAP}
    llm_names = {d["llm_name"] for d in DIMENSION_MAP}
    all_names = human_names | llm_names

    trace_scores = defaultdict(dict)

    # Paginate through scores
    page = 1
    page_size = 100
    fetched = 0

    try:
        while True:
            scores_page = langfuse.api.scores.list(page=page, limit=page_size)

            # Handle different response shapes from Langfuse SDK
            if hasattr(scores_page, "data"):
                scores_list = scores_page.data
            elif isinstance(scores_page, list):
                scores_list = scores_page
            else:
                break

            if not scores_list:
                break

            for score in scores_list:
                name = score.name if hasattr(score, "name") else score.get("name", "")
                if name in all_names:
                    trace_id = score.trace_id if hasattr(score, "trace_id") else score.get("trace_id", "")
                    value = score.value if hasattr(score, "value") else score.get("value", 0)
                    comment = ""
                    if hasattr(score, "comment"):
                        comment = score.comment or ""
                    elif isinstance(score, dict):
                        comment = score.get("comment", "") or ""

                    trace_scores[trace_id][name] = {
                        "value": value,
                        "comment": comment,
                    }
                    fetched += 1

            # Check if there are more pages
            if hasattr(scores_page, "meta") and hasattr(scores_page.meta, "total_pages"):
                if page >= scores_page.meta.total_pages:
                    break
            elif len(scores_list) < page_size:
                break

            page += 1

    except Exception as e:
        st.warning(f"Score fetch stopped: {e}. Showing {fetched} scores so far.")

    return trace_scores


# ── Build paired data ──────────────────────────────────────────────────────────

def build_paired_data(trace_scores: dict) -> list[dict]:
    """Build rows where we have both human AND LLM scores for at least one dimension."""
    rows = []

    for trace_id, scores in trace_scores.items():
        human_names = {d["human_name"] for d in DIMENSION_MAP}
        llm_names = {d["llm_name"] for d in DIMENSION_MAP}

        has_human = any(name in scores for name in human_names)
        has_llm = any(name in scores for name in llm_names)

        if not (has_human and has_llm):
            continue

        row = {"trace_id": trace_id}

        # Extract comment (often contains failure tags)
        for name, data in scores.items():
            if data.get("comment"):
                row["comment"] = data["comment"]
                break

        # Build dimension pairs
        for dim in DIMENSION_MAP:
            human_data = scores.get(dim["human_name"])
            llm_data = scores.get(dim["llm_name"])

            if human_data and llm_data:
                human_norm = human_data["value"] / dim["human_scale"]
                llm_norm = llm_data["value"] / dim["llm_scale"]
                gap = human_norm - llm_norm

                row[f"{dim['label']}_human"] = human_norm
                row[f"{dim['label']}_llm"] = llm_norm
                row[f"{dim['label']}_gap"] = gap
                row[f"{dim['label']}_abs_gap"] = abs(gap)

        rows.append(row)

    return rows


# ── Main UI ────────────────────────────────────────────────────────────────────

if st.button("🔄 Fetch Scores from Langfuse", type="primary", use_container_width=True):
    with st.spinner("Fetching scores from Langfuse..."):
        trace_scores = fetch_all_scores()
        paired_data = build_paired_data(trace_scores)

    st.session_state.audit_trace_scores = trace_scores
    st.session_state.audit_paired_data = paired_data
    st.success(
        f"Loaded {len(trace_scores)} traces with scores, "
        f"{len(paired_data)} have both human and LLM judge scores."
    )


if "audit_paired_data" not in st.session_state:
    st.info("Click **🔄 Fetch Scores from Langfuse** to load and analyze feedback data.")
    st.stop()

paired_data = st.session_state.audit_paired_data

if not paired_data:
    st.warning(
        "⚠️ No traces found with both human AND LLM judge scores. "
        "Run some evaluations and submit feedback first, then return here."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Aggregate Agreement Dashboard
# ══════════════════════════════════════════════════════════════════════════════

st.header("📊 Aggregate Agreement Dashboard")
st.caption(
    "For each dimension, compare the mean human score vs mean LLM judge score. "
    "Large gaps indicate rubric calibration issues."
)

agg_rows = []
for dim in DIMENSION_MAP:
    label = dim["label"]
    human_key = f"{label}_human"
    llm_key = f"{label}_llm"
    gap_key = f"{label}_gap"
    abs_gap_key = f"{label}_abs_gap"

    # Filter to rows that have this dimension
    dim_rows = [r for r in paired_data if human_key in r]
    if not dim_rows:
        continue

    n_pairs = len(dim_rows)
    mean_human = sum(r[human_key] for r in dim_rows) / n_pairs
    mean_llm = sum(r[llm_key] for r in dim_rows) / n_pairs
    mean_gap = sum(r[gap_key] for r in dim_rows) / n_pairs
    mean_abs_gap = sum(r[abs_gap_key] for r in dim_rows) / n_pairs

    # Pearson correlation (simplified)
    if n_pairs >= 3:
        h_vals = [r[human_key] for r in dim_rows]
        l_vals = [r[llm_key] for r in dim_rows]
        h_mean = mean_human
        l_mean = mean_llm
        cov = sum((h - h_mean) * (l - l_mean) for h, l in zip(h_vals, l_vals)) / n_pairs
        h_std = (sum((h - h_mean) ** 2 for h in h_vals) / n_pairs) ** 0.5
        l_std = (sum((l - l_mean) ** 2 for l in l_vals) / n_pairs) ** 0.5
        correlation = cov / (h_std * l_std) if h_std > 0 and l_std > 0 else 0.0
    else:
        correlation = None

    agg_rows.append({
        "Dimension": label,
        "N Pairs": n_pairs,
        "Mean Human (0-1)": round(mean_human, 3),
        "Mean LLM (0-1)": round(mean_llm, 3),
        "Mean Gap (H-L)": round(mean_gap, 3),
        "Mean |Gap|": round(mean_abs_gap, 3),
        "Correlation": round(correlation, 3) if correlation is not None else "—",
        "Status": "🔴 Miscalibrated" if mean_abs_gap > 0.3 else ("🟡 Check" if mean_abs_gap > 0.15 else "🟢 Aligned"),
    })

if agg_rows:
    df_agg = pd.DataFrame(agg_rows)
    st.dataframe(df_agg, use_container_width=True, hide_index=True)

    # Visual comparison
    chart_data = {
        row["Dimension"]: {
            "Human": row["Mean Human (0-1)"],
            "LLM Judge": row["Mean LLM (0-1)"],
        }
        for row in agg_rows
    }
    df_chart = pd.DataFrame(chart_data).T
    st.bar_chart(df_chart, horizontal=True, color=["#e05c5c", "#4a90d9"])

    # Flag miscalibrated dimensions
    miscal = [r for r in agg_rows if r["Mean |Gap|"] > 0.3]
    if miscal:
        for r in miscal:
            st.error(
                f"🔴 **{r['Dimension']}**: Mean |gap| = {r['Mean |Gap|']} — "
                f"human avg = {r['Mean Human (0-1)']}, LLM avg = {r['Mean LLM (0-1)']}. "
                f"Rubric recalibration recommended."
            )
else:
    st.info("Not enough paired scores to compute aggregates.")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Worst Disagreements
# ══════════════════════════════════════════════════════════════════════════════

st.header("🔎 Worst Disagreements (Per Dimension)")
st.caption(
    "For each dimension, the top cases where human and LLM judge disagree most. "
    "These are the highest-priority cases for manual rubric review."
)

for dim in DIMENSION_MAP:
    label = dim["label"]
    abs_gap_key = f"{label}_abs_gap"
    human_key = f"{label}_human"
    llm_key = f"{label}_llm"
    gap_key = f"{label}_gap"

    dim_rows = [r for r in paired_data if abs_gap_key in r]
    if not dim_rows:
        continue

    # Sort by absolute gap descending
    worst = sorted(dim_rows, key=lambda r: r[abs_gap_key], reverse=True)[:5]

    if not worst or worst[0][abs_gap_key] < 0.1:
        continue

    with st.expander(f"**{label}** — top {len(worst)} disagreements", expanded=False):
        table = []
        for r in worst:
            table.append({
                "Trace ID": r["trace_id"][:20] + "...",
                "Human (0-1)": f"{r[human_key]:.2f}",
                "LLM (0-1)": f"{r[llm_key]:.2f}",
                "Gap": f"{r[gap_key]:+.2f}",
                "|Gap|": f"{r[abs_gap_key]:.2f}",
                "Comment": (r.get("comment", "")[:80] + "...") if r.get("comment") else "—",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Recalibration Recommendations
# ══════════════════════════════════════════════════════════════════════════════

st.header("💡 Recalibration Recommendations")
st.caption(
    "Programmatically generated recommendations based on systematic human-vs-LLM gaps. "
    "Positive gap = humans rate higher than LLM (rubric too strict). "
    "Negative gap = LLM rates higher than humans (rubric too lenient)."
)

recommendations = []
rubric_map = {
    "Evidence / Groundedness": "rubrics/groundedness.md",
    "Reasoning Quality": "rubrics/reasoning_coherence.md",
    "Search Strategy": "rubrics/search_strategy.md",
}

for row in agg_rows:
    dim_label = row["Dimension"]
    mean_gap = row["Mean Gap (H-L)"]
    abs_gap = row["Mean |Gap|"]

    if abs_gap < 0.15:
        continue

    rubric_file = rubric_map.get(dim_label, "N/A")

    if mean_gap > 0.15:
        severity = "HIGH" if abs_gap > 0.3 else "MODERATE"
        recommendations.append({
            "dimension": dim_label,
            "issue": "Rubric too strict",
            "severity": severity,
            "detail": (
                f"Humans rated **{dim_label}** {abs(mean_gap):.0%} higher on average than the LLM judge. "
                f"The rubric `{rubric_file}` may be penalizing too harshly — consider relaxing guardrails "
                f"or adding intermediate scoring criteria."
            ),
        })
    elif mean_gap < -0.15:
        severity = "HIGH" if abs_gap > 0.3 else "MODERATE"
        recommendations.append({
            "dimension": dim_label,
            "issue": "Rubric too lenient",
            "severity": severity,
            "detail": (
                f"The LLM judge over-rates **{dim_label}** by {abs(mean_gap):.0%} compared to human reviewers. "
                f"The rubric `{rubric_file}` may be too permissive — consider tightening scoring criteria "
                f"or adding hard guardrails for common failure patterns."
            ),
        })
    else:
        recommendations.append({
            "dimension": dim_label,
            "issue": "High variance",
            "severity": "MODERATE",
            "detail": (
                f"**{dim_label}** shows high per-case disagreement (mean |gap| = {abs_gap:.2f}) "
                f"despite similar averages. The rubric may be ambiguous — "
                f"consider adding more concrete examples or refining scoring anchors."
            ),
        })

if recommendations:
    for rec in recommendations:
        icon = "🔴" if rec["severity"] == "HIGH" else "🟡"
        if rec["issue"] == "Rubric too strict":
            st.warning(f"{icon} **{rec['dimension']}** — {rec['issue']} ({rec['severity']})\n\n{rec['detail']}")
        elif rec["issue"] == "Rubric too lenient":
            st.error(f"{icon} **{rec['dimension']}** — {rec['issue']} ({rec['severity']})\n\n{rec['detail']}")
        else:
            st.info(f"{icon} **{rec['dimension']}** — {rec['issue']} ({rec['severity']})\n\n{rec['detail']}")
else:
    st.success("✅ All dimensions are well-calibrated — no recalibration needed!")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Recalibration Queue (Priority)
# ══════════════════════════════════════════════════════════════════════════════

st.header("📋 Recalibration Queue")
st.caption(
    "Traces ranked by maximum dimension gap — these need human audit first. "
    "Download as CSV for offline review."
)

queue_rows = []
for r in paired_data:
    max_gap = 0
    max_dim = ""
    for dim in DIMENSION_MAP:
        label = dim["label"]
        abs_gap_key = f"{label}_abs_gap"
        if abs_gap_key in r and r[abs_gap_key] > max_gap:
            max_gap = r[abs_gap_key]
            max_dim = label

    if max_gap > 0:
        priority = "🔴 High" if max_gap > 0.5 else ("🟡 Medium" if max_gap > 0.3 else "🟢 Low")
        queue_rows.append({
            "Priority": priority,
            "Max Gap": round(max_gap, 3),
            "Dimension": max_dim,
            "Trace ID": r["trace_id"],
            "Comment": (r.get("comment", "")[:60] + "...") if r.get("comment") else "—",
        })

# Sort by max gap descending
queue_rows.sort(key=lambda r: r["Max Gap"], reverse=True)

if queue_rows:
    df_queue = pd.DataFrame(queue_rows)
    st.dataframe(df_queue, use_container_width=True, hide_index=True)

    # Summary stats
    n_high = sum(1 for r in queue_rows if "High" in r["Priority"])
    n_medium = sum(1 for r in queue_rows if "Medium" in r["Priority"])
    n_low = sum(1 for r in queue_rows if "Low" in r["Priority"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total in Queue", len(queue_rows))
    c2.metric("🔴 High Priority", n_high)
    c3.metric("🟡 Medium", n_medium)
    c4.metric("🟢 Low", n_low)

    # CSV download
    csv = df_queue.to_csv(index=False)
    st.download_button(
        label="📥 Download Recalibration Queue (CSV)",
        data=csv,
        file_name="recalibration_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("No disagreements found — nothing to queue for recalibration.")


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Raw Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("📦 Raw Paired Scores", expanded=False):
    st.caption("Full paired scores data for debugging.")
    if paired_data:
        st.dataframe(pd.DataFrame(paired_data), use_container_width=True, hide_index=True)
    else:
        st.info("No paired data available.")

st.divider()
st.caption(
    f"Loaded {len(paired_data)} traces with paired human + LLM scores. "
    "Recommendations are generated programmatically from aggregate disagreement patterns."
)

