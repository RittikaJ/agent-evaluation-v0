"""
Consistency Test: rephrase each question and compare agent F1.

Takes questions from dataset_local.json, generates one rephrasing per question
via Claude, runs the agent on both the original and rephrased versions (using
the same context), and reports F1 drop to reveal brittleness.

Usage:
    python consistency_test.py                          # uses dataset_local.json (10 Qs)
    python consistency_test.py --extra 10               # adds 10 more from hotpot dev set
    python consistency_test.py --output results_consistency.json
"""

import argparse
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agent import run_agent

load_dotenv()

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"


# ── Helpers ───────────────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split()).strip()


def f1_score(predicted: str, gold: str) -> float:
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rephrase_question(question: str) -> str:
    """Use Claude to produce a semantically equivalent rephrasing."""
    prompt = f"""Rephrase the following question. The rephrased version MUST:
1. Ask for exactly the same information
2. Use noticeably different wording or sentence structure
3. Keep all proper nouns, entity names, and dates unchanged
4. Remain a complete, natural-sounding question

Original: {question}

Respond with ONLY the rephrased question — no explanation, no quotes."""

    response = _client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def load_extra_questions(n: int, exclude_ids: set) -> list[dict]:
    """Load additional questions from the local HotpotQA dev file (if available)."""
    hotpot_path = Path("hotpot_dev_distractor_v1.json")
    if not hotpot_path.exists():
        print("  Note: hotpot_dev_distractor_v1.json not found — skipping extra questions.")
        return []

    print(f"  Loading extra questions from {hotpot_path}...")
    with open(hotpot_path) as f:
        hotpot = json.load(f)

    candidates = [
        item for item in hotpot
        if item["_id"] not in exclude_ids and item.get("answer")
    ]
    random.seed(42)
    sampled = random.sample(candidates, min(n, len(candidates)))

    return [
        {
            "id": item["_id"],
            "input": {
                "question": item["question"],
                "context": item["context"],
            },
            "expected_output": {"answer": item["answer"]},
            "metadata": {
                "type": item.get("type", "unknown"),
                "level": item.get("level", "unknown"),
            },
        }
        for item in sampled
    ]


# ── Main ──────────────────────────────────────────────────────────────────────


def run_consistency_test(
    dataset_path: str = "dataset_local.json",
    n_extra: int = 0,
) -> dict:
    """Run the consistency test and return full results dict."""

    with open(dataset_path) as f:
        dataset = json.load(f)

    exclude_ids = {item["id"] for item in dataset}

    if n_extra > 0:
        extra = load_extra_questions(n_extra, exclude_ids)
        dataset = dataset + extra

    n = len(dataset)
    print(f"\nLoaded {n} questions. Running original + rephrased for each...\n")

    results = []

    for i, item in enumerate(dataset):
        question = item["input"]["question"]
        context = item["input"]["context"]
        gold_answer = item["expected_output"]["answer"]
        qtype = item["metadata"].get("type", "?")
        level = item["metadata"].get("level", "?")

        print(f"[{i + 1}/{n}] {question[:70]}...")

        # ── Original ──
        print("  ▶ Running original...")
        orig_result = run_agent(question, context)
        orig_f1 = f1_score(orig_result["answer"], gold_answer)

        # ── Rephrase ──
        print("  ✏️  Rephrasing...")
        rephrased = rephrase_question(question)
        print(f"  ↳ {rephrased[:70]}...")

        # ── Rephrased ──
        print("  ▶ Running rephrased...")
        reph_result = run_agent(rephrased, context)
        reph_f1 = f1_score(reph_result["answer"], gold_answer)

        f1_drop = orig_f1 - reph_f1
        brittle = f1_drop > 0.20

        flag = "⚠️ BRITTLE" if brittle else ("✅" if f1_drop <= 0.05 else "〰️")
        print(
            f"  orig_f1={orig_f1:.3f}  reph_f1={reph_f1:.3f}  "
            f"drop={f1_drop:+.3f}  {flag}"
        )

        results.append({
            "id": item["id"],
            "original_question": question,
            "rephrased_question": rephrased,
            "gold_answer": gold_answer,
            "original_answer": orig_result["answer"],
            "rephrased_answer": reph_result["answer"],
            "original_f1": orig_f1,
            "rephrased_f1": reph_f1,
            "f1_drop": f1_drop,
            "brittle": brittle,
            "type": qtype,
            "level": level,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────

    avg_orig = sum(r["original_f1"] for r in results) / n
    avg_reph = sum(r["rephrased_f1"] for r in results) / n
    avg_drop = sum(r["f1_drop"] for r in results) / n
    max_drop = max(r["f1_drop"] for r in results)
    brittle_count = sum(1 for r in results if r["brittle"])
    consistency_rate = 1.0 - (brittle_count / n)

    # Per-type stats
    type_stats = {}
    for qtype in sorted({r["type"] for r in results}):
        tr = [r for r in results if r["type"] == qtype]
        type_stats[qtype] = {
            "n": len(tr),
            "avg_orig_f1": sum(r["original_f1"] for r in tr) / len(tr),
            "avg_reph_f1": sum(r["rephrased_f1"] for r in tr) / len(tr),
            "avg_drop": sum(r["f1_drop"] for r in tr) / len(tr),
            "brittle_count": sum(1 for r in tr if r["brittle"]),
        }

    summary = {
        "n_questions": n,
        "avg_original_f1": avg_orig,
        "avg_rephrased_f1": avg_reph,
        "avg_f1_drop": avg_drop,
        "max_f1_drop": max_drop,
        "brittle_count": brittle_count,
        "brittleness_rate": brittle_count / n,
        "consistency_rate": consistency_rate,
        "type_breakdown": type_stats,
    }

    # ── Report ─────────────────────────────────────────────────────────────

    sep = "=" * 62
    print(f"\n{sep}")
    print("CONSISTENCY TEST RESULTS")
    print(sep)
    print(f"Questions tested       : {n}")
    print(f"Avg F1 (original)      : {avg_orig:.3f}")
    print(f"Avg F1 (rephrased)     : {avg_reph:.3f}")
    print(f"Mean F1 drop           : {avg_drop:+.3f}")
    print(f"Max F1 drop            : {max_drop:+.3f}")
    print(f"Brittle cases (>0.2 ↓) : {brittle_count}/{n}  ({brittle_count / n * 100:.0f}%)")
    print(f"Consistency rate       : {consistency_rate * 100:.0f}%")

    for qtype, stats in type_stats.items():
        print(
            f"\n  {qtype.title():12s}  n={stats['n']}  "
            f"avg_drop={stats['avg_drop']:+.3f}  "
            f"brittle={stats['brittle_count']}/{stats['n']}"
        )

    print(f"\n{sep}")
    if avg_drop > 0.15:
        print("⚠️  HIGH brittleness — answers shift significantly on rephrasing.")
        print("   Root cause: agent over-indexes on specific question keywords.")
        print("   Fix: make search queries more entity-focused, less phrasing-dependent.")
    elif avg_drop > 0.07:
        print("〰️  MODERATE brittleness — some sensitivity to question wording.")
        print("   The agent is mostly stable but has a few fragile cases.")
    else:
        print("✅  LOW brittleness — agent is consistent across question phrasings.")

    # Highlight worst cases
    worst = sorted(results, key=lambda r: r["f1_drop"], reverse=True)[:3]
    print("\nTop-3 most brittle cases:")
    for r in worst:
        print(f"  drop={r['f1_drop']:+.3f}  \"{r['original_question'][:60]}...\"")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency test for the QA agent")
    parser.add_argument(
        "--extra",
        type=int,
        default=10,
        metavar="N",
        help="Load N additional questions from hotpot_dev_distractor_v1.json (default: 10, giving 20 total)",
    )
    parser.add_argument(
        "--output",
        default="results_consistency.json",
        help="Output JSON file (default: results_consistency.json)",
    )
    args = parser.parse_args()

    data = run_consistency_test(n_extra=args.extra)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅  Full results saved to {args.output}")

