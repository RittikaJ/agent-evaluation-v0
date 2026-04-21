"""
Full-scale dataset setup: select 896 balanced questions from HotpotQA,
upload to Langfuse, and write dataset_local.json for Streamlit pages.

Selects 448 bridge + 448 comparison questions using a fixed seed for
reproducibility. Safe to re-run: uses deterministic item IDs for upsert.

Usage:
    python setup_dataset_full.py                    # 896 questions (default)
    python setup_dataset_full.py --n 200            # custom count (100 bridge + 100 comparison)
    python setup_dataset_full.py --local-only       # skip Langfuse upload, write dataset_local.json only
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_FILE = "hotpot_dev_distractor_v1.json"
SEED = 42


# ── Download ──────────────────────────────────────────────────────────────────


def download_hotpotqa() -> list[dict]:
    """Download the HotpotQA dev distractor set (~44MB) if not cached."""
    if Path(HOTPOT_FILE).exists():
        print(f"Using cached {HOTPOT_FILE}")
        with open(HOTPOT_FILE) as f:
            return json.load(f)

    print(f"Downloading {HOTPOT_URL} ...")
    resp = requests.get(HOTPOT_URL, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    with open(HOTPOT_FILE, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} examples to {HOTPOT_FILE}")
    return data


# ── Selection ─────────────────────────────────────────────────────────────────


def select_questions(data: list[dict], n: int = 896) -> list[dict]:
    """Select n questions balanced across bridge and comparison types.

    Uses fixed SEED=42 for reproducibility. Half bridge, half comparison.
    Falls back to available count if a pool is smaller than requested.
    """
    random.seed(SEED)
    half = n // 2

    bridge = [item for item in data if item["type"] == "bridge"]
    comparison = [item for item in data if item["type"] == "comparison"]

    n_bridge = min(half, len(bridge))
    n_comparison = min(n - n_bridge, len(comparison))

    selected = random.sample(bridge, n_bridge) + random.sample(comparison, n_comparison)
    random.shuffle(selected)

    b = sum(1 for s in selected if s["type"] == "bridge")
    c = sum(1 for s in selected if s["type"] == "comparison")
    print(f"Selected {len(selected)} questions: {b} bridge + {c} comparison")

    # Level distribution
    levels = {}
    for s in selected:
        lv = s.get("level", "unknown")
        levels[lv] = levels.get(lv, 0) + 1
    for lv, cnt in sorted(levels.items()):
        print(f"  {lv}: {cnt}")

    return selected


# ── Gold trajectory ───────────────────────────────────────────────────────────


def build_ground_truth_trajectory(item: dict) -> dict:
    """Build the ideal tool-call trajectory from supporting_facts.

    Mirrors the logic in setup_dataset.py (read-only contract).
    """
    supporting_facts = item["supporting_facts"]
    qtype = item["type"]

    seen = set()
    ordered_titles = []
    for title, _ in supporting_facts:
        if title not in seen:
            seen.add(title)
            ordered_titles.append(title)

    actions = []
    for title in ordered_titles:
        actions.append({
            "tool": "search_paragraphs",
            "input": {"query": f"search for information about {title}"},
            "expected_result_title": title,
        })
        actions.append({
            "tool": "read_paragraph",
            "input": {"title": title},
        })

    if qtype == "bridge" and len(ordered_titles) >= 2:
        reasoning_desc = (
            f"Find information from '{ordered_titles[0]}', then use that "
            f"to find and read '{ordered_titles[1]}' to answer the question."
        )
    elif qtype == "comparison" and len(ordered_titles) >= 2:
        reasoning_desc = (
            f"Find information about '{ordered_titles[0]}' and "
            f"'{ordered_titles[1]}', then compare them to answer."
        )
    else:
        reasoning_desc = f"Find and read paragraphs: {', '.join(ordered_titles)}"

    return {
        "actions": actions,
        "expected_paragraphs": ordered_titles,
        "reasoning_type": qtype,
        "reasoning_description": reasoning_desc,
    }


# ── Langfuse upload ──────────────────────────────────────────────────────────


def upload_to_langfuse(selected_items: list[dict], dataset_name: str):
    """Upload selected items to Langfuse with deterministic IDs for upsert."""
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
        timeout=60,
    )

    langfuse.create_dataset(name=dataset_name)
    print(f"Created/updated dataset '{dataset_name}'")

    for i, item in enumerate(selected_items):
        trajectory = build_ground_truth_trajectory(item)

        dataset_item = {
            "input": {
                "question": item["question"],
                "context": item["context"],
            },
            "expected_output": {
                "answer": item["answer"],
                "trajectory": trajectory,
            },
            "metadata": {
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": item["supporting_facts"],
                "hotpotqa_id": item["_id"],
            },
        }

        for attempt in range(3):
            try:
                langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    id=f"{dataset_name}-{i}",
                    input=dataset_item["input"],
                    expected_output=dataset_item["expected_output"],
                    metadata=dataset_item["metadata"],
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt + 1} for item {i + 1}: {e}")
                    time.sleep(2)
                else:
                    raise

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Uploaded {i + 1}/{len(selected_items)} items...")

    langfuse.flush()
    print(f"\nDone! {len(selected_items)} items uploaded to Langfuse dataset '{dataset_name}'")


# ── Write dataset_local.json ──────────────────────────────────────────────────


def write_dataset_local(selected_items: list[dict], dataset_name: str, output_path: str = "dataset_local.json"):
    """Write the local JSON file used by Streamlit pages and consistency_test.py.

    Schema: [{"id", "input": {"question", "context"}, "expected_output": {"answer", "trajectory"}, "metadata": {"type", "level"}}]
    """
    local_items = []
    for i, item in enumerate(selected_items):
        trajectory = build_ground_truth_trajectory(item)
        local_items.append({
            "id": f"{dataset_name}-{i}",
            "input": {
                "question": item["question"],
                "context": item["context"],
            },
            "expected_output": {
                "answer": item["answer"],
                "trajectory": trajectory,
            },
            "metadata": {
                "type": item["type"],
                "level": item["level"],
            },
        })

    with open(output_path, "w") as f:
        json.dump(local_items, f, indent=2)

    print(f"Wrote {len(local_items)} items to {output_path}")
    return local_items


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Set up full-scale HotpotQA dataset (896 questions)")
    parser.add_argument("--n", type=int, default=896, help="Number of questions to select (default: 896, half bridge + half comparison)")
    parser.add_argument("--local-only", action="store_true", help="Skip Langfuse upload, only write dataset_local.json")
    parser.add_argument("--output", default="dataset_local.json", help="Output path for local dataset JSON")
    args = parser.parse_args()

    dataset_name = f"DeepEval-hotpotqa-{args.n}"

    data = download_hotpotqa()
    selected = select_questions(data, n=args.n)

    if not args.local_only:
        upload_to_langfuse(selected, dataset_name)

    write_dataset_local(selected, dataset_name, args.output)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Items: {len(selected)}")
    print(f"Local file: {args.output}")
    if not args.local_only:
        print(f"Langfuse dataset: {dataset_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

