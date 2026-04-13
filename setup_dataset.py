"""
One-time dataset setup script.

Downloads the HotpotQA dev distractor set, selects 20 multi-hop questions
(10 bridge + 10 comparison), builds ground truth trajectories from supporting_facts,
and uploads everything to Langfuse as a named dataset.

Safe to re-run: uses deterministic item IDs (DATASET_NAME-{i}) so Langfuse
upserts rather than duplicates.

Usage:
    python setup_dataset.py
"""

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
DATASET_NAME = "DeepEval-hotpotqa-20"
SEED = 42


def download_hotpotqa():
    """Download the HotpotQA dev distractor set (~44MB, 7405 examples) if not already cached locally.

    Returns the full list of example dicts. Each example has keys:
    _id, question, answer, context, supporting_facts, type, level.
    """
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


def select_questions(data, n=10):
    """Randomly select n questions with a balanced mix of bridge and comparison types.

    Bridge questions require chained reasoning (find A → use A to find B).
    Comparison questions require parallel retrieval (find A and B → compare).
    Uses a fixed seed (SEED=42) for reproducibility across runs.
    """
    random.seed(SEED)

    half = n // 2
    bridge = [item for item in data if item["type"] == "bridge"]
    comparison = [item for item in data if item["type"] == "comparison"]

    selected = random.sample(bridge, half) + random.sample(comparison, n - half)
    random.shuffle(selected)

    print(f"Selected {len(selected)} questions")
    b = sum(1 for s in selected if s["type"] == "bridge")
    c = sum(1 for s in selected if s["type"] == "comparison")
    print(f"  bridge: {b}, comparison: {c}")

    return selected


def build_ground_truth_trajectory(item):
    """Build the ideal tool-call trajectory for a question from its supporting_facts.

    HotpotQA's supporting_facts field is a list of [title, sent_id] pairs — the exact
    paragraphs and sentences needed to answer. From this we construct:

    - actions: the ideal sequence of search_paragraphs → read_paragraph calls
    - expected_paragraphs: the titles the agent should retrieve
    - reasoning_type: "bridge" (chained) or "comparison" (parallel)
    - reasoning_description: human-readable explanation of the ideal reasoning path

    For bridge questions the paragraphs have an implicit order (first leads to second).
    For comparison questions the two paragraphs are independent.
    """
    supporting_facts = item["supporting_facts"]
    question = item["question"]
    qtype = item["type"]

    # Get unique paragraph titles in order of appearance
    seen = set()
    ordered_titles = []
    for title, _ in supporting_facts:
        if title not in seen:
            seen.add(title)
            ordered_titles.append(title)

    # Build action sequence
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

    # Build reasoning description
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


def upload_to_langfuse(selected_items):
    """Upload selected items to Langfuse as a named dataset.

    Each item is created with a deterministic ID (DATASET_NAME-{index}) so that
    re-running this function upserts rather than creating duplicates.

    Each Langfuse dataset item contains:
    - input: {"question": str, "context": list of [title, [sentences...]]}
    - expected_output: {"answer": str, "trajectory": ground truth trajectory dict}
    - metadata: {"type", "level", "supporting_facts", "hotpotqa_id"}

    Retries up to 3 times per item on timeout (context payloads can be large).
    """
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
        timeout=60,
    )

    # Create or get dataset
    langfuse.create_dataset(name=DATASET_NAME)
    print(f"Created/updated dataset '{DATASET_NAME}'")

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
                    dataset_name=DATASET_NAME,
                    id=f"{DATASET_NAME}-{i}",
                    input=dataset_item["input"],
                    expected_output=dataset_item["expected_output"],
                    metadata=dataset_item["metadata"],
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt+1} for item {i+1}: {e}")
                    time.sleep(2)
                else:
                    raise
        print(f"  Uploaded item {i+1}/{len(selected_items)}: {item['question'][:60]}... ({item['level']}/{item['type']})")

    langfuse.flush()
    print(f"\nDone! {len(selected_items)} items uploaded to Langfuse dataset '{DATASET_NAME}'")


def main():
    data = download_hotpotqa()
    selected = select_questions(data, n=20)
    upload_to_langfuse(selected)


if __name__ == "__main__":
    main()
