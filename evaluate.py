"""
Evaluation harness: runs the agent on every Langfuse dataset item and scores it.

Three evaluation layers:
  Layer 1 — Answer Quality (deterministic): Exact Match and token-level F1
            comparing the agent's answer to the gold answer.
  Layer 2 — Trajectory Quality (deterministic): paragraph retrieval precision/recall,
            action sequence order (LCS vs gold), and search efficiency.
  Layer 3 — Reasoning Quality (LLM-as-judge): Claude scores the agent's trajectory
            on groundedness, reasoning coherence, and search strategy (1-5 each)
            using rubrics from rubrics/*.md.

Composite score (what Phase 2 optimizes):
  score = 0.4 * answer_f1 + 0.35 * trajectory_score + 0.25 * reasoning_quality

Each run creates a Langfuse experiment with per-item scores and a trace per question.

Usage:
    python evaluate.py
"""

import json
import os
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langfuse import Langfuse

from agent import run_agent

load_dotenv()

DATASET_NAME = "rittika-hotpotqa-10"
JUDGE_MODEL = "claude-sonnet-4-6"

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)
judge_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Layer 1: Answer Quality (deterministic) ──────────────────────────────────


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison.

    Steps: lowercase → remove articles (a/an/the) → strip punctuation → collapse whitespace.
    This is the standard SQuAD/HotpotQA normalization used by both EM and F1.
    """
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s.strip()


def exact_match(predicted: str, gold: str) -> float:
    """1.0 if normalized predicted == normalized gold, else 0.0."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def f1_score(predicted: str, gold: str) -> float:
    """Token-level F1 between normalized predicted and gold answers.

    Precision = |predicted ∩ gold| / |predicted|
    Recall    = |predicted ∩ gold| / |gold|
    F1        = harmonic mean of precision and recall
    """
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


# ── Layer 2: Trajectory Quality (deterministic) ──────────────────────────────


def paragraph_retrieval_scores(trajectory: list, gold_trajectory: dict) -> dict:
    """Compute paragraph retrieval precision, recall, and F1.

    Compares the set of paragraph titles the agent actually retrieved
    (via search results or explicit read_paragraph calls) against the
    gold set of expected paragraphs from the ground truth trajectory.

    Returns dict with keys: paragraph_precision, paragraph_recall, retrieval_f1.
    """
    gold_titles = set(gold_trajectory.get("expected_paragraphs", []))
    if not gold_titles:
        return {"paragraph_precision": 0.0, "paragraph_recall": 0.0, "retrieval_f1": 0.0}

    # Titles the agent actually read (via read_paragraph or found via search)
    retrieved_titles = set()
    for step in trajectory:
        if step["tool"] == "read_paragraph":
            retrieved_titles.add(step["input"].get("title", ""))
        elif step["tool"] == "search_paragraphs":
            # Count paragraphs returned by search
            try:
                results = json.loads(step["output"]) if isinstance(step["output"], str) else step["output"]
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict) and "title" in r:
                            retrieved_titles.add(r["title"])
            except (json.JSONDecodeError, TypeError):
                pass

    if not retrieved_titles:
        return {"paragraph_precision": 0.0, "paragraph_recall": 0.0, "retrieval_f1": 0.0}

    intersection = gold_titles & retrieved_titles
    precision = len(intersection) / len(retrieved_titles)
    recall = len(intersection) / len(gold_titles)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"paragraph_precision": precision, "paragraph_recall": recall, "retrieval_f1": f1}


def action_order_score(trajectory: list, gold_trajectory: dict) -> float:
    """Score how well the agent's tool call order matches the gold sequence.

    Uses Longest Common Subsequence (LCS) on tool names only (not inputs).
    Score = LCS length / gold sequence length. Rewards agents that follow the
    correct search→read→search→read pattern without penalizing extra calls.
    """
    gold_actions = [a["tool"] for a in gold_trajectory.get("actions", [])]
    agent_actions = [step["tool"] for step in trajectory]

    if not gold_actions:
        return 1.0 if not agent_actions else 0.0

    # LCS
    m, n = len(agent_actions), len(gold_actions)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if agent_actions[i - 1] == gold_actions[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n] / len(gold_actions)


def search_efficiency(trajectory: list, gold_trajectory: dict) -> float:
    """Ratio of gold action count to actual action count (0-1).

    Score = len(gold) / max(len(agent), len(gold)).
    Penalizes both too few calls (missed evidence) and too many (redundant).
    A perfect score means the agent used exactly the minimum number of tool calls.
    """
    gold_actions = gold_trajectory.get("actions", [])
    if not gold_actions:
        return 1.0 if not trajectory else 0.0
    return len(gold_actions) / max(len(trajectory), len(gold_actions))


def redundancy_ratio(trajectory: list) -> float:
    """Fraction of duplicate search queries (0-1, lower is better).

    Computed as 1 - (unique_queries / total_queries). Tracked for diagnostics
    but not included in the composite score.
    """
    queries = [
        step["input"].get("query", "").lower().strip()
        for step in trajectory
        if step["tool"] == "search_paragraphs"
    ]
    if not queries:
        return 0.0
    unique = len(set(queries))
    return 1.0 - (unique / len(queries))


def trajectory_composite(retrieval_f1: float, action_score: float, efficiency: float) -> float:
    """Weighted trajectory score: 40% retrieval + 40% action order + 20% efficiency."""
    return 0.4 * retrieval_f1 + 0.4 * action_score + 0.2 * efficiency


# ── Layer 3: Reasoning Quality (LLM-as-judge) ────────────────────────────────


def load_rubric(name: str) -> str:
    """Load a scoring rubric from rubrics/{name}.md for the LLM judge."""
    return Path(f"rubrics/{name}.md").read_text()


def judge_reasoning(
    question: str,
    answer: str,
    gold_answer: str,
    trajectory: list,
    rubric_name: str,
    rubric_text: str,
) -> int:
    """Use Claude as a judge to score one reasoning dimension (1-5).

    Sends the question, agent trajectory, agent answer, gold answer, and rubric
    to a separate Claude call. The judge sees the full tool call sequence but
    NOT the gold supporting_facts (to avoid leaking evidence).

    Returns an integer score 1-5 (clamped). Falls back to 1 on parse failure.
    """
    trajectory_text = ""
    for i, step in enumerate(trajectory):
        output_preview = str(step["output"])[:500]
        trajectory_text += (
            f"Step {i+1}: {step['tool']}({json.dumps(step['input'])})\n"
            f"  Output: {output_preview}\n\n"
        )
    if not trajectory_text:
        trajectory_text = "(No tool calls were made)"

    prompt = f"""You are an evaluation judge. Score the agent's {rubric_name} on a scale of 1-5.

## Rubric
{rubric_text}

## Question
{question}

## Agent's Trajectory
{trajectory_text}

## Agent's Final Answer
{answer}

## Ground Truth Answer
{gold_answer}

## Instructions
1. Read the rubric carefully, including the hard guardrails.
2. Analyze the agent's trajectory and answer against the rubric criteria.
3. Apply hard guardrails first — they override the normal scoring.
4. Give your score as a single integer 1-5.

Respond with ONLY a JSON object: {{"reasoning": "brief explanation", "score": N}}"""

    with langfuse.start_as_current_generation(
        name=f"judge-{rubric_name}",
        model=JUDGE_MODEL,
        input=prompt,
    ) as generation:
        response = judge_client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        generation.update(
            output=response.content[0].text,
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )

    text = response.content[0].text.strip()
    try:
        result = json.loads(text)
        score = int(result["score"])
        return max(1, min(5, score))
    except (json.JSONDecodeError, KeyError, ValueError):
        # Try to extract a number
        match = re.search(r'"score"\s*:\s*(\d)', text)
        if match:
            return max(1, min(5, int(match.group(1))))
        return 1


# ── Main evaluation loop ─────────────────────────────────────────────────────


def evaluate():
    """Main evaluation loop: run agent on all dataset items, score, and report.

    Creates a timestamped Langfuse experiment. For each item:
    1. Runs the agent within a Langfuse dataset run (auto-creates trace)
    2. Computes Layer 1 (answer F1/EM), Layer 2 (trajectory), Layer 3 (LLM judge)
    3. Uploads per-item scores to Langfuse

    Prints per-case breakdown and aggregate composite score at the end.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hotpotqa_eval_{timestamp}"

    # Load rubrics
    rubrics = {
        "groundedness": load_rubric("groundedness"),
        "reasoning_coherence": load_rubric("reasoning_coherence"),
        "search_strategy": load_rubric("search_strategy"),
    }

    # Get dataset from Langfuse
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = dataset.items

    print(f"Running evaluation '{experiment_name}' on {len(items)} items...\n")

    all_scores = []

    for i, item in enumerate(items):
        question = item.input["question"]
        context = item.input["context"]
        gold_answer = item.expected_output["answer"]
        gold_trajectory = item.expected_output["trajectory"]
        metadata = item.metadata or {}

        print(f"Case {i+1}: {question[:60]}...")

        # Run agent + judges within a Langfuse dataset run so every generation
        # (agent turns + judge calls) nests under the same trace.
        with item.run(
            run_name=experiment_name,
            run_metadata={"question_type": metadata.get("type"), "question_level": metadata.get("level")},
        ) as span:
            result = run_agent(question, context)
            trace_id = span.trace_id

            # ── Layer 3: Reasoning Quality (LLM-as-judge) ──
            # Inside the span so judge generations nest under this trace.
            groundedness = judge_reasoning(
                question, result["answer"], gold_answer,
                result["trajectory"], "groundedness", rubrics["groundedness"],
            )
            reasoning_coherence = judge_reasoning(
                question, result["answer"], gold_answer,
                result["trajectory"], "reasoning_coherence", rubrics["reasoning_coherence"],
            )
            search_strategy = judge_reasoning(
                question, result["answer"], gold_answer,
                result["trajectory"], "search_strategy", rubrics["search_strategy"],
            )

            span.update(
                input=question,
                output=result["answer"],
                metadata={
                    "trajectory_length": len(result["trajectory"]),
                    "token_usage": result["token_usage"],
                },
            )

        # ── Layer 1: Answer Quality ──
        answer_em = exact_match(result["answer"], gold_answer)
        answer_f1 = f1_score(result["answer"], gold_answer)

        # ── Layer 2: Trajectory Quality ──
        retrieval = paragraph_retrieval_scores(result["trajectory"], gold_trajectory)
        action_score = action_order_score(result["trajectory"], gold_trajectory)
        efficiency = search_efficiency(result["trajectory"], gold_trajectory)
        redundancy = redundancy_ratio(result["trajectory"])
        traj_score = trajectory_composite(retrieval["retrieval_f1"], action_score, efficiency)

        reasoning_avg = (groundedness + reasoning_coherence + search_strategy) / 3.0

        scores = {
            "answer_f1": answer_f1,
            "answer_em": answer_em,
            "paragraph_precision": retrieval["paragraph_precision"],
            "paragraph_recall": retrieval["paragraph_recall"],
            "retrieval_f1": retrieval["retrieval_f1"],
            "action_order": action_score,
            "efficiency": efficiency,
            "redundancy": redundancy,
            "trajectory_score": traj_score,
            "groundedness": groundedness,
            "reasoning_coherence": reasoning_coherence,
            "search_strategy": search_strategy,
            "reasoning_avg": reasoning_avg,
        }
        all_scores.append(scores)

        # Compute composite for this item
        reasoning_norm = reasoning_avg / 5.0
        item_composite = 0.4 * answer_f1 + 0.35 * traj_score + 0.25 * reasoning_norm

        # Upload business-friendly scores to Langfuse (7 per trace)
        langfuse_scores = {
            "Answer Accuracy": answer_f1,
            "Evidence Quality": traj_score,
            "Groundedness": groundedness,
            "Reasoning Quality": reasoning_coherence,
            "Search Quality": search_strategy,
            "Overall Score": item_composite,
        }
        for score_name, score_val in langfuse_scores.items():
            langfuse.create_score(
                name=score_name,
                value=score_val,
                trace_id=trace_id,
                data_type="NUMERIC",
            )

        level = metadata.get("level", "?")
        qtype = metadata.get("type", "?")
        print(
            f"  answer_f1={answer_f1:.2f}  trajectory={traj_score:.2f}  "
            f"reasoning={reasoning_avg:.1f}  ({level}/{qtype})"
        )

    # ── Aggregate scores ──
    n = len(all_scores)
    avg = lambda key: sum(s[key] for s in all_scores) / n

    # Layer 1: Answer Score (0-1)
    answer_score = avg("answer_f1")

    # Layer 2: Trajectory Score (0-1)
    trajectory_score = avg("trajectory_score")

    # Layer 3: Reasoning Score (0-1, normalized from 1-5 judge scores)
    reasoning_score = (
        avg("groundedness") / 5.0
        + avg("reasoning_coherence") / 5.0
        + avg("search_strategy") / 5.0
    ) / 3.0

    # Composite: weighted combination of the three
    composite = 0.4 * answer_score + 0.35 * trajectory_score + 0.25 * reasoning_score

    print("\n---")
    print(f"answer_score: {answer_score:.4f}")
    print(f"  answer_f1: {avg('answer_f1'):.4f}")
    print(f"  answer_em: {avg('answer_em'):.4f}")
    print(f"trajectory_score: {trajectory_score:.4f}")
    print(f"  retrieval_f1: {avg('retrieval_f1'):.4f}")
    print(f"  action_order: {avg('action_order'):.4f}")
    print(f"  efficiency: {avg('efficiency'):.4f}")
    print(f"reasoning_score: {reasoning_score:.4f}")
    print(f"  groundedness: {avg('groundedness') / 5.0:.4f}")
    print(f"  reasoning_coherence: {avg('reasoning_coherence') / 5.0:.4f}")
    print(f"  search_strategy: {avg('search_strategy') / 5.0:.4f}")
    print(f"composite: {composite:.4f}")

    langfuse.flush()


if __name__ == "__main__":
    evaluate()
