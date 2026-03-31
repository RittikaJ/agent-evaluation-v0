# Evaluation Details

Detailed documentation of the 3-layer evaluation pipeline.

## Layer 1: Answer Score (deterministic)

Pure string comparison between the agent's answer and the gold answer.

| Metric | How It Works |
|--------|-------------|
| **F1** | Token-level precision/recall after normalization (lowercase, strip articles/punctuation, collapse whitespace). If gold is `"Bonn"` and agent says `"The city of Bonn has a population of 327,913"`, F1 is low because precision is terrible. |
| **Exact Match** | 1 if normalized predicted == normalized gold, else 0. Strict — `"Barack Obama"` vs `"Obama"` is a miss. |

**`answer_score`** = average F1 across all questions.

## Layer 2: Trajectory Score (deterministic)

Compares the agent's actual tool calls against the gold trajectory.

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Retrieval F1** | Did the agent find the right evidence paragraphs? | Set intersection of retrieved titles vs gold titles → precision, recall, F1 |
| **Action Order** | Did the agent follow the right tool sequence? | Longest Common Subsequence of tool names (agent vs gold) / gold length |
| **Efficiency** | Did the agent waste tool calls? | gold action count / max(agent count, gold count) |

**`trajectory_score`** = 0.4 * retrieval_f1 + 0.4 * action_order + 0.2 * efficiency

## Layer 3: Reasoning Score (LLM-as-judge)

Layers 1 and 2 are deterministic — string comparison and sequence matching. But some things can't be measured mechanically: *"Does this reasoning chain actually make sense?"* or *"Is this answer genuinely supported by the evidence?"* That's what LLM-as-judge is for.

**How it works**: For each question, we make 3 separate Claude API calls (using `claude-sonnet-4-6`). Each call is a *different* Claude instance acting as a judge — completely independent from the agent being evaluated. The judge receives:

- A **rubric** (from `rubrics/*.md`) defining what each score 1-5 means, plus hard guardrails
- The **question**
- The agent's **full trajectory** (every tool call with its input and output)
- The agent's **final answer**
- The **gold answer**

Critically, the judge does NOT see the gold `supporting_facts` — so it can't cheat by knowing which paragraphs were needed.

The judge returns a JSON response: `{"reasoning": "brief explanation", "score": N}`. Each rubric has **hard guardrails** that override normal scoring (e.g., "if the agent retrieved nothing but still answered, max score is 2").

**Why LLM-as-judge instead of more deterministic metrics?** It catches failure modes that string matching can't:
- An agent that gets the right answer by coincidence (good F1 but no reasoning)
- An agent that searches well but hallucinates an answer unrelated to what it found
- An agent that copy-pastes the whole question as a search query (lazy but might still retrieve results)

**The three dimensions**:

| Dimension | What the Judge Evaluates | Key Guardrails |
|-----------|-------------------------|----------------|
| **Groundedness** | Is the answer supported by evidence the agent actually retrieved? | No paragraphs retrieved → max 2. Contradicts evidence → 1. |
| **Reasoning Coherence** | Does the multi-hop chain make logical sense? | Only one search for a multi-hop question → max 3. Contradicts own evidence → 1. |
| **Search Strategy** | Were search queries well-chosen and targeted? | Verbatim copy of full question as query → max 3. No searches → 1. |

**`reasoning_score`** = average of the three dimensions, each normalized from 1-5 to 0-1.

## Composite Score

```
composite = 0.4 * answer_score + 0.35 * trajectory_score + 0.25 * reasoning_score
```

The weighting ensures:
- Getting the right answer with a bad trajectory scores poorly
- A good trajectory with a wrong answer also scores poorly
- The agent must improve both *what* it answers and *how* it gets there

## CLI Output Format (`python evaluate.py`)

```
Case 1:  answer_f1=0.06  trajectory=0.53  reasoning=2.3  (hard/bridge)
Case 2:  answer_f1=0.04  trajectory=0.63  reasoning=5.0  (hard/comparison)
...
---
answer_score: 0.0611
  answer_f1: 0.0611
  answer_em: 0.0000
trajectory_score: 0.5500
  retrieval_f1: 0.4000
  action_order: 0.4750
  efficiency: 1.0000
reasoning_score: 0.6333
  groundedness: 0.5400
  reasoning_coherence: 0.5200
  search_strategy: 0.8400
composite: 0.3753
```

## Diagnosing Failures

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Low answer + high trajectory | Agent finds evidence but gives verbose/wrong answer | Improve answer extraction in agent or prompt |
| High answer + low trajectory | Agent got lucky or hardcoded | Improve search/retrieval strategy |
| Low trajectory + low reasoning | Agent isn't searching properly | Add multi-hop reasoning to prompt |

## Ground Truth Trajectory

HotpotQA provides `supporting_facts` — the exact `[title, sentence_id]` pairs needed to answer. `setup_dataset.py` converts these into an ideal tool call sequence:

```
Bridge question:
  search("Akademisches Kunstmuseum") → read("Akademisches Kunstmuseum")
  → search("Bonn") → read("Bonn") → answer

Comparison question:
  search("Roger Donaldson") → read("Roger Donaldson")
  → search("André Cayatte") → read("André Cayatte") → compare → answer
```

This gold trajectory is what Layer 2 evaluation compares the agent's actual behavior against.
