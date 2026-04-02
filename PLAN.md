# Plan: Autoresearch Agent Experiment with HotpotQA + Langfuse

## Context

We're building an autoresearch experiment where Claude Code iteratively improves a Knowledge QA agent. The agent answers multi-hop questions from HotpotQA. Evals and dataset live in Langfuse. The autoresearch loop edits both the system prompt and agent code to improve scores.

This project combines two ideas:
1. **Agent evaluation** — using Langfuse to track dataset, scores, trajectories, and experiment runs
2. **Autoresearch** — an automated loop where an AI agent iteratively improves code/prompts based on eval feedback

The story for the blog series: *"I gave an AI agent a broken QA system and watched it fix itself — tracked every improvement in Langfuse."*
At the end we should have a strong narrative of how the agent evolved, with concrete examples of improvements and their impact on scores and a blog series and tutorial on how to set up similar experiments.

## Project Phases

### Phase 1: Agent + Langfuse Evaluation Setup
Build and validate the complete evaluation pipeline. At the end of this phase we have:
- A working (naive) Claude QA agent
- 30 HotpotQA questions with ground truth answers + trajectories in Langfuse
- A three-layer evaluation pipeline (answer quality, trajectory quality, reasoning quality)
- Scores visible in Langfuse dashboard
- Everything runs end-to-end: `python evaluate.py` → agent runs → scores appear in Langfuse

**Files**: `setup_dataset.py`, `agent.py`, `system_prompt.md`, `evaluate.py`, `rubrics/`, `requirements.txt`, `.env`, `.gitignore`

### Phase 2: Autoresearch Loop
Wire up the autoresearch loop that iteratively improves the agent. At the end of this phase we have:
- `program.md` with loop instructions for Claude Code
- `results.tsv` tracking score progression
- A demonstrated improvement curve from ~0.1 to ~0.8+ over multiple iterations
- Multiple experiment runs in Langfuse showing the improvement narrative

**Files**: `program.md`, `results.tsv`

**Phase 2 depends on Phase 1 being complete and validated.**

## Background

### HotpotQA Dataset
- **Source**: https://hotpotqa.github.io/
- **Format**: JSON, each example has `_id`, `question`, `answer`, `context` (list of paragraphs), `supporting_facts`, `type` (bridge/comparison), `level` (easy/medium/hard)
- **Why HotpotQA**: Multi-hop questions force the agent to search and reason across multiple paragraphs — trajectory quality directly affects answer quality
- **Download**: Dev distractor set from `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json` (44MB)
- **Subset**: We pick 30 questions — 10 easy, 10 medium, 10 hard (mix of "bridge" and "comparison" types)

### Agent Tools
- We use HotpotQA's provided context paragraphs as a retrieval tool (not real web search)
- The agent gets a `search_paragraphs(query)` tool that searches the provided context paragraphs
- This makes evals **reproducible, free, and fast**
- The agent still needs to figure out *which* paragraphs to read and *how* to combine info across them

### Langfuse Integration
- Dataset stored in Langfuse (30 items with question, expected answer, metadata)
- Each eval run creates a Langfuse experiment
- Scores (F1, EM, trajectory metrics) uploaded per-item
- In Phase 2, the autoresearch loop reads the printed aggregate score

### LLM Provider
- The agent being evaluated uses **Claude (Anthropic SDK)**
- In Phase 2, Claude Code edits the agent's code and prompts via the autoresearch loop

### Eval-Agents Reference
- Patterns borrowed from `/Users/rittikajindal/Library/CloudStorage/OneDrive-ThomsonReutersIncorporated/personal_projects/eval-agents/implementations/`
- Key patterns: Google ADK agents, Langfuse tracing, three-layer evaluation (item + trace + run), LLM-as-judge
- We adapt these patterns for Claude + Langfuse evaluation (Phase 1) and autoresearch (Phase 2)

## Architecture

```
agent-evaluation-v0/
├── PLAN.md               # This file
│
│ ── Phase 1: Agent + Langfuse Evaluation ──
├── setup_dataset.py      # One-time: downloads 30 HotpotQA questions, uploads to Langfuse (read-only)
├── agent.py              # Claude-based QA agent with tool use
├── system_prompt.md      # Agent instructions
├── evaluate.py           # Runs agent on Langfuse dataset, scores, returns aggregate (read-only)
├── rubrics/              # LLM-as-judge scoring guides (read-only)
│   ├── groundedness.md
│   ├── reasoning_coherence.md
│   └── search_strategy.md
├── requirements.txt      # anthropic, langfuse, requests
├── .env                  # API keys (ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
├── .gitignore            # .env, __pycache__, *.pyc, hotpot_dev_distractor_v1.json
│
│ ── Phase 2: Autoresearch Loop ──
├── program.md            # Autoresearch loop instructions for Claude Code (read-only)
└── results.tsv           # Score log (append-only)
```

## Phase 1 Files — Detailed Specifications

### 1. `setup_dataset.py` (one-time setup, read-only)

**Purpose**: Download HotpotQA, pick 30 questions, build ground truth trajectories, upload to Langfuse as a dataset.

**Steps**:
1. Download `hotpot_dev_distractor_v1.json` from CMU if not already cached locally
2. Filter and select 30 questions:
    - 10 easy (5 bridge + 5 comparison)
    - 10 medium (5 bridge + 5 comparison)
    - 10 hard (5 bridge + 5 comparison)
3. For each selected question, **build a ground truth trajectory** from the `supporting_facts` field:
    - HotpotQA's `supporting_facts` is a list of `[title, sent_id]` pairs — the exact paragraphs and sentences needed
    - From this, we construct the ideal tool call sequence:
        - For **bridge** questions (chain reasoning): the trajectory is ordered — search for entity A → read paragraph A → search for entity B (found via A) → read paragraph B → answer
        - For **comparison** questions: the trajectory is parallel — search for entity A → read paragraph A → search for entity B → read paragraph B → compare → answer
    - Ground truth trajectory format:
      ```json
      {
        "actions": [
          {"tool": "search_paragraphs", "input": {"query": "..."}, "expected_result_title": "Title A"},
          {"tool": "read_paragraph", "input": {"title": "Title A"}},
          {"tool": "search_paragraphs", "input": {"query": "..."}, "expected_result_title": "Title B"},
          {"tool": "read_paragraph", "input": {"title": "Title B"}}
        ],
        "expected_paragraphs": ["Title A", "Title B"],
        "reasoning_type": "bridge" | "comparison",
        "reasoning_description": "Find X from paragraph A, then use X to find Y in paragraph B"
      }
      ```
    - **Note**: The `query` field in the ground truth trajectory is a *description* of what should be searched for, not the exact query string. The evaluator checks whether the agent searched for the right *concepts*, not exact string matches.

4. For each selected question, create a Langfuse dataset item:
    - `input`: `{"question": "...", "context": [...paragraphs...]}`
    - `expected_output`: `{"answer": "...", "trajectory": {...ground truth trajectory...}}`
    - `metadata`: `{"type": "bridge"|"comparison", "level": "easy"|"medium"|"hard", "supporting_facts": [...], "hotpotqa_id": "..."}`
5. Upload to Langfuse dataset named `"rittika-hotpotqa-30"`

**Key detail**: The `context` field contains the paragraphs the agent can search through. Each paragraph is `[title, [sentence1, sentence2, ...]]`. The agent's `search_paragraphs` tool searches within these.

**Ground truth trajectory generation**: We use `supporting_facts` + the question structure to build ideal trajectories. For bridge questions, the `supporting_facts` paragraphs have an implicit order (first paragraph leads to second). For comparison questions, the two paragraphs are independent. The script uses a heuristic to determine the order based on which entity appears in the question vs which entity is discovered.

### 2. `agent.py`

**Purpose**: Claude-based QA agent that answers multi-hop questions using tool use.

**Interface**:
```python
def run_agent(question: str, context: list) -> dict:
    """
    Args:
        question: The question to answer
        context: List of [title, [sentences...]] paragraphs to search

    Returns:
        {
            "answer": str,           # The agent's answer
            "trajectory": list,      # List of tool calls: [{"tool": "search_paragraphs", "input": {...}, "output": ...}, ...]
            "token_usage": int,      # Total tokens used
        }
    """
```

**Tools available to the agent**:
- `search_paragraphs(query: str) -> list[dict]`: Searches the context paragraphs by keyword matching. Returns matching paragraphs with titles and sentences. The agent uses this to find relevant information.
- `read_paragraph(title: str) -> str`: Returns all sentences for a specific paragraph title. For when the agent knows which paragraph to read.

**Starting state (deliberately naive)**:
- Minimal system prompt: "Answer the question. You have access to search_paragraphs and read_paragraph tools."
- Simple loop: call Claude once, if it uses a tool execute it and call again, repeat until it gives a final answer
- No planning, no multi-step strategy, no answer extraction logic
- This should score ~0.1-0.3 F1 on the 30 questions

### 3. `system_prompt.md`

**Starting content**:
```
Answer the question. You have access to search_paragraphs and read_paragraph tools.
```

### 4. `rubrics/` (read-only)

LLM-as-judge scoring guides. Each rubric defines what each score (1-5) means, hard guardrails, and scoring instructions. `evaluate.py` loads these rubrics and includes them in the judge prompt.

#### `rubrics/groundedness.md`

Scores whether the agent's answer is supported by the paragraphs it actually retrieved.

| Score | Criteria |
|-------|----------|
| 5 | Answer directly and fully follows from retrieved evidence. Every claim is traceable to a specific retrieved paragraph. |
| 4 | Answer is mostly supported by evidence, with minor inferences that are reasonable. |
| 3 | Answer is partially supported. Some claims are grounded, others are inferred or weakly connected. |
| 2 | Answer has weak grounding. Most claims are not directly supported by retrieved evidence. |
| 1 | Answer contradicts retrieved evidence, or has no basis in what the agent actually read. |

**Hard guardrails**:
- If the agent retrieved no paragraphs but still answered: max score = 2
- If the answer directly contradicts a retrieved paragraph: score = 1
- If the agent answered "I don't know" or equivalent: score = 3 (honest when lacking evidence)

#### `rubrics/reasoning_coherence.md`

Scores whether the agent's reasoning chain makes logical sense for a multi-hop question.

| Score | Criteria |
|-------|----------|
| 5 | Clear multi-hop chain: found fact A, explicitly used A to find fact B, combined A+B to answer. Each step logically follows from the previous. |
| 4 | Mostly coherent chain with minor gaps. The multi-hop structure is visible but one connection is implicit rather than explicit. |
| 3 | Some reasoning visible but significant jumps or gaps. The agent found relevant information but didn't clearly connect the hops. |
| 2 | Weak reasoning. The agent retrieved some information but the connection between retrieval steps and the final answer is unclear. |
| 1 | No visible reasoning. The agent appears to guess, or its reasoning contradicts its own evidence. |

**Hard guardrails**:
- If the agent made only one search for a multi-hop question: max score = 3
- If the agent's reasoning contradicts its own tool outputs: score = 1

#### `rubrics/search_strategy.md`

Scores whether the agent's search queries were well-chosen for the question.

| Score | Criteria |
|-------|----------|
| 5 | Targeted, efficient queries that decompose the multi-hop question. First query finds entity A, second query uses information from A to find entity B. No redundant searches. |
| 4 | Good queries with minor inefficiency. Mostly targeted but one query could have been more specific. |
| 3 | Reasonable but unfocused queries. The agent searched for relevant topics but didn't decompose the question into precise sub-queries. |
| 2 | Weak queries. The agent copied the whole question as a search, or searched for loosely related terms. |
| 1 | Irrelevant queries. The agent searched for unrelated terms, or didn't search at all. |

**Hard guardrails**:
- If the agent's first query is a verbatim copy of the full question: max score = 3
- If the agent made no searches: score = 1

### 5. `evaluate.py` (read-only)

**Purpose**: Run the agent on all 30 Langfuse dataset items, score each across three evaluation layers, print aggregate.

#### Layer 1: Answer Quality (Item-level, deterministic)

Per-question scoring on the agent's final answer vs ground truth:

- **Exact Match (EM)**: 1 if normalized(predicted) == normalized(gold), else 0
    - Normalization: lowercase, strip articles (a/an/the), strip punctuation, collapse whitespace
- **F1**: Token-level precision/recall/F1 between normalized predicted and gold answer tokens
    - Precision = |predicted ∩ gold| / |predicted|
    - Recall = |predicted ∩ gold| / |gold|
    - F1 = 2 * P * R / (P + R) if P + R > 0, else 0

#### Layer 2: Trajectory Quality (Trace-level, deterministic)

Per-question scoring on the agent's actual trajectory vs the **ground truth trajectory** from the dataset.

##### 2a. Paragraph Retrieval (did the agent find the right evidence?)

- **Paragraph Recall**: Of the gold paragraphs, how many did the agent retrieve?
    - `paragraph_recall = |retrieved_titles ∩ gold_titles| / |gold_titles|`
    - Score 0.0-1.0. A score of 1.0 means the agent found all the evidence it needed.

- **Paragraph Precision**: Of the paragraphs the agent retrieved, how many were gold?
    - `paragraph_precision = |retrieved_titles ∩ gold_titles| / |retrieved_titles|`
    - Score 0.0-1.0. A score of 1.0 means no wasted reads.

- **Retrieval F1**: Harmonic mean of paragraph precision and recall.
    - `retrieval_f1 = 2 * P * R / (P + R)`

##### 2b. Action Sequence (did the agent follow the right steps?)

Compare the agent's tool call sequence against the ground truth trajectory's `actions` list:

- **Action Type Match**: Did the agent use the right tools in the right order?
    - Extract tool names from agent trajectory: `["search_paragraphs", "read_paragraph", "search_paragraphs", "read_paragraph"]`
    - Compare against gold trajectory action types using longest common subsequence (LCS)
    - `action_order_score = LCS(agent_actions, gold_actions) / len(gold_actions)`
    - Score 0.0-1.0. Rewards agents that follow the correct search→read→search→read pattern.

- **Reasoning Type Match**: Did the agent's approach match the question type? (binary 0/1)
    - For bridge questions: did the agent use output from the first search to inform the second search?
    - For comparison questions: did the agent search for both entities?
    - Detected heuristically: check if the second search query contains terms from the first search result.

##### 2c. Efficiency (did the agent waste effort?)

- **Search Efficiency**: Tool calls relative to minimum needed.
    - `efficiency = len(gold_actions) / max(len(agent_actions), len(gold_actions))`
    - Score 0.0-1.0. Penalizes both too few calls (missed evidence) and too many (redundant).

- **Redundancy Ratio**: Fraction of duplicate/near-duplicate searches.
    - `redundancy = 1.0 - (unique_queries / total_queries)` if total_queries > 0, else 0
    - Lower is better. Reported but not included in composite score.

##### 2d. Trajectory composite
- `trajectory_score = 0.4 * retrieval_f1 + 0.4 * action_order_score + 0.2 * efficiency`

#### Layer 3: Reasoning Quality (Trace-level, LLM-as-judge)

Per-question scoring using Claude as a judge on the agent's reasoning trace:

- **Groundedness** (1-5): Is the final answer supported by the paragraphs the agent actually retrieved?
    - 5 = Answer directly follows from retrieved evidence
    - 3 = Partially supported, some inference
    - 1 = Answer contradicts or has no basis in retrieved evidence

- **Reasoning Coherence** (1-5): Did the agent's chain of thought make logical sense?
    - 5 = Clear multi-hop chain: found A, used A to find B, combined to answer
    - 3 = Some reasoning visible but jumps or gaps
    - 1 = No visible reasoning, random guessing

- **Search Strategy** (1-5): Were the agent's search queries well-chosen?
    - 5 = Targeted, efficient queries that decompose the multi-hop question
    - 3 = Reasonable but unfocused queries
    - 1 = Irrelevant or copy-pasted-question queries

**Note on LLM-as-judge**: Uses a separate Claude call with a rubric. The judge sees: the question, the agent's full trajectory (tool calls + responses), the agent's final answer, and the ground truth answer. It does NOT see the gold supporting_facts (to avoid leaking that info).

#### Aggregate Scores

**Primary metric** (what Phase 2 optimizes):
- `composite_score = 0.4 * average_answer_f1 + 0.35 * average_trajectory_score + 0.25 * average_reasoning_quality`
- This weights answer correctness and trajectory quality roughly equally, with reasoning quality as a bonus
- Getting the right answer with a bad trajectory scores poorly; a good trajectory with a wrong answer also scores poorly

**Secondary metrics** (tracked in Langfuse, visible in dashboard):
- `average_answer_f1`: Mean answer F1 across 30 questions
- `average_em`: Mean exact match across 30 questions
- `average_retrieval_f1`: Mean paragraph retrieval F1 across 30 questions
- `average_action_order`: Mean action sequence match across 30 questions
- `average_efficiency`: Mean search efficiency across 30 questions
- `average_trajectory_score`: Mean trajectory composite (retrieval + action order + efficiency)
- `average_groundedness`: Mean groundedness score (1-5, normalized to 0-1)
- `average_reasoning_coherence`: Mean reasoning coherence (1-5, normalized to 0-1)
- `average_search_strategy`: Mean search strategy score (1-5, normalized to 0-1)

#### Langfuse Integration

- Creates a new experiment run in Langfuse for each evaluation
- Each agent call is wrapped in a Langfuse trace (captures full tool call sequence)
- Per-item scores uploaded: answer_f1, answer_em, retrieval_f1, paragraph_precision, paragraph_recall, groundedness, reasoning_coherence, search_strategy
- Aggregate scores uploaded as run-level metrics
- Experiment name includes timestamp for tracking across iterations

#### Output format:

```
Case 1:  answer_f1=0.40  trajectory=0.55  reasoning=3.0  (easy/bridge)
Case 2:  answer_f1=0.00  trajectory=0.00  reasoning=1.0  (medium/comparison)
...
Case 30: answer_f1=1.00  trajectory=0.90  reasoning=5.0  (hard/bridge)
---
answer_f1: 0.2833
trajectory_score: 0.3100
reasoning_avg: 2.4000
score: 0.2683
```

The per-case breakdown helps diagnose failures:
- Low `answer_f1` + high `trajectory` = agent found the right evidence but extracted wrong answer
- High `answer_f1` + low `trajectory` = agent got lucky or hardcoded, didn't follow the right process
- Low `trajectory` + low `reasoning` = agent isn't searching properly at all

### 6. `requirements.txt`

```
anthropic
langfuse
requests
```

### 7. `.env` (not committed)

```
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 8. `.gitignore`

```
.env
__pycache__/
*.pyc
hotpot_dev_distractor_v1.json
run.log
```

### Phase 1 Verification

1. `pip install -r requirements.txt` — install deps
2. `python setup_dataset.py` — downloads HotpotQA, picks 30 questions, builds ground truth trajectories, uploads to Langfuse
3. Verify dataset appears in Langfuse dashboard under `rittika-hotpotqa-30` with 30 items, each containing question, gold answer, and gold trajectory
4. `python evaluate.py` — runs naive agent on all 30 questions, prints per-case scores across all three layers, prints composite `score: ~0.05-0.15`
5. Verify experiment run appears in Langfuse with per-item scores (answer_f1, trajectory_score, groundedness, reasoning_coherence, search_strategy)
6. Verify aggregate metrics visible in Langfuse dashboard

**Phase 1 is complete when**: `python evaluate.py` runs end-to-end without errors, produces a composite score, and all scores are visible in Langfuse.

---

## Phase 2 Files — Detailed Specifications

### 9. `program.md` (autoresearch loop instructions, read-only)

**Content**: Instructions for Claude Code to follow in the autoresearch loop.

**Rules**:
- **Can edit**: `agent.py`, `system_prompt.md`
- **Cannot edit**: `evaluate.py`, `setup_dataset.py`
- **Cannot**: install new packages beyond requirements.txt, hardcode answers
- **Goal**: maximize the `score:` output from `python evaluate.py`

**What the autoresearch loop can improve in `agent.py`**:
- Tool use logic (when to search vs read, how many iterations)
- Answer extraction (parsing, formatting)
- Error handling (retries, fallbacks)

**What the autoresearch loop can improve in `system_prompt.md`**:
- Add chain-of-thought instructions
- Add multi-hop reasoning strategy ("first find X, then use X to find Y")
- Add answer format instructions ("give a short, precise answer")
- Add tool use guidelines ("always search before answering")

**Loop**:
1. Read all files for context
2. Run `python evaluate.py` → get baseline score
3. Log baseline to `results.tsv`
4. LOOP:
   a. Analyze per-case scores — which cases fail? Why? (use the diagnostic pattern: low answer + high trajectory = extraction problem, etc.)
   b. Edit `agent.py` and/or `system_prompt.md` with an improvement
   c. `git commit` the change
   d. Run `python evaluate.py > run.log 2>&1`
   e. Read score from `run.log`
   f. Log to `results.tsv`
   g. If score improved → keep. If worse → `git reset --hard HEAD~1`
   h. Repeat

**Hints for the autoresearch agent**:
- Study evaluate.py to understand what the composite score measures — it's not just answer correctness
- The trajectory score (35% of composite) rewards finding the right paragraphs in the right order
- The reasoning score (25% of composite) rewards groundedness and coherent multi-hop chains
- Multi-hop questions need multiple searches — add logic for iterative reasoning
- Answer format matters — "Barack Obama" vs "Obama" can affect F1
- Look at which difficulty levels and question types fail most and target those
- Check if failures are answer extraction problems or retrieval problems — the fix is different

### 10. `results.tsv`

**Format**:
```
commit	score	status	description
```

- commit: git short hash (7 chars)
- score: composite score from evaluate.py
- status: `keep`, `discard`, or `crash`
- description: what was changed

### Phase 2 Verification

1. Start autoresearch loop: `claude "Read program.md and follow its instructions exactly. Start now."`
2. Watch `results.tsv` grow as the loop iterates
3. Verify each iteration creates a new experiment run in Langfuse
4. Check Langfuse dashboard for improving scores across runs
5. Confirm the improvement curve: ~0.05-0.15 baseline → ~0.80+ after 15-20 iterations

**Phase 2 is complete when**: The autoresearch loop has run 15+ iterations, `results.tsv` shows a clear improvement curve, and Langfuse shows multiple experiment runs with progressively better scores.

---

## Why This Won't Be a One-Shot Solve

1. **Naive starting state**: The agent starts with a one-line system prompt and no reasoning strategy
2. **Multi-hop complexity**: Questions require finding fact A, using it to find fact B, combining them
3. **Tool use strategy**: The naive agent may not search at all, or search once and guess
4. **Answer extraction**: The agent needs to learn to give concise answers, not full paragraphs
5. **Difficulty progression**: Easy questions may work early, but medium/hard need iterative improvement
6. **Trajectory quality is 50% of the score**: Getting the right answer isn't enough — the agent also needs to find the right evidence paragraphs and reason coherently. A lucky guess with no evidence scores poorly.
7. **LLM-as-judge is hard to game**: The reasoning quality scores (groundedness, coherence, search strategy) can't be hacked by string matching — the agent must genuinely improve its reasoning process.

**Expected improvement trajectory**:
| Iteration | What improves | Approx composite score |
|-----------|--------------|----------------------|
| 0 | Baseline: minimal prompt, basic loop | 0.05-0.15 |
| 1-2 | Better system prompt (encourage search, be concise) | 0.15-0.25 |
| 3-5 | Multi-hop reasoning (search → read → search again) | 0.30-0.40 |
| 6-8 | Targeted retrieval (better search queries, read supporting paragraphs) | 0.40-0.50 |
| 9-11 | Answer extraction + reasoning chain improvements | 0.50-0.60 |
| 12-14 | Comparison-type question strategy, edge case handling | 0.60-0.70 |
| 15-18 | Fine-tuning search efficiency, reducing redundant calls | 0.70-0.80 |
| 19+ | Hard question strategies, groundedness improvements | 0.80+ |

