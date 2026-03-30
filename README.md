# agent-evaluation-v0

An experiment combining **Langfuse-based agent evaluation** with **autoresearch** — an automated loop where Claude Code iteratively improves a QA agent's code and prompts based on eval feedback.

The story: *"I gave an AI agent a broken QA system and watched it fix itself — tracked every improvement in Langfuse."*

## Quick Start

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Set up .env with your API keys
cp .env.example .env  # then fill in keys

# 3. Download HotpotQA + upload dataset to Langfuse
python setup_dataset.py

# 4. Run evaluation (baseline)
python evaluate.py

# 5. (Phase 2) Start the autoresearch loop
claude "Read program.md and follow its instructions exactly. Start now."
```

## Project Structure

```
agent-evaluation-v0/
│
│ ── Phase 1: Agent + Evaluation Pipeline ──
├── setup_dataset.py      # Downloads HotpotQA, picks 10 questions, uploads to Langfuse
├── agent.py              # Claude QA agent with search/read tools (EDITABLE by autoresearch)
├── system_prompt.md      # Agent instructions (EDITABLE by autoresearch)
├── evaluate.py           # 3-layer scoring, uploads results to Langfuse (read-only)
├── rubrics/              # LLM-as-judge scoring guides (read-only)
│   ├── groundedness.md
│   ├── reasoning_coherence.md
│   └── search_strategy.md
├── requirements.txt      # anthropic, langfuse, requests, python-dotenv
├── .env                  # API keys (not committed)
├── .gitignore
│
│ ── Phase 2: Autoresearch Loop ──
├── program.md            # Loop instructions for Claude Code (read-only)
├── results.tsv           # Score log across iterations (append-only)
│
│ ── Docs ──
├── PLAN.md               # Detailed implementation plan
└── README.md             # This file
```

---

## Dataset

**Source**: [HotpotQA](https://hotpotqa.github.io/) dev distractor set (7,405 multi-hop questions).

We select **10 questions** (5 bridge + 5 comparison) and upload them to Langfuse as dataset `rittika-hotpotqa-10`. The upload uses deterministic item IDs (`rittika-hotpotqa-10-0` through `rittika-hotpotqa-10-9`) so `setup_dataset.py` is safe to re-run without creating duplicates.

### Question Types

| Type | Reasoning Pattern | Example |
|------|------------------|---------|
| **Bridge** | Chained: find A → use A to find B → combine | "What is the population of the city where the Akademisches Kunstmuseum is located?" |
| **Comparison** | Parallel: find A, find B → compare | "Were both Roger Donaldson and André Cayatte french filmmakers?" |

### What Each Dataset Item Contains

| Field | Contents |
|-------|----------|
| `input.question` | The multi-hop question |
| `input.context` | 10 paragraphs as `[title, [sentences...]]` — only 2 are relevant, 8 are distractors |
| `expected_output.answer` | Gold answer (short string like "Yes", "327,913") |
| `expected_output.trajectory` | Ground truth trajectory (ideal tool call sequence, built from `supporting_facts`) |
| `metadata` | Question type, level, HotpotQA ID, supporting_facts |

### Ground Truth Trajectory

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

---

## The Agent (`agent.py`)

A Claude-powered QA agent with two tools:

| Tool | Description |
|------|-------------|
| `search_paragraphs(query)` | Keyword search over the context paragraphs. Splits query into terms, returns any paragraph whose title + sentences contain at least one term. |
| `read_paragraph(title)` | Returns all sentences for a specific paragraph by exact title match. |

**Agent loop**: send question to Claude → if Claude uses tools, execute them and send results back → repeat until Claude responds with a text answer (no tools) → return that as the final answer.

**Starting state** (deliberately naive):
- Minimal system prompt: *"Answer the question. You have access to search_paragraphs and read_paragraph tools."*
- Simple loop with no planning, no multi-step strategy, no answer extraction logic
- Max 10 tool iterations before giving up

Phase 2's autoresearch loop improves `agent.py` and `system_prompt.md` to raise scores.

---

## Evaluation (`evaluate.py`)

Each question is scored across **three independent layers**, each measuring something different. The layers combine into a composite score, but are also reported individually so you can see exactly where the agent is strong or weak.

### Layer 1: Answer Score (deterministic)

Pure string comparison between the agent's answer and the gold answer.

| Metric | How It Works |
|--------|-------------|
| **F1** | Token-level precision/recall after normalization (lowercase, strip articles/punctuation, collapse whitespace). If gold is `"Bonn"` and agent says `"The city of Bonn has a population of 327,913"`, F1 is low because precision is terrible. |
| **Exact Match** | 1 if normalized predicted == normalized gold, else 0. Strict — `"Barack Obama"` vs `"Obama"` is a miss. |

**`answer_score`** = average F1 across all questions.

### Layer 2: Trajectory Score (deterministic)

Compares the agent's actual tool calls against the gold trajectory.

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Retrieval F1** | Did the agent find the right evidence paragraphs? | Set intersection of retrieved titles vs gold titles → precision, recall, F1 |
| **Action Order** | Did the agent follow the right tool sequence? | Longest Common Subsequence of tool names (agent vs gold) / gold length |
| **Efficiency** | Did the agent waste tool calls? | gold action count / max(agent count, gold count) |

**`trajectory_score`** = 0.4 * retrieval_f1 + 0.4 * action_order + 0.2 * efficiency

### Layer 3: Reasoning Score (LLM-as-judge)

Layers 1 and 2 are deterministic — string comparison and sequence matching. But some things can't be measured mechanically: *"Does this reasoning chain actually make sense?"* or *"Is this answer genuinely supported by the evidence?"* That's what LLM-as-judge is for.

**How it works**: For each question, we make 3 separate Claude API calls (using `claude-sonnet-4-20250514`). Each call is a *different* Claude instance acting as a judge — completely independent from the agent being evaluated. The judge receives:

- A **rubric** (from `rubrics/*.md`) defining what each score 1-5 means, plus hard guardrails
- The **question**
- The agent's **full trajectory** (every tool call with its input and output)
- The agent's **final answer**
- The **gold answer**

Critically, the judge does NOT see the gold `supporting_facts` — so it can't cheat by knowing which paragraphs were needed. It has to evaluate the agent's reasoning on its own merits.

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

**`reasoning_score`** = average of the three dimensions, each normalized from 1-5 to 0-1. This is 25% of the composite — the smallest weight because LLM judges have inherent variance, but it catches things the deterministic metrics cannot.

### Composite Score

```
composite = 0.4 * answer_score + 0.35 * trajectory_score + 0.25 * reasoning_score
```

This is **the metric Phase 2 optimizes**. The weighting ensures:
- Getting the right answer with a bad trajectory scores poorly
- A good trajectory with a wrong answer also scores poorly
- The agent must improve both *what* it answers and *how* it gets there

### Output Format

```
Case 1:  answer_f1=0.08  trajectory=0.80  reasoning=5.0  (hard/bridge)
Case 2:  answer_f1=0.04  trajectory=0.63  reasoning=5.0  (hard/comparison)
...
---
answer_score: 0.0528
  answer_f1: 0.0528
  answer_em: 0.0000
trajectory_score: 0.6195
  retrieval_f1: 0.4238
  action_order: 0.6250
  efficiency: 1.0000
reasoning_score: 0.8200
  groundedness: 0.7600
  reasoning_coherence: 0.7800
  search_strategy: 0.9200
composite: 0.4429
```

### Diagnosing Failures

The per-case breakdown helps identify what's wrong:

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Low answer + high trajectory | Agent finds evidence but gives verbose/wrong answer | Improve answer extraction in agent or prompt |
| High answer + low trajectory | Agent got lucky or hardcoded | Improve search/retrieval strategy |
| Low trajectory + low reasoning | Agent isn't searching properly | Add multi-hop reasoning to prompt |

### Langfuse Integration

Each evaluation run:
- Creates a Langfuse experiment with a timestamped name (`hotpotqa_eval_YYYYMMDD_HHMMSS`)
- Wraps each agent call in a Langfuse trace via `item.run()` context manager
- Uploads per-item scores (answer_f1, retrieval_f1, groundedness, etc.) linked to each trace
- All visible in the Langfuse dashboard for comparison across runs

---

## Phase 2: Autoresearch Loop

An automated loop where Claude Code iteratively improves the agent:

1. Run `python evaluate.py` → get current scores
2. Analyze per-case failures (which questions fail? why?)
3. Edit `agent.py` and/or `system_prompt.md` with an improvement
4. Commit, re-evaluate, keep if score improved or revert if worse
5. Repeat

**Rules**: can only edit `agent.py` and `system_prompt.md`. Cannot edit `evaluate.py` or `setup_dataset.py`. Cannot install new packages or hardcode answers.

**Expected progression**: ~0.05 answer_score baseline → 0.80+ after 15-20 iterations.

Progress is logged to `results.tsv` and each iteration creates a new Langfuse experiment.

---

## Requirements

- Python 3.10+
- `uv` for package management
- Anthropic API key (for the QA agent + LLM judge)
- Langfuse account + API keys (for dataset and eval tracking)
- Claude Code (for running the Phase 2 autoresearch loop)

### Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Attribution

- HotpotQA dataset: [Yang et al., 2018](https://hotpotqa.github.io/) — CC BY-SA 4.0
- Evaluation patterns adapted from the [Vector Institute Agentic AI Evaluation Bootcamp](https://github.com/VectorInstitute/agentic-ai-evaluation)
