"""
Streamlit demo app for the HotpotQA agent evaluation pipeline.

Entry point: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Agent Eval",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Agent Eval")
st.markdown("#### How to evaluate an AI agent — end to end")

st.markdown("""
This app demonstrates a complete evaluation pipeline for an AI agent that answers
multi-hop questions. We score the agent on **what it answers**, **how it finds evidence**,
and **whether its reasoning makes sense** — all tracked in Langfuse.
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# The Setup
# ══════════════════════════════════════════════════════════════════════════════

st.header("The Setup")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Agent")
    st.markdown("""
    A **Claude Sonnet 4.6** agent with two tools:

    | Tool | What it does |
    |------|-------------|
    | `search_paragraphs(query)` | Keyword search over context paragraphs |
    | `read_paragraph(title)` | Read a specific paragraph by title |

    The agent receives a question and 10 context paragraphs.
    Only **2 are relevant** — the other 8 are distractors.
    It must figure out which to read and how to combine them.
    """)

with col2:
    st.subheader("The Dataset")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Questions", "20")
    col_b.metric("Bridge", "10")
    col_c.metric("Comparison", "10")

    st.markdown("""
    **Bridge questions** require chained reasoning:
    > *Find A → use A to discover B → combine A+B to answer*

    **Comparison questions** require parallel retrieval:
    > *Find A, find B → compare them*

    Source: [HotpotQA](https://hotpotqa.github.io/) dev set
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# The 3-Layer Evaluation
# ══════════════════════════════════════════════════════════════════════════════

st.header("The 3-Layer Evaluation")

st.markdown("""
Most agent evals only check if the answer is correct. We go deeper — scoring
the agent's **answer**, its **trajectory** (tool calls), and its **reasoning quality**.
""")

tab1, tab2, tab3 = st.tabs([
    "📝 Layer 1: Answer Score (40%)",
    "🔍 Layer 2: Trajectory Score (35%)",
    "🧠 Layer 3: Reasoning Score (25%)",
])

with tab1:
    st.markdown("### Did the agent get the right answer?")
    st.info("**Deterministic** — pure string comparison, no LLM involved")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **F1 Score** — Token-level precision and recall

        | | Example |
        |---|---|
        | Gold answer | `Bonn` |
        | Agent answer | `The city of Bonn has a population of 327,913` |
        | Precision | 1/9 = 11% (1 matching token out of 9) |
        | Recall | 1/1 = 100% (found the 1 gold token) |
        | **F1** | **20%** |

        The agent *found* the answer but buried it in a verbose response.
        """)
    with col2:
        st.markdown("""
        **Exact Match** — Yes or No

        After normalization (lowercase, strip articles, remove punctuation):

        | | |
        |---|---|
        | `"Barack Obama"` vs `"Barack Obama"` | ✅ Yes |
        | `"Barack Obama"` vs `"Obama"` | ❌ No |
        | `"The United States"` vs `"united states"` | ✅ Yes |

        Strict but fair — the standard HotpotQA metric.
        """)

    st.warning("**This is the naive agent's bottleneck** — it gives verbose answers (F1 ~8%) instead of concise ones.")

with tab2:
    st.markdown("### Did the agent find the right evidence in the right order?")
    st.info("**Deterministic** — compares the agent's tool calls against the gold trajectory")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Retrieval F1** (40% of trajectory)

        Did the agent find the gold paragraphs?

        ```
        Gold:      {Bonn, Akademisches Kunstmuseum}
        Agent:     {Bonn, Akademisches Kunstmuseum, Berlin}

        Precision: 2/3 = 67%
        Recall:    2/2 = 100%
        F1:        80%
        ```
        """)

    with col2:
        st.markdown("""
        **Action Order** (40% of trajectory)

        Did the agent follow the right tool sequence?
        Uses Longest Common Subsequence (LCS):

        ```
        Gold:  [search, read, search, read]
        Agent: [search, search, read, read]

        LCS = 3 out of 4 → 75%
        ```

        Rewards correct ordering, forgives extra calls.
        """)

    with col3:
        st.markdown("""
        **Efficiency** (20% of trajectory)

        Did the agent waste tool calls?

        ```
        Gold needs:  4 calls
        Agent used:  4 calls → 100%
        Agent used:  7 calls →  57%
        Agent used:  2 calls → 100%*
        ```

        *Too few calls get caught by retrieval F1 instead.
        """)

    st.code("trajectory_score = 0.4 * retrieval_f1 + 0.4 * action_order + 0.2 * efficiency", language=None)

with tab3:
    st.markdown("### Does the agent's reasoning actually make sense?")
    st.info("**LLM-as-judge** — a separate Claude call scores each dimension using a rubric")

    st.markdown("""
    Some things can't be measured with string matching:
    - Did the agent get lucky, or did it actually reason through the problem?
    - Is the answer supported by what it retrieved, or did it hallucinate?
    - Were the search queries targeted, or did it just paste the whole question?

    We make **3 separate Claude API calls**, each with a rubric defining what 1-5 means:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Groundedness")
        st.markdown("""
        *Is the answer backed by retrieved evidence?*

        | Score | Meaning |
        |-------|---------|
        | 5 | Every claim traces to a paragraph |
        | 3 | Partially supported |
        | 1 | Contradicts evidence |

        **Guardrail**: Retrieved nothing but answered → max 2
        """)

    with col2:
        st.markdown("#### Reasoning Coherence")
        st.markdown("""
        *Does the multi-hop chain make sense?*

        | Score | Meaning |
        |-------|---------|
        | 5 | Clear chain: A → B → answer |
        | 3 | Found info but gaps in logic |
        | 1 | No reasoning, just guessing |

        **Guardrail**: Only 1 search for multi-hop → max 3
        """)

    with col3:
        st.markdown("#### Search Strategy")
        st.markdown("""
        *Were the queries well-chosen?*

        | Score | Meaning |
        |-------|---------|
        | 5 | Targeted, decomposed queries |
        | 3 | Relevant but unfocused |
        | 1 | Irrelevant or no searches |

        **Guardrail**: Copy-pasted full question → max 3
        """)

    st.markdown("""
    The judge sees: question, full trajectory (tool calls + outputs), agent answer, gold answer.
    It does **not** see the gold supporting facts — so it can't cheat.
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Composite Score
# ══════════════════════════════════════════════════════════════════════════════

st.header("Composite Score")

col1, col2 = st.columns([2, 1])

with col1:
    st.code("composite = 0.4 * answer + 0.35 * trajectory + 0.25 * reasoning", language=None)
    st.markdown("""
    The weighting ensures:
    - Getting the right answer with a bad trajectory scores poorly
    - A good trajectory with a wrong answer also scores poorly
    - The agent must improve **both** what it answers and how it gets there
    """)

with col2:
    st.markdown("**Baseline (naive agent)**")
    st.metric("Answer", "6.1%")
    st.metric("Trajectory", "55.0%")
    st.metric("Reasoning", "63.3%")
    st.metric("Composite", "37.5%")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Multi-Layer Comprehensive Evaluation
# ══════════════════════════════════════════════════════════════════════════════

st.header("Multi-Layer Comprehensive Evaluation")

st.markdown("""
Beyond the 3-layer composite, the **Comprehensive Run & Evaluate** page scores the agent
across **all 9 metrics** — giving a complete picture of agent behavior from deterministic
accuracy to LLM-judged reasoning quality to plan-level assessment.
""")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 📏 Deterministic Metrics")
        st.markdown("""
        Fast, reproducible, no LLM involved.

        | Metric | What it measures |
        |--------|-----------------|
        | **Answer F1** | Token-level overlap with gold answer |
        | **Retrieval F1** | Found the right paragraphs? |
        | **Action Order** | Correct tool call sequence (LCS) |
        | **Efficiency** | Minimum calls needed? |

        *Available in both the Core and Comprehensive pages.*
        """)

with col2:
    with st.container(border=True):
        st.markdown("#### 🧠 LLM-Judge Rubric Metrics")
        st.markdown("""
        Separate Claude calls with rubric-driven scoring (1–5).

        | Metric | What it measures |
        |--------|-----------------|
        | **Groundedness** | Answer backed by retrieved evidence? |
        | **Reasoning Coherence** | Multi-hop chain makes sense? |
        | **Search Strategy** | Queries well-chosen & targeted? |

        *Catches hallucination and reasoning gaps that F1 misses.*
        """)

with col3:
    with st.container(border=True):
        st.markdown("#### 🗺️ Plan-Level Metrics")
        st.markdown("""
        LLM judge evaluates the agent's planning behavior.

        | Metric | What it measures |
        |--------|-----------------|
        | **Plan Quality** | Decomposed into targeted sub-queries? |
        | **Plan Adherence** | Followed through on search plan? |

        *Evaluates the cognitive strategy, not just the output.*
        """)

st.info(
    "💡 Use **Run & Evaluate** in the sidebar for fast core-metric runs (4 metrics, no LLM judge calls). "
    "Use **Comprehensive Run & Evaluate** for the full 9-metric evaluation with failure diagnosis, "
    "plan quality, tool efficacy, and cost tracking."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Failure Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

st.header("Diagnosing Failures")

st.markdown("The per-question breakdown reveals *why* the agent fails:")

col1, col2, col3 = st.columns(3)

with col1:
    st.error("**Low answer + High trajectory**")
    st.markdown("Agent finds the right evidence but gives a verbose or wrong answer. Fix: improve answer extraction.")

with col2:
    st.error("**High answer + Low trajectory**")
    st.markdown("Agent got lucky or hardcoded. Fix: improve search and retrieval strategy.")

with col3:
    st.error("**Low trajectory + Low reasoning**")
    st.markdown("Agent isn't searching properly at all. Fix: add multi-hop reasoning to the prompt.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# The Showcase Narrative
# ══════════════════════════════════════════════════════════════════════════════

st.header("📖 Evaluation Story: How It Got There, Not Just What It Said")

st.markdown("""
> *"We evaluated not just what the agent said — but how it got there."*

Standard QA benchmarks stop at the final answer. Pass or fail. That's not enough when
your agent uses tools, retrieves evidence, and chains multi-hop reasoning.
An agent can score 100% by **getting lucky** — or crash to 0% because it asked one
wrong search query even though its reasoning was perfect. Answer-only metrics miss both.

This pipeline evaluates **three interdependent layers** simultaneously:
""")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 1️⃣ What did it answer?")
        st.markdown("""
        **Token-level F1 & Exact Match**

        The output signal. Did the final answer match the gold answer?
        Strict, reproducible, zero hallucination.

        *But a correct answer can hide a broken process.*
        """)

with col2:
    with st.container(border=True):
        st.markdown("#### 2️⃣ How did it gather evidence?")
        st.markdown("""
        **Trajectory quality: retrieval + order + efficiency**

        The process signal. Did the agent find the right paragraphs?
        In the right order? Without wasted calls?

        *Evidence of reasoning, not just luck.*
        """)

with col3:
    with st.container(border=True):
        st.markdown("#### 3️⃣ Did it actually reason?")
        st.markdown("""
        **LLM-as-judge: groundedness, coherence, search strategy**

        The cognition signal. Was the answer grounded in what it retrieved?
        Did the multi-hop chain make sense?

        *Catches hallucination that metrics miss.*
        """)

st.markdown("""
Only an agent that scores well on **all three** layers can be trusted at scale.
A high answer F1 with a poor trajectory means it got lucky.
A good trajectory with a low reasoning score means it searched correctly but
drew the wrong conclusion — a synthesis bug, not a retrieval bug.

**This decomposition is the difference between knowing an agent is unreliable
and knowing exactly *why* — and what to fix.**
""")

st.info(
    "💡 **New in this release**: "
    "**Run & Evaluate** now focuses on 4 core deterministic metrics (Answer F1, Retrieval F1, "
    "Action Order, Efficiency) for fast iteration. "
    "**Comprehensive Run & Evaluate** adds all 9 metrics including LLM-judge rubric scores "
    "(Groundedness, Reasoning, Search Strategy) and plan-level scores (Plan Quality, Plan Adherence) "
    "with failure diagnosis, sub-goal decomposition, tool efficacy, and cost tracking. "
    "Run `python consistency_test.py` to measure brittleness across rephrased questions. "
    "Use **Failure Analysis** in the sidebar to see retrieval vs synthesis vs planning "
    "breakdown across all dataset items."
)

st.divider()

st.caption(
    "Navigate using the sidebar: Dataset Explorer → Run & Evaluate (Core) → Core Feedback → "
    "Comprehensive Run & Evaluate → Comprehensive Feedback → Failure Analysis"
)
