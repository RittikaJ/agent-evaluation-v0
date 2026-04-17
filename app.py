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

# ── Sidebar: persistent navigation helper ────────────────────────────────────

with st.sidebar:
    st.markdown("### How to use this app")
    st.markdown("""
    1. **Explore** the dataset
    2. **Run** the agent on a question
    3. **Review** scores & give feedback
    """)
    st.divider()
    st.caption("Built with Streamlit + Langfuse")

# ── Hero Section ─────────────────────────────────────────────────────────────

st.title("🧪 Agent Eval")
st.markdown(
    "Evaluate an AI agent end-to-end: **what it answers**, "
    "**how it finds evidence**, and **whether its reasoning makes sense**."
)

# ── 3-Layer Scoring ──────────────────────────────────────────────────────────

st.markdown(
    "We score **3 layers**: answer accuracy, trajectory quality, and reasoning — "
    "weighted into a single composite."
)

st.code("composite = 0.4 × answer + 0.35 × trajectory + 0.25 × reasoning", language=None)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Answer Baseline", "6.1%")
col2.metric("Trajectory Baseline", "55.0%")
col3.metric("Reasoning Baseline", "63.3%")
col4.metric("Composite Baseline", "37.5%")

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
        | `"Barack Obama"` vs `"Barack Obama"` | Yes |
        | `"Barack Obama"` vs `"Obama"` | No |
        | `"The United States"` vs `"united states"` | Yes |

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

    st.code("trajectory_score = 0.4 × retrieval_f1 + 0.4 × action_order + 0.2 × efficiency", language=None)

with tab3:
    st.markdown("### Does the agent's reasoning actually make sense?")
    st.info("**LLM-as-judge** — a separate Claude call scores each dimension using a rubric")

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
