# Reasoning Coherence Rubric

Score whether the agent's reasoning chain makes logical sense for a multi-hop question.

| Score | Criteria |
|-------|----------|
| 5 | Clear multi-hop chain: found fact A, explicitly used A to find fact B, combined A+B to answer. Each step logically follows from the previous. |
| 4 | Mostly coherent chain with minor gaps. The multi-hop structure is visible but one connection is implicit rather than explicit. |
| 3 | Some reasoning visible but significant jumps or gaps. The agent found relevant information but didn't clearly connect the hops. |
| 2 | Weak reasoning. The agent retrieved some information but the connection between retrieval steps and the final answer is unclear. |
| 1 | No visible reasoning. The agent appears to guess, or its reasoning contradicts its own evidence. |

## Hard Guardrails

- If the agent made only one search for a multi-hop question: max score = 3
- If the agent's reasoning contradicts its own tool outputs: score = 1
