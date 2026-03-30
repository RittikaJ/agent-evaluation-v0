# Search Strategy Rubric

Score whether the agent's search queries were well-chosen for the question.

| Score | Criteria |
|-------|----------|
| 5 | Targeted, efficient queries that decompose the multi-hop question. First query finds entity A, second query uses information from A to find entity B. No redundant searches. |
| 4 | Good queries with minor inefficiency. Mostly targeted but one query could have been more specific. |
| 3 | Reasonable but unfocused queries. The agent searched for relevant topics but didn't decompose the question into precise sub-queries. |
| 2 | Weak queries. The agent copied the whole question as a search, or searched for loosely related terms. |
| 1 | Irrelevant queries. The agent searched for unrelated terms, or didn't search at all. |

## Hard Guardrails

- If the agent's first query is a verbatim copy of the full question: max score = 3
- If the agent made no searches: score = 1
