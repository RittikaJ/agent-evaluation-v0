# Groundedness Rubric

Score whether the agent's answer is supported by the paragraphs it actually retrieved.

| Score | Criteria |
|-------|----------|
| 5 | Answer directly and fully follows from retrieved evidence. Every claim is traceable to a specific retrieved paragraph. |
| 4 | Answer is mostly supported by evidence, with minor inferences that are reasonable. |
| 3 | Answer is partially supported. Some claims are grounded, others are inferred or weakly connected. |
| 2 | Answer has weak grounding. Most claims are not directly supported by retrieved evidence. |
| 1 | Answer contradicts retrieved evidence, or has no basis in what the agent actually read. |

## Hard Guardrails

- If the agent retrieved no paragraphs but still answered: max score = 2
- If the answer directly contradicts a retrieved paragraph: score = 1
- If the agent answered "I don't know" or equivalent: score = 3 (honest when lacking evidence)
