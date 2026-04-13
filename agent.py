"""
Claude-based QA agent with tool use for multi-hop question answering.

This is the agent being evaluated. It starts deliberately naive (minimal system
prompt, simple tool loop) so that Phase 2's autoresearch loop has room to improve it.

The agent has two tools:
- search_paragraphs(query): keyword search over the provided context paragraphs
- read_paragraph(title): read all sentences of a specific paragraph by title

The agent loop: send question → if Claude uses tools, execute them and loop back →
when Claude responds with text (no tools), return that as the answer.

Phase 2 can edit this file and system_prompt.md to improve scores.
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = Path("system_prompt.md").read_text().strip()

# Tool definitions for Claude
TOOLS = [
    {
        "name": "search_paragraphs",
        "description": "Search the available context paragraphs by keyword. Returns paragraphs whose title or sentences contain the query terms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant paragraphs",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_paragraph",
        "description": "Read all sentences of a specific paragraph by its title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The exact title of the paragraph to read",
                }
            },
            "required": ["title"],
        },
    },
]


def execute_search_paragraphs(query: str, context: list) -> list[dict]:
    """Search context paragraphs by keyword matching.

    Splits the query into individual terms and returns any paragraph whose
    title + sentences contain at least one query term (case-insensitive).
    Returns a list of {"title": str, "sentences": list[str]} dicts.
    """
    query_lower = query.lower()
    query_terms = query_lower.split()
    results = []

    for title, sentences in context:
        text = (title + " " + " ".join(sentences)).lower()
        if any(term in text for term in query_terms):
            results.append({
                "title": title,
                "sentences": sentences,
            })

    return results


def execute_read_paragraph(title: str, context: list) -> str | None:
    """Return all sentences joined as a single string for the given paragraph title.

    Returns None if no paragraph with that exact title exists in context.
    """
    for ctx_title, sentences in context:
        if ctx_title == title:
            return " ".join(sentences)
    return None


def execute_tool(tool_name: str, tool_input: dict, context: list) -> str:
    """Dispatch a tool call by name and return the result as a string.

    This is the bridge between Claude's tool_use blocks and the local
    tool implementations. Returns a human-readable error for unknown tools.
    """
    if tool_name == "search_paragraphs":
        results = execute_search_paragraphs(tool_input["query"], context)
        if not results:
            return "No paragraphs found matching your query."
        return json.dumps(results, indent=2)
    elif tool_name == "read_paragraph":
        result = execute_read_paragraph(tool_input["title"], context)
        if result is None:
            return f"No paragraph found with title '{tool_input['title']}'."
        return result
    else:
        return f"Unknown tool: {tool_name}"


def run_agent(question: str, context: list) -> dict:
    """
    Run the QA agent on a question with given context paragraphs.

    Args:
        question: The question to answer
        context: List of [title, [sentences...]] paragraphs to search

    Returns:
        {
            "answer": str,
            "trajectory": list,  # List of tool calls with inputs and outputs
            "token_usage": int,  # Total tokens (input + output) for backward compat
            "token_usage_details": {  # Granular breakdown for accurate cost calculation
                "input_tokens": int,
                "output_tokens": int,
                "cache_read_input_tokens": int,
                "cache_creation_input_tokens": int,
            },
        }
    """
    messages = [{"role": "user", "content": question}]
    trajectory = []
    total_tokens = 0
    usage_details = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    max_iterations = 10

    for _ in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Accumulate granular token counts from the API response
        usage = response.usage
        usage_details["input_tokens"] += usage.input_tokens
        usage_details["output_tokens"] += usage.output_tokens
        usage_details["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        usage_details["cache_creation_input_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        total_tokens += usage.input_tokens + usage.output_tokens

        # Check if the model wants to use tools
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No tool use — extract final answer from text
            text_blocks = [b for b in response.content if b.type == "text"]
            answer = " ".join(b.text for b in text_blocks).strip()
            return {
                "answer": answer,
                "trajectory": trajectory,
                "token_usage": total_tokens,
                "token_usage_details": usage_details,
            }

        # Process tool calls
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_input = tool_block.input
            tool_output = execute_tool(tool_name, tool_input, context)

            trajectory.append({
                "tool": tool_name,
                "input": tool_input,
                "output": tool_output,
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": tool_output,
            })

        # Add assistant message and tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Hit max iterations — return whatever we have
    return {
        "answer": "Unable to determine the answer.",
        "trajectory": trajectory,
        "token_usage": total_tokens,
        "token_usage_details": usage_details,
    }
