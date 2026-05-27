"""
Orchestrator — Claude API agentic loop.

Runs a tool-use loop for a given agent (technical, fundamental, or sentiment).
The loop continues until Claude returns a stop_reason of "end_turn" (no more
tool calls), indicating the agent has completed its analysis and decision.
"""
import json
import logging

import anthropic

import config
from agents import AGENTS
from tools import ALL_TOOL_SPECS, TOOL_DISPATCH

logger = logging.getLogger("app.orchestrator")


def run_agent(agent_name: str = "technical", max_turns: int = 20) -> str:
    """
    Run one full agent cycle for the given model type.

    Parameters
    ----------
    agent_name : "technical" | "fundamental" | "sentiment"
    max_turns  : Safety cap on tool-use iterations

    Returns
    -------
    str — the agent's final text response (summary of actions taken)
    """
    agent = AGENTS[agent_name]

    if not agent.ALLOWED_TOOLS:
        logger.info(f"{agent_name} agent not yet implemented — skipping.")
        return f"{agent_name} agent not yet implemented."

    client  = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    tools   = agent.filter_tools(ALL_TOOL_SPECS)
    messages = []

    # Initial user turn — trigger the agent's decision loop
    messages.append({
        "role": "user",
        "content": (
            f"Run your analysis and decision cycle for the {agent_name} model "
            f"on {config.TICKER}. Follow your system prompt instructions."
        ),
    })

    logger.info(f"Starting {agent_name} agent cycle (max_turns={max_turns})")
    final_text = ""

    for turn in range(max_turns):
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            system=agent.SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        logger.debug(f"Turn {turn+1}: stop_reason={response.stop_reason}")

        # Collect all text content blocks
        text_blocks = [b.text for b in response.content if b.type == "text"]
        if text_blocks:
            final_text = "\n".join(text_blocks)

        # If no tool calls, the agent is done
        if response.stop_reason == "end_turn":
            logger.info(f"{agent_name} agent finished after {turn+1} turn(s).")
            break

        # Process tool calls
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input

            logger.info(f"  Tool call: {tool_name}({json.dumps(tool_input, default=str)[:120]})")

            fn = TOOL_DISPATCH.get(tool_name)
            if fn is None:
                result_content = f"Error: unknown tool '{tool_name}'"
                logger.error(f"  Unknown tool: {tool_name}")
            else:
                try:
                    result = fn(**tool_input)
                    result_content = json.dumps(result, default=str)
                    logger.info(f"  Result: {result_content[:200]}")
                except Exception as exc:
                    result_content = f"Error executing {tool_name}: {exc}"
                    logger.error(f"  Tool error: {exc}", exc_info=True)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_content,
            })

        # Append tool results as user turn
        messages.append({"role": "user", "content": tool_results})

    else:
        logger.warning(f"{agent_name} agent hit max_turns={max_turns} without finishing.")

    return final_text
