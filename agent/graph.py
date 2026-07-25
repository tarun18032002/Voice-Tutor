"""Single-LLM tutor graph used by the LiveKit adapter.

Flow:
    START -> tutor
      - direct answer -> END
      - tool call -> tools -> tutor
      - loop limit -> tool_limit -> END
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import get_settings
from agent.models import TutorRuntimeContext, TutorState
from agent.tools import TOOLS
from agent.tutor import tutor_node

logger = logging.getLogger(__name__)
settings = get_settings()


def route_after_tutor(state: TutorState) -> Any:
    """Route to a tool, a graceful limit response, or graph completion."""

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return END

    if state.get("llm_calls", 0) >= settings.max_llm_calls_per_turn:
        return "tool_limit"

    return "tools"


async def tool_limit_node(state: TutorState) -> dict[str, Any]:
    """Resolve pending calls and stop a runaway model-tool loop safely."""

    last_message = state["messages"][-1]
    pending_calls = getattr(last_message, "tool_calls", None) or []

    messages: list[Any] = []
    for call in pending_calls:
        tool_call_id = call.get("id") or "tool-call-limit"
        messages.append(
            ToolMessage(
                content="Tool-call limit reached for this student turn.",
                tool_call_id=tool_call_id,
            )
        )

    messages.append(
        AIMessage(
            content=(
                "I could not finish that lookup right now. "
                "Please ask the question in a simpler way."
            )
        )
    )

    logger.warning(
        "Tool-call loop stopped llm_calls=%s pending_calls=%s",
        state.get("llm_calls", 0),
        len(pending_calls),
    )

    return {"messages": messages}


def build_tutor_graph():
    """Build and compile the reusable tutor graph."""

    builder = StateGraph(
        TutorState,
        context_schema=TutorRuntimeContext,
    )
    builder.add_node("tutor", tutor_node)
    builder.add_node("tool_limit", tool_limit_node)
    builder.add_edge(START, "tutor")

    if TOOLS:
        builder.add_node("tools", ToolNode(TOOLS))
        builder.add_conditional_edges(
            "tutor",
            route_after_tutor,
            {
                "tools": "tools",
                "tool_limit": "tool_limit",
                END: END,
            },
        )
        builder.add_edge("tools", "tutor")
    else:
        builder.add_edge("tutor", END)

    builder.add_edge("tool_limit", END)
    return builder.compile()


_tutor_graph = build_tutor_graph()


def get_tutor_graph():
    """Return the process-wide compiled graph."""

    return _tutor_graph
