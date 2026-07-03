"""
graph.py
--------
LangGraph graph assembly for the multi-agent Tutor Agent.

Architecture
------------
START
  ├── retriever
  └── planner
        ↓
  orchestrator
    ├── SEARCH → websearch → tutor
    └── ANSWER → tutor
        ↓
       END
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agent.models import InputTutorAgentState, TutorAgentState
from agent.orchestrator import orchestrator_node
from agent.planner import planner_node
from agent.retriver import retriever_node
from agent.tutor import tutor_node
from agent.websearch import websearch_node

logger = logging.getLogger(__name__)


import inspect
from langsmith import traceable


def traced_node(name: str, fn):
    if inspect.iscoroutinefunction(fn):
        @traceable(name=name, run_type="chain")
        async def async_wrapper(state):
            return await fn(state)

        return async_wrapper

    @traceable(name=name, run_type="chain")
    def sync_wrapper(state):
        return fn(state)

    return sync_wrapper


def route_after_orchestrator(state: TutorAgentState) -> str:
    action = state.orchestrator_action

    if action == "SEARCH":
        return "web"

    if action == "ANSWER":
        return "tutor"

    raise ValueError(
        f"Invalid orchestrator action. Expected 'SEARCH' or 'ANSWER', got {action!r}."
    )


def build_tutor_graph():
    """Build and compile the tutor graph."""

    builder = StateGraph(
        TutorAgentState,
        input_schema=InputTutorAgentState,
    )

    # Nodes
    builder.add_node("retriever", traced_node("Retriever Agent", retriever_node))
    builder.add_node("planner", traced_node("Planner Agent", planner_node))
    builder.add_node("orchestrator", traced_node("Orchestrator Agent", orchestrator_node))
    builder.add_node("web", traced_node("Web Search Agent", websearch_node))
    builder.add_node("tutor", traced_node("Tutor Agent", tutor_node))

    # Parallel fan-out: retrieval and pedagogical planning can happen together.
    builder.add_edge(START, "retriever")
    builder.add_edge(START, "planner")

    # Fan-in: orchestrator runs only after retriever and planner are complete.
    builder.add_edge("retriever", "orchestrator")

    # Conditional route: either search the web or answer directly.
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "web": "web",
            "tutor": "tutor",
        },
    )

    # Web results are additional knowledge for the tutor.
    builder.add_edge("web", "planner")

    builder.add_edge("planner", "tutor")

    # Final tutor response ends the graph.
    builder.add_edge("tutor", END)

    return builder.compile()


# Backward-compatible alias if other files still call build_planner_graph().
def build_planner_graph():
    return build_tutor_graph()


# Optional cached graph helper for app/voice integrations.
def get_tutor_graph():
    return build_tutor_graph()


