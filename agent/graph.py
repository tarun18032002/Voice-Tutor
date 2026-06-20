
"""
graph.py
--------
LangGraph graph assembly for the Tutor Agent.

Flow
----
User Query
    │
[orchestrator]
    │
┌────┴────┐
│         │
retriever web
│         │
└────┬────┘
     │
[knowledge_agent]
     │
[planner]
     │
[tutor]
     │
    END
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from agent.input_mapper import input_mapper_node
from agent.knowledge_agent import knowledge_agent_node
from agent.models import TutorAgentState, TutorInputState
from agent.orchestrator import orchestrator_node
from agent.planner import planner_node
from agent.retriver import retriever_node
from agent.tutor import tutor_node
from agent.websearch import websearch_node

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Edge condition functions
# ──────────────────────────────────────────────────────────────────────────────

def route_after_orchestrator(
    state: TutorAgentState,
) -> Literal["retriever", "web", "knowledge_agent"]:
    """
    Route based on the orchestrator's AgentAction.
    """
    action = state.AgentAction

    if action == "retriever":
        return "retriever"

    if action == "web":
        return "web"

    logger.debug("Orchestrator chose 'answer' – skipping retrieval.")
    return "knowledge_agent"


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

def build_tutor_graph() -> StateGraph:
    """
    Assemble and compile the Tutor Agent LangGraph.
    """

    builder = StateGraph(
        TutorAgentState,
        # input_schema=TutorInputState,
    )

    # Register nodes
    builder.add_node("input_mapper", input_mapper_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("web", websearch_node)
    builder.add_node("knowledge_agent", knowledge_agent_node)
    builder.add_node("planner", planner_node)
    builder.add_node("tutor", tutor_node)

    # Start
    builder.add_edge(START, "input_mapper")
    builder.add_edge("input_mapper", "orchestrator")

    # Orchestrator routing
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "retriever": "retriever",
            "web": "web",
            "knowledge_agent": "knowledge_agent",
        },
    )

    # Retriever/Web → Knowledge Agent
    builder.add_edge("retriever", "knowledge_agent")
    builder.add_edge("web", "knowledge_agent")

    # Knowledge Agent → Planner → Tutor → END
    builder.add_edge("knowledge_agent", "planner")
    builder.add_edge("planner", "tutor")
    builder.add_edge("tutor", END)

    return builder.compile()




