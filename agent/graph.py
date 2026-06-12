"""
graph.py
--------
LangGraph graph assembly for the Tutor Agent.

Full flow
---------
                     User Query
                         │
                   [orchestrator]
                         │
          ┌──────────────┼──────────────┐
          │              │              │
      retriever         web          (answer –
          │              │           skip to KA)
          └──────────────┘
                   [knowledge_agent]
                         │
                     [planner]
                         │
                      [tutor]
                         │
               [comprehension_checker]
                         │
              ┌──────────┴──────────┐
              │                     │
           pass=False            pass=True
              │                     │
          [planner] ◄──────────    END
          (re-plan)
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .comprehension_checker import comprehension_checker_node
from .knowledge_agent import knowledge_agent_node
from .models import TutorAgentState, TutorInputState
from .models import TutorAgentState
from .planner import planner_node
from .orchestrator import orchestrator_node
from .planner import planner_node
from .retriver import retriever_node
from .tutor import tutor_node
from .websearch import websearch_node

logger = logging.getLogger(__name__)


# ── Edge condition functions ───────────────────────────────────────────────────

def route_after_orchestrator(
    state: TutorAgentState,
) -> Literal["retriever", "web", "knowledge_agent"]:
    """
    Route based on the orchestrator's AgentAction.
    'answer' means skip retrieval and go straight to the knowledge agent
    (using whatever info is already in state).
    """
    action = state.AgentAction
    if action == "retriever":
        return "retriever"
    if action == "web":
        return "web"
    # 'answer' or any unexpected value → straight to knowledge agent
    logger.debug("Orchestrator chose 'answer' – skipping retrieval.")
    return "knowledge_agent"


def route_after_comprehension(
    state: TutorAgentState,
) -> Literal["planner", END]:  # type: ignore[valid-type]
    """
    If comprehension passed (or max retries hit) → END.
    Otherwise → back to planner for a re-plan.
    """
    if state.ComprehensionPassed:
        logger.info("Comprehension passed – ending session.")
        return END
    logger.info(
        "Comprehension failed (retry %d) – re-planning.", state.RetryCount
    )
    return "planner"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_tutor_graph() -> StateGraph:
    """
    Assemble and compile the Tutor Agent LangGraph.

    Returns
    -------
    A compiled LangGraph ``StateGraph`` ready to be invoked.
    """
    builder = StateGraph(TutorAgentState, input_schema=TutorInputState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("orchestrator",          orchestrator_node)
    builder.add_node("retriever",             retriever_node)
    builder.add_node("web",                   websearch_node)
    builder.add_node("knowledge_agent",       knowledge_agent_node)
    builder.add_node("planner",               planner_node)
    builder.add_node("tutor",                 tutor_node)
    builder.add_node("comprehension_checker", comprehension_checker_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator")

    # ── Orchestrator → branch ─────────────────────────────────────────────────
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "retriever":      "retriever",
            "web":            "web",
            "knowledge_agent":"knowledge_agent",
        },
    )

    # ── Retriever / web → knowledge agent ────────────────────────────────────
    builder.add_edge("retriever", "knowledge_agent")
    builder.add_edge("web",       "knowledge_agent")

    # ── Knowledge agent → planner → tutor → comprehension checker ────────────
    builder.add_edge("knowledge_agent",       "planner")
    builder.add_edge("planner",               "tutor")
    builder.add_edge("tutor",                 "comprehension_checker")

    # ── Comprehension checker → re-plan loop or END ───────────────────────────
    builder.add_conditional_edges(
        "comprehension_checker",
        route_after_comprehension,
        {
            "planner": "planner",
            END:       END,
        },
    )

    return builder.compile()


# ── Singleton accessor (lazy) ──────────────────────────────────────────────────

_graph = None


def get_tutor_graph() -> StateGraph:
    """Return the compiled graph, building it once."""
    global _graph
    if _graph is None:
        _graph = build_tutor_graph()
        logger.info("Tutor graph compiled.")
    return _graph