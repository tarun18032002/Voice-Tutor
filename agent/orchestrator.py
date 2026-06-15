"""
orchestrator.py
---------------
Orchestrator node – the brain/router of the Tutor Agent graph.

Decides the next route:
  - "retriever" → RAG retrieval needed
  - "web"       → web search needed
  - "answer"    → enough context, skip to knowledge agent
"""

from __future__ import annotations

import logging
import os

from agent.llm import LLMClient
from agent.models import OrchestratorConfig, TutorAgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the orchestrator for an AI tutoring system.

Your sole job is to choose the BEST next route given the student's query and
whatever information has already been gathered.

Routes
------
- "retriever" : Use this when RetrievedInfo is empty or clearly insufficient
                and the answer likely exists in the course material (RAG).
- "web"       : Use this ONLY when the query needs fresh, current, or
                internet-only information that the course material cannot provide.
- "answer"    : Use this when the existing RetrievedInfo or WebInfo is already
                sufficient to answer the student query.

Rules
-----
1. Prefer "answer" if existing info directly addresses the StudentQuery.
2. Prefer "retriever" when RetrievedInfo is absent or incomplete.
3. Prefer "web" only for current/real-world facts not in the course material.
4. Never repeat the same route twice in one session (avoid loops).
"""


def _build_user_message(state: TutorAgentState) -> str:
    parts = [f"StudentQuery: {state.StudentQuery}"]
    if state.RetrievedInfo:
        parts.append(f"RetrievedInfo:\n{state.RetrievedInfo}")
    if state.WebInfo:
        parts.append(f"WebInfo:\n{state.WebInfo}")
    if state.AgentAction:
        parts.append(f"PreviousRoute: {state.AgentAction} (do NOT repeat this route)")
    return "\n\n".join(parts)


class OrchestratorNode:
    """Stateless callable node for LangGraph."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    # LangGraph calls nodes as callables: node(state) -> dict
    def __call__(self, state: TutorAgentState) -> dict:
        client = LLMClient(api_key=self._api_key)
        response: OrchestratorConfig = client.parse(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            response_format=OrchestratorConfig,
        )

        logger.info("Orchestrator → route=%s | reason=%s", response.route, response.reason)

        if response.route not in {"retriever", "web", "answer"}:
            raise ValueError(f"Orchestrator returned invalid route: {response.route!r}")

        return {
            "AgentAction": response.route,
            "OrchestratorReason": response.reason,
        }


# ── Module-level function alias (for graph.py convenience) ───────────────────

def orchestrator_node(state: TutorAgentState) -> dict:
    return OrchestratorNode()(state)