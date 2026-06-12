"""
knowledge_agent.py
------------------
Knowledge Agent node.

Takes whatever information has been collected (RAG + web) and produces
a single, coherent synthesised knowledge block that all downstream nodes
(planner, tutor) will use.
"""

from __future__ import annotations

import logging
import os

from .llm import LLMClient
from .models import KnowledgeAgentConfig, TutorAgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a knowledge synthesis agent inside an AI tutoring system.

Your job is to merge all available information sources into a single,
coherent, factually-accurate knowledge block that directly addresses
the student's query.

Rules
-----
1. Combine retrieved course material and web search results seamlessly.
2. Remove duplicates; resolve contradictions by preferring course material.
3. Preserve all key facts, definitions, and examples relevant to the query.
4. Output ONLY the synthesised knowledge – no meta-commentary.
5. If sources are empty or irrelevant, state clearly what is available.
"""


def _build_user_message(state: TutorAgentState) -> str:
    parts = [f"StudentQuery: {state.StudentQuery}"]
    if state.RetrievedInfo:
        parts.append(f"Course Material (RAG):\n{state.RetrievedInfo}")
    if state.WebInfo:
        parts.append(f"Web Search Results:\n{state.WebInfo}")
    if not state.RetrievedInfo and not state.WebInfo:
        parts.append("No information sources available.")
    return "\n\n".join(parts)


class KnowledgeAgentNode:
    """LangGraph-compatible knowledge synthesis node."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    def __call__(self, state: TutorAgentState) -> dict:
        client = LLMClient(api_key=self._api_key)
        response: KnowledgeAgentConfig = client.parse(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            response_format=KnowledgeAgentConfig,
            reasoning_effort="medium",
        )

        logger.info(
            "KnowledgeAgent: synthesised %d chars.", len(response.synthesis)
        )
        if not response.synthesis:
            raise ValueError("KnowledgeAgentNode returned empty synthesis.")

        return {"SynthesisInfo": response.synthesis}


# ── Module-level function alias ───────────────────────────────────────────────

def knowledge_agent_node(state: TutorAgentState) -> dict:
    return KnowledgeAgentNode()(state)