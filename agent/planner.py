"""
planner.py
----------
Planner node for the multi-agent Tutor Agent.

Role
----
The planner does NOT answer the student's question and does NOT create facts.
It creates a short pedagogical strategy that tells the Tutor Agent how to teach
through voice: tone, structure, explanation style, and follow-up style.

Expected state fields
---------------------
Input:
    state.query

Output:
    state.Planner_info
    state.planner_strategy
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agent.llm import LLMClient
from agent.models import PlannerConfig, TutorAgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Pedagogical Planner for a voice-based AI tutor for children.

Your job is to decide HOW the Tutor Agent should teach the student's query.
You must not answer the query and you must not invent factual content.

Create a short teaching strategy focused on:
1. Tone and persona.
2. Voice-friendly delivery style.
3. Teaching structure.
4. Best teaching mode: STORY, ANALOGY, HINT, STEP_BY_STEP, EXAMPLE, or SOCRATIC.
5. A short comprehension check question the Tutor Agent can ask at the end.

Rules:
- Do NOT answer the student's question.
- Do NOT include factual explanations of the topic.
- Keep the plan useful for real-time voice tutoring.
- Prefer simple words, short sentences, and child-friendly pacing.
- Output maximum 4 short bullets.
"""


def _get_state_value(state: TutorAgentState | dict[str, Any], key: str, default: Any = "") -> Any:
    """Support both Pydantic state objects and plain dict state."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _build_user_message(state: TutorAgentState | dict[str, Any]) -> str:
    query = _get_state_value(state, "query", "")
    if not query:
        raise ValueError("PlannerNode requires state.query to be set.")

    return f"Student query:\n{query}\n\nCreate the teaching strategy only."


class PlannerNode:
    """LangGraph-compatible planner node."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self._model = model or os.getenv("PLANNER_MODEL", "gpt-5-mini")

    async def __call__(self, state: TutorAgentState | dict[str, Any]) -> dict[str, str]:
        client = LLMClient(api_key=self._api_key)
        response: PlannerConfig = await client.agenerate_response(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            text_format=PlannerConfig,
        )

        plan = response.plan.strip()
        if not plan:
            raise ValueError("PlannerNode returned an empty plan.")

        logger.info("PlannerNode: strategy generated (%d chars).", len(plan))

        # Match the current TutorAgentState field names.
        # Planner_info is a readable copy; planner_strategy is what the orchestrator/tutor should use.
        return {
            "Planner_info": plan,
            "planner_strategy": plan,
        }


# Module-level function used by graph.py

async def planner_node(state: TutorAgentState | dict[str, Any]) -> dict[str, str]:
    return await PlannerNode()(state)
