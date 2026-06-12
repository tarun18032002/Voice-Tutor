"""
tutor.py
--------
Tutor node.

Follows the planner's step-by-step plan to generate a student-facing
answer.  The answer is stored in state.TutorAnswer and later evaluated
by the Comprehension Checker.
"""

from __future__ import annotations

import logging
import os

from .llm import LLMClient
from .models import TutorAgentState, TutorConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a friendly, encouraging AI tutor.

Your job is to answer the student's query by following the provided plan
EXACTLY, step by step.

Rules
-----
1. Follow every step in the plan.
2. Write in clear, simple language appropriate for a student.
3. Use examples, analogies, or bullet points where helpful.
4. If the plan cannot be fully satisfied by the available information,
   say so explicitly rather than guessing.
5. Do NOT include the plan text itself in your answer.
"""


def _build_user_message(state: TutorAgentState) -> str:
    parts = [f"StudentQuery: {state.StudentQuery}"]
    if state.PlanInfo:
        parts.append(f"Plan:\n{state.PlanInfo}")
    if state.SynthesisInfo:
        parts.append(f"AvailableKnowledge:\n{state.SynthesisInfo}")
    return "\n\n".join(parts)


class TutorNode:
    """LangGraph-compatible tutor answer node."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    def __call__(self, state: TutorAgentState) -> dict:
        client = LLMClient(api_key=self._api_key)
        response: TutorConfig = client.parse(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            response_format=TutorConfig,
            reasoning_effort="medium",
        )

        logger.info("TutorNode: answer generated (%d chars).", len(response.answer))
        if not response.answer:
            raise ValueError("TutorNode returned an empty answer.")

        return {"TutorAnswer": response.answer}


# ── Module-level function alias ───────────────────────────────────────────────

def tutor_node(state: TutorAgentState) -> dict:
    return TutorNode()(state)