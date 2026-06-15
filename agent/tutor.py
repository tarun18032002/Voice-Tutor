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

from agent.llm import LLMClient
from agent.models import TutorAgentState, TutorConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
If TeachingMode is STORY:
Explain through a short story.

If TeachingMode is ANALOGY:
Use a real-world analogy.

If TeachingMode is HINT:
Provide hints instead of directly revealing everything.

If TeachingMode is STEP_BY_STEP:
Explain in numbered steps.

If TeachingMode is EXAMPLE:
Teach through examples.

If TeachingMode is SOCRATIC:
Teach by asking guiding questions.
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
        response: TutorConfig = client.generate_response(
            input=[
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