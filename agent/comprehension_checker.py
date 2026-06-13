"""
comprehension_checker.py
------------------------
Comprehension Checker node.

Evaluates whether the Tutor's answer is clear, complete, and would be
understood by a student.  Sets state.ComprehensionPassed and provides
feedback that the Planner uses on a re-plan cycle.

The graph uses the return value to branch:
  - ComprehensionPassed=True  → END
  - ComprehensionPassed=False → back to Planner (up to MAX_RETRIES)
"""

from __future__ import annotations

import logging
import os

from .llm import LLMClient
from .models import ComprehensionConfig, TutorAgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.getenv("COMPREHENSION_MAX_RETRIES", "2"))

_SYSTEM_PROMPT = """\
You are a comprehension quality checker for an AI tutoring system.

Your job is to evaluate whether the Tutor's answer adequately addresses
the student's query.

Evaluation criteria
-------------------
1. Correctness  – Is the answer factually accurate relative to the query?
2. Completeness – Does it cover all aspects of the query?
3. Clarity      – Is it written in language a student can understand?
4. Examples     – For concept questions, does it include helpful examples?

Output
------
- passed: true  → answer is satisfactory; student would understand.
- passed: false → answer has gaps; provide specific, actionable feedback
                  describing EXACTLY what is wrong or missing.
"""


def _build_user_message(state: TutorAgentState) -> str:
    parts = [
        f"StudentQuery: {state.StudentQuery}",
        f"TutorAnswer:\n{state.TutorAnswer}",
    ]
    if state.SynthesisInfo:
        parts.append(f"AvailableKnowledge (ground truth):\n{state.SynthesisInfo}")
    return "\n\n".join(parts)


class ComprehensionCheckerNode:
    """LangGraph-compatible comprehension checker node."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    def __call__(self, state: TutorAgentState) -> dict:
        # Safety: if we've already retried too many times, pass through.
        if state.RetryCount >= MAX_RETRIES:
            logger.warning(
                "ComprehensionChecker: max retries (%d) reached – forcing pass.", MAX_RETRIES
            )
            return {
                "ComprehensionPassed": True,
                "ComprehensionFeedback": "Max retries reached; returning best available answer.",
            }

        client = LLMClient(api_key=self._api_key)
        response: ComprehensionConfig = client.generate_response(
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            text_format=ComprehensionConfig,
        )

        logger.info(
            "ComprehensionChecker: passed=%s | feedback=%s",
            response.passed,
            response.feedback[:120],
        )

        updates: dict = {
            "ComprehensionPassed": response.passed,
            "ComprehensionFeedback": response.feedback,
        }
        if not response.passed:
            updates["RetryCount"] = state.RetryCount + 1

        return updates


# ── Module-level function alias ───────────────────────────────────────────────

def comprehension_checker_node(state: TutorAgentState) -> dict:
    return ComprehensionCheckerNode()(state)