"""
planner.py
----------
Planner node.

Takes the synthesised knowledge and the student query and produces a
step-by-step plan that the Tutor node will follow to answer the student.

On re-plan cycles (comprehension failed) the previous answer and
comprehension feedback are included so the plan addresses the gaps.
"""

from __future__ import annotations

import logging
import os

from agent.llm import LLMClient
from agent.models import PlannerConfig, TutorAgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You decide HOW to teach.

Possible modes:
- story
- analogy
- example
- hint
- step_by_step

Return only 1-3 short bullet points.

Do NOT write the answer.
Maximum 50 words.

Your job is to produce a clear, structured answer-plan that the Tutor
agent will follow step-by-step.

Rules
-----
1. Base the plan ENTIRELY on the SynthesisInfo provided.
2. For simple queries, one step is sufficient.
3. For complex queries, list explicit numbered steps.
4. If a previous answer failed comprehension, address the specific
   feedback in your plan (re-explain, add examples, simplify, etc.).
5. Do NOT write the answer itself – only the plan.
"""


def _build_user_message(state: TutorAgentState) -> str:
    parts = [f"StudentQuery: {state.StudentQuery}"]
    if state.SynthesisInfo:
        parts.append(f"SynthesisInfo:\n{state.SynthesisInfo}")
    if state.TutorAnswer and state.ComprehensionFeedback:
        parts.append(f"PreviousAnswer:\n{state.TutorAnswer}")
        parts.append(f"ComprehensionFeedback:\n{state.ComprehensionFeedback}")
    return "\n\n".join(parts)


class PlannerNode:
    """LangGraph-compatible planner node."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    def __call__(self, state: TutorAgentState) -> dict:
        client = LLMClient(api_key=self._api_key)
        response: PlannerConfig = client.generate_response(
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            text_format=PlannerConfig,
        )

        logger.info("PlannerNode: plan generated (%d chars).", len(response.plan))
        if not response.plan:
            raise ValueError("PlannerNode returned an empty plan.")

        return {"PlanInfo": response.plan}


# ── Module-level function alias ───────────────────────────────────────────────

def planner_node(state: TutorAgentState) -> dict:
    return PlannerNode()(state)