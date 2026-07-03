"""
tutor.py
--------
Tutor node.

Generates the final child-friendly answer using:
- student query
- pedagogical planner strategy
- internal retrieved knowledge
- optional web search knowledge
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import AIMessage

from agent.llm import LLMClient
from agent.models import TutorAgentState, TutorConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are a friendly voice-based AI tutor for children.

Your job:
Explain the student's topic clearly, simply, and accurately.

Rules:
- Follow the planner strategy.
- Use only the available knowledge and safe general knowledge.
- Keep the answer concise and voice-friendly.
- Use short sentences.
- Avoid heavy jargon.
- End with one simple comprehension-check question.
- If the available knowledge is not enough, say what is missing instead of guessing.
"""


def _get_state_value(state: TutorAgentState | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _build_user_message(state: TutorAgentState | dict[str, Any]) -> str:
    query = _get_state_value(state, "query", "")
    planner_strategy = _get_state_value(state, "planner_strategy", "")
    retrieved_info = _get_state_value(state, "Retriver_info", [])
    web_info = _get_state_value(state, "WebSearch_info", "")

    if isinstance(retrieved_info, list):
        retrieved_text = "\n\n---\n\n".join(retrieved_info)
    else:
        retrieved_text = str(retrieved_info or "")

    return f"""Student query:
{query}

Planner strategy:
{planner_strategy if planner_strategy else "No planner strategy provided."}

Internal retrieved knowledge:
{retrieved_text if retrieved_text else "No internal retrieved knowledge available."}

Web search knowledge:
{web_info if web_info else "No web search knowledge available."}

Now write the final tutor answer.
"""


class TutorNode:
    """LangGraph-compatible tutor answer node."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self._model = model or os.getenv("TUTOR_MODEL", "gpt-5-mini")

    def __call__(self, state: TutorAgentState | dict[str, Any]) -> dict[str, Any]:
        client = LLMClient(api_key=self._api_key)
        response: TutorConfig = client.generate_response(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(state)},
            ],
            text_format=TutorConfig,
        )

        answer = response.answer.strip()
        if not answer:
            raise ValueError("TutorNode returned an empty answer.")

        logger.info("TutorNode: answer generated (%d chars).", len(answer))

        return {
            "TutorAnswer": answer,
            "messages": [AIMessage(content=answer)],
        }


def tutor_node(state: TutorAgentState | dict[str, Any]) -> dict[str, Any]:
    return TutorNode()(state)
