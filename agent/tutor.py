"""Tutor model node for the LangGraph workflow."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.llm import model
from agent.models import TutorRuntimeContext, TutorState
from agent.prompts import TUTOR_SYSTEM_PROMPT
from agent.tools import TOOLS

logger = logging.getLogger(__name__)
settings = get_settings()

model_with_tools = (
    model.bind_tools(
        TOOLS,
        parallel_tool_calls=False,
    )
    if TOOLS
    else model
)

_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:why|how|that|this|it|those|these|again|simpler|simple|"
    r"example|another|more|continue|first|second|third|previous|"
    r"mean|means|part|point|step|explain|repeat|understand)\b",
    re.I,
)


def _has_output(response: Any) -> bool:
    if getattr(response, "tool_calls", None):
        return True

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return bool(content)

    return content is not None


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
        elif getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
        else:
            continue

        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    return ""


def _looks_like_follow_up(text: str) -> bool:
    """Use a conservative heuristic to avoid injecting stale tool data."""

    if not text:
        return False

    words = text.split()
    if len(words) > 24:
        return False

    return bool(_FOLLOW_UP_PATTERN.search(text))


def _build_model_messages(
    state: TutorState,
    runtime: Runtime[TutorRuntimeContext],
) -> list[Any]:
    messages: list[Any] = [SystemMessage(content=TUTOR_SYSTEM_PROMPT)]
    user_text = _latest_user_text(state["messages"])

    if _looks_like_follow_up(user_text):
        reference_context = runtime.context.recent_reference_context(
            settings.session_reference_context_chars
        )

        if reference_context:
            messages.append(
                SystemMessage(
                    content=(
                        "The following information was already fetched earlier "
                        "in this same room. It may be unrelated to the new "
                        "message. Use it only when the new message is clearly a "
                        "follow-up to that topic. When it is sufficient, answer "
                        "from it and do not call a tool again.\n\n"
                        + reference_context
                    )
                )
            )

    messages.extend(state["messages"])
    return messages


async def tutor_node(
    state: TutorState,
    runtime: Runtime[TutorRuntimeContext],
) -> dict[str, Any]:
    """Invoke the only LLM used by the graph."""

    started = time.perf_counter()
    llm_calls = state.get("llm_calls", 0) + 1

    try:
        response = await asyncio.wait_for(
            model_with_tools.ainvoke(
                _build_model_messages(state, runtime)
            ),
            timeout=settings.llm_total_timeout_seconds,
        )

        if not _has_output(response):
            logger.warning(
                "Tutor model returned an empty response room=%s",
                runtime.context.room_id,
            )
            response = AIMessage(
                content=(
                    "I could not finish that explanation. "
                    "Please ask me the question one more time."
                )
            )

    except asyncio.TimeoutError:
        logger.warning(
            "Tutor model timed out room=%s",
            runtime.context.room_id,
        )
        response = AIMessage(
            content=(
                "That explanation took too long to prepare. "
                "Please ask me the question one more time."
            )
        )
    except asyncio.CancelledError:
        logger.info(
            "Tutor model call cancelled room=%s",
            runtime.context.room_id,
        )
        raise
    except Exception:
        logger.exception(
            "Tutor model invocation failed room=%s",
            runtime.context.room_id,
        )
        response = AIMessage(
            content=(
                "I had trouble preparing that explanation. "
                "Please ask me the question one more time."
            )
        )

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "Tutor model completed room=%s llm_call=%s duration_ms=%s",
        runtime.context.room_id,
        llm_calls,
        duration_ms,
    )

    return {
        "messages": [response],
        "llm_calls": llm_calls,
    }
