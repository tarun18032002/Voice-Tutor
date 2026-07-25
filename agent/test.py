"""Manual LangGraph smoke test.

This file is not imported by production. Run it from the project root:

    python -m agent.test "What is carbon dioxide?"
"""

from __future__ import annotations

import asyncio
import sys

from langchain.messages import AIMessage, HumanMessage

from agent.config import (
    configure_logging,
    get_settings,
    validate_environment,
)
from agent.models import TutorRuntimeContext


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


async def run_smoke_test(question: str) -> None:
    validate_environment(include_livekit=False)

    # Import after validation so missing configuration fails clearly.
    from agent.graph import get_tutor_graph

    settings = get_settings()
    graph = get_tutor_graph()
    context = TutorRuntimeContext(
        room_id="local-smoke-test",
        cache_ttl_seconds=settings.session_cache_ttl_seconds,
        max_cache_entries=settings.session_cache_max_entries,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": settings.graph_recursion_limit},
        context=context,
    )

    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            print(_message_text(message))
            return

    raise RuntimeError("The graph completed without an assistant response.")


def main() -> None:
    configure_logging()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "What is carbon dioxide? Explain it simply."

    asyncio.run(run_smoke_test(question))


if __name__ == "__main__":
    main()
