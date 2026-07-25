"""Single source of truth for tools exposed to the tutor model."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.retriver import retrieve_knowledge
from agent.websearch import web_search

settings = get_settings()

TOOLS: list[Any] = []

if settings.enable_retrieval:
    TOOLS.append(retrieve_knowledge)

if settings.enable_web_search:
    TOOLS.append(web_search)


def get_tools() -> list[Any]:
    """Return a copy so callers cannot mutate the global tool list."""

    return list(TOOLS)
