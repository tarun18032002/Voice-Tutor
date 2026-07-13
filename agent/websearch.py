"""
websearch.py
------------
Web Search node powered by Tavily.

Runs only when the Orchestrator chooses SEARCH. It uses optimized search
queries from the orchestrator when available; otherwise it falls back to the
student query.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langsmith import traceable
from tavily import TavilyClient

from agent.models import TutorAgentState

logger = logging.getLogger(__name__)

MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))


@dataclass
class WebSearchTool:
    api_key: str = field(default_factory=lambda: os.environ["TAVILY_API_KEY"])
    max_results: int = MAX_RESULTS

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        self._client = TavilyClient(api_key=self.api_key)

    @traceable(name="Tavily Web Search", run_type="tool")
    def search(self, query: str) -> str:
        logger.info("WebSearchTool: searching for '%s'", query[:80])
        try:
            raw: dict[str, Any] = self._client.search(
                query=query,
                max_results=self.max_results,
                include_answer=True,
            )
        except Exception as exc:
            logger.exception("Tavily search failed.")
            raise RuntimeError("Web search operation failed.") from exc

        parts: list[str] = []

        if raw.get("answer"):
            parts.append(f"Summary: {raw['answer']}")

        for result in raw.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            if content:
                parts.append(f"[{title}]({url})\n{content}")

        return "\n\n---\n\n".join(parts)


class WebSearchNode:
    """LangGraph-compatible web search node."""

    def __init__(self) -> None:
        self._tool = WebSearchTool()

    def __call__(self, state: TutorAgentState | dict[str, Any]) -> dict[str, str]:
        if isinstance(state, dict):
            queries = state.get("search_queries") or [state.get("query", "")]
        else:
            queries = state.search_queries or [state.query]

        queries = [q for q in queries if q]
        if not queries:
            raise ValueError("WebSearchNode requires at least one query.")

        results = [self._tool.search(query) for query in queries[:3]]
        web_info = "\n\n=== NEXT SEARCH RESULT SET ===\n\n".join(r for r in results if r)

        return {
            "WebSearch_info": web_info,
            "voice_status": "I found some information from the web." if web_info else "I could not find any relevant information online."
        }


def websearch_node(state: TutorAgentState | dict[str, Any]) -> dict[str, str]:
    return WebSearchNode()(state)
