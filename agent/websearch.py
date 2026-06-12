"""
websearch.py
------------
Web Search node powered by Tavily.

Performs a targeted search for the student query and stores clean,
concatenated result snippets in state.WebInfo.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from tavily import TavilyClient

from .models import TutorAgentState

logger = logging.getLogger(__name__)

MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))


@dataclass
class WebSearchTool:
    """
    Thin, typed wrapper around the Tavily Search API.

    Parameters
    ----------
    api_key:
        Tavily API key.  Falls back to the TAVILY_API_KEY env var.
    max_results:
        Number of search results to fetch per query.
    """

    api_key: str = field(default_factory=lambda: os.environ["TAVILY_API_KEY"])
    max_results: int = MAX_RESULTS

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        self._client = TavilyClient(api_key=self.api_key)

    def search(self, query: str) -> str:
        """
        Execute a web search and return a clean text block.

        Returns
        -------
        Concatenated snippet strings separated by ``---``.
        Raises RuntimeError on Tavily failure.
        """
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

        # Tavily may return a top-level summarised answer
        if raw.get("answer"):
            parts.append(f"Summary: {raw['answer']}")

        for result in raw.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            if content:
                parts.append(f"[{title}]({url})\n{content}")

        combined = "\n\n---\n\n".join(parts)
        logger.debug("WebSearchTool: assembled %d chars of web content.", len(combined))
        return combined


class WebSearchNode:
    """LangGraph-compatible web search node."""

    def __init__(self) -> None:
        self._tool = WebSearchTool()

    def __call__(self, state: TutorAgentState) -> dict:
        web_info = self._tool.search(state.StudentQuery)
        return {"WebInfo": web_info}


# ── Module-level function alias ───────────────────────────────────────────────

def websearch_node(state: TutorAgentState) -> dict:
    return WebSearchNode()(state)