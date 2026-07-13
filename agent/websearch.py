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
import json
import asyncio
from langchain.tools import tool 

from langsmith import traceable
from tavily import TavilyClient

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


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------




@tool
async def web_search(query: str) -> str:
    """
    Search the public web.

    Use this tool only when the student's question requires current,
    recent, changing, or externally verified information.

    Do not use web search for stable general-knowledge questions.
    """

    try:
        # If search() is synchronous, keep it off the event loop.
        results = await asyncio.to_thread(
            WebSearchTool().search,
            query,
        )

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "web_search_results": results,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("Web search failed")

        return json.dumps(
            {
                "status": "error",
                "query": query,
                "message": str(exc),
            }
        )
