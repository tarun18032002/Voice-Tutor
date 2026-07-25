"""Tavily web search and the current-information LangChain tool."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from tavily import TavilyClient

from agent.config import get_settings
from agent.models import TutorRuntimeContext

logger = logging.getLogger(__name__)
settings = get_settings()

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_SECRET_PATTERN = re.compile(
    r"\b(?:sk|tvly)-[A-Za-z0-9_-]{12,}\b|"
    r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+",
    re.I,
)
_LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)\d{12,}(?!\d)")


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).strip()[:500]


def _contains_sensitive_data(query: str) -> bool:
    return bool(
        _EMAIL_PATTERN.search(query)
        or _SECRET_PATTERN.search(query)
        or _LONG_NUMBER_PATTERN.search(query)
    )


def _safe_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_chars]


def _domain_from_url(value: Any) -> str:
    if not value:
        return ""

    try:
        domain = urlparse(str(value)).netloc.lower()
    except Exception:
        return ""

    return domain.removeprefix("www.")[:200]


def _emit_status(message: str) -> None:
    if not settings.enable_tool_status_messages:
        return

    try:
        writer = get_stream_writer()
        writer(message)
    except Exception:
        logger.debug("Unable to stream web-search status", exc_info=True)


@dataclass(slots=True)
class WebSearchTool:
    """Reusable Tavily client with bounded output."""

    api_key: str = field(
        default_factory=lambda: settings.tavily_api_key,
        repr=False,
    )
    max_results: int = field(
        default_factory=lambda: settings.web_search_max_results
    )
    timeout_seconds: float = field(
        default_factory=lambda: settings.web_search_timeout_seconds
    )
    max_context_chars: int = field(
        default_factory=lambda: settings.web_search_max_context_chars
    )
    search_depth: str = field(
        default_factory=lambda: settings.web_search_depth
    )
    _client: TavilyClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not configured.")

        self._client = TavilyClient(api_key=self.api_key)
        logger.info("Tavily web-search client initialized")

    def search(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        raw: dict[str, Any] = self._client.search(
            query=query,
            search_depth=self.search_depth,
            max_results=self.max_results,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            timeout=self.timeout_seconds,
        )

        answer = _safe_text(raw.get("answer"), 1500)
        results: list[dict[str, Any]] = []
        used_chars = len(answer)

        for item in raw.get("results", []):
            if not isinstance(item, dict):
                continue

            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break

            content = _safe_text(item.get("content"), min(1500, remaining))
            if not content:
                continue

            used_chars += len(content)
            score_value = item.get("score")
            score = (
                round(float(score_value), 6)
                if isinstance(score_value, (int, float))
                else None
            )

            results.append(
                {
                    "title": _safe_text(item.get("title"), 200),
                    "domain": _domain_from_url(item.get("url")),
                    "published_date": _safe_text(
                        item.get("published_date"),
                        40,
                    ),
                    "content": content,
                    "score": score,
                }
            )

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Web search completed count=%s duration_ms=%s",
            len(results),
            duration_ms,
        )

        if not answer and not results:
            return {
                "status": "not_found",
                "message": "No useful current web results were found.",
                "query": query,
                "answer": "",
                "results": [],
            }

        return {
            "status": "success",
            "query": query,
            "answer": answer,
            "results": results,
        }


@lru_cache(maxsize=1)
def get_web_search_client() -> WebSearchTool:
    """Create one Tavily client per worker process."""

    return WebSearchTool()


@tool
async def web_search(
    query: str,
    runtime: ToolRuntime[TutorRuntimeContext],
) -> dict[str, Any]:
    """Search the public web for current or changing information.

    Use this only when the question depends on recent events, current facts,
    changing information, or external verification. Never include personal
    information, credentials, or secrets in the query.
    """

    clean_query = _normalize_query(query)
    if not clean_query:
        return {
            "status": "error",
            "message": "A non-empty web-search query is required.",
        }

    if _contains_sensitive_data(clean_query):
        logger.warning("Blocked a web query containing sensitive-looking data")
        return {
            "status": "blocked",
            "message": (
                "The query appears to contain personal or secret information. "
                "Answer without web search or ask for a privacy-safe question."
            ),
        }

    context = runtime.context
    cached = context.get_cached_result("web", clean_query)
    if cached is not None:
        cached["cached"] = True
        logger.info("Web-search cache hit room=%s", context.room_id)
        return cached

    _emit_status("Let me check the latest information.")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                get_web_search_client().search,
                clean_query,
            ),
            timeout=settings.web_search_timeout_seconds + 1.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Web search timed out room=%s", context.room_id)
        return {
            "status": "timeout",
            "message": "The current-information search timed out.",
        }
    except asyncio.CancelledError:
        logger.info("Web search cancelled room=%s", context.room_id)
        raise
    except Exception:
        logger.exception("Web search failed room=%s", context.room_id)
        return {
            "status": "error",
            "message": "Current web information is unavailable right now.",
        }

    context.set_cached_result("web", clean_query, result)
    return result
