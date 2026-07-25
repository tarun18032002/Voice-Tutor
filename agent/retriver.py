"""Qdrant retrieval and the private-knowledge LangChain tool.

The filename intentionally remains ``retriver.py`` to preserve compatibility
with the existing project imports. Rename it only as a separate cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from openai import OpenAI
from qdrant_client import QdrantClient

from agent.config import get_settings
from agent.models import TutorRuntimeContext

logger = logging.getLogger(__name__)
settings = get_settings()


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).strip()[:500]


def _safe_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_chars]


def _emit_status(message: str) -> None:
    if not settings.enable_tool_status_messages:
        return

    try:
        writer = get_stream_writer()
        writer(message)
    except Exception:
        logger.debug("Unable to stream retrieval status", exc_info=True)


class BaseRetriever(ABC):
    """Interface for private-knowledge retrieval backends."""

    @abstractmethod
    def retrieve(self, query: str) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(slots=True)
class QdrantRetriever(BaseRetriever):
    """Qdrant vector retriever using OpenAI embeddings."""

    url: str = field(default_factory=lambda: settings.qdrant_url)
    collection: str = field(
        default_factory=lambda: settings.qdrant_collection
    )
    api_key: str = field(
        default_factory=lambda: settings.qdrant_api_key,
        repr=False,
    )
    text_field: str = field(
        default_factory=lambda: settings.qdrant_text_field
    )
    embed_model: str = field(
        default_factory=lambda: settings.embedding_model
    )
    top_k: int = field(default_factory=lambda: settings.retriever_top_k)
    min_score: float = field(
        default_factory=lambda: settings.retriever_min_score
    )
    max_context_chars: int = field(
        default_factory=lambda: settings.retriever_max_context_chars
    )
    _client: QdrantClient = field(init=False, repr=False)
    _openai: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("QDRANT_URL is not configured.")

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key or None,
            timeout=settings.qdrant_timeout_seconds,
        )
        self._openai = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.embedding_timeout_seconds,
            max_retries=1,
        )

        logger.info(
            "Qdrant retriever initialized collection=%s",
            self.collection,
        )

    def _embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            input=text,
            model=self.embed_model,
        )
        return response.data[0].embedding

    def retrieve(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        vector = self._embed(query)

        query_kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "query": vector,
            "limit": self.top_k,
            "with_payload": True,
        }

        # Zero means disabled. Let Qdrant apply threshold semantics for the
        # collection's configured distance metric instead of comparing scores
        # manually in application code.
        if self.min_score != 0.0:
            query_kwargs["score_threshold"] = self.min_score

        response = self._client.query_points(**query_kwargs)

        chunks: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        used_chars = 0

        for hit in response.points:
            payload = getattr(hit, "payload", None) or {}
            text = _safe_text(
                payload.get(self.text_field),
                self.max_context_chars,
            )

            if not text or text in seen_text:
                continue

            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break

            text = text[:remaining]
            seen_text.add(text)
            used_chars += len(text)

            score_value = getattr(hit, "score", None)
            score = (
                round(float(score_value), 6)
                if isinstance(score_value, (int, float))
                else None
            )

            title = _safe_text(
                payload.get("title")
                or payload.get("document_title")
                or payload.get("file_name"),
                200,
            )
            source = _safe_text(
                payload.get("source")
                or payload.get("document_id")
                or payload.get("file_path"),
                300,
            )

            chunks.append(
                {
                    "text": text,
                    "score": score,
                    "title": title,
                    "source": source,
                }
            )

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Knowledge retrieval completed count=%s duration_ms=%s",
            len(chunks),
            duration_ms,
        )

        if not chunks:
            return {
                "status": "not_found",
                "message": "No relevant private learning material was found.",
                "query": query,
                "chunks": [],
            }

        return {
            "status": "success",
            "query": query,
            "chunks": chunks,
        }


@lru_cache(maxsize=1)
def get_retriever() -> QdrantRetriever:
    """Create the Qdrant and OpenAI clients once per worker process."""

    return QdrantRetriever()


@tool
async def retrieve_knowledge(
    query: str,
    runtime: ToolRuntime[TutorRuntimeContext],
) -> dict[str, Any]:
    """Search uploaded lessons, notes, textbooks, and private tutor content.

    Use this only for private learning material. Do not use it for stable
    general knowledge or for an ordinary follow-up that can be answered from
    the current conversation or previously fetched room context.
    """

    clean_query = _normalize_query(query)
    if not clean_query:
        return {
            "status": "error",
            "message": "A non-empty private-knowledge query is required.",
        }

    context = runtime.context
    cached = context.get_cached_result("retrieval", clean_query)
    if cached is not None:
        cached["cached"] = True
        logger.info("Knowledge retrieval cache hit room=%s", context.room_id)
        return cached

    _emit_status("Let me check the lesson material.")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                get_retriever().retrieve,
                clean_query,
            ),
            timeout=settings.retriever_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Knowledge retrieval timed out room=%s",
            context.room_id,
        )
        return {
            "status": "timeout",
            "message": "The private learning search timed out.",
        }
    except asyncio.CancelledError:
        logger.info(
            "Knowledge retrieval cancelled room=%s",
            context.room_id,
        )
        raise
    except Exception:
        logger.exception(
            "Knowledge retrieval failed room=%s",
            context.room_id,
        )
        return {
            "status": "error",
            "message": "The private learning search is unavailable right now.",
        }

    context.set_cached_result("retrieval", clean_query, result)
    return result
