"""Shared LangGraph state and room-scoped runtime context."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Annotated, Any, Literal

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class TutorState(TypedDict):
    """State used during one LangGraph invocation."""

    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: NotRequired[int]


CacheKind = Literal["retrieval", "web"]


@dataclass(slots=True)
class CacheEntry:
    """One cached tool result."""

    query: str
    value: dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class TutorRuntimeContext:
    """Mutable context isolated to one LiveKit room/session."""

    room_id: str
    cache_ttl_seconds: float = 600.0
    max_cache_entries: int = 20
    _retrieval_cache: dict[str, CacheEntry] = field(
        default_factory=dict,
        repr=False,
    )
    _web_cache: dict[str, CacheEntry] = field(
        default_factory=dict,
        repr=False,
    )
    _latest_retrieval_key: str | None = field(default=None, repr=False)
    _latest_web_key: str | None = field(default=None, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    @staticmethod
    def make_cache_key(query: str) -> str:
        """Create a stable, case-insensitive key without changing provider text."""

        return " ".join(query.casefold().split()).strip()[:500]

    def get_cached_result(
        self,
        kind: CacheKind,
        query: str,
    ) -> dict[str, Any] | None:
        key = self.make_cache_key(query)
        if not key or self.cache_ttl_seconds <= 0:
            return None

        with self._lock:
            cache = self._select_cache(kind)
            self._prune_expired(cache)
            entry = cache.get(key)
            if entry is None:
                return None
            return copy.deepcopy(entry.value)

    def set_cached_result(
        self,
        kind: CacheKind,
        query: str,
        value: dict[str, Any],
    ) -> None:
        key = self.make_cache_key(query)
        if not key or self.cache_ttl_seconds <= 0:
            return

        with self._lock:
            cache = self._select_cache(kind)
            self._prune_expired(cache)

            if key not in cache and len(cache) >= self.max_cache_entries:
                oldest_key = min(
                    cache,
                    key=lambda current_key: cache[current_key].created_at,
                )
                cache.pop(oldest_key, None)

            cache[key] = CacheEntry(
                query=query,
                value=copy.deepcopy(value),
            )

            if kind == "retrieval":
                self._latest_retrieval_key = key
            else:
                self._latest_web_key = key

    def recent_reference_context(self, max_chars: int) -> str:
        """Return recent tool data for a likely conversational follow-up."""

        if max_chars <= 0 or self.cache_ttl_seconds <= 0:
            return ""

        with self._lock:
            self._prune_expired(self._retrieval_cache)
            self._prune_expired(self._web_cache)

            entries: list[tuple[str, CacheEntry]] = []

            if self._latest_retrieval_key:
                entry = self._retrieval_cache.get(self._latest_retrieval_key)
                if entry is not None:
                    entries.append(("private learning material", entry))

            if self._latest_web_key:
                entry = self._web_cache.get(self._latest_web_key)
                if entry is not None:
                    entries.append(("current web information", entry))

            entries.sort(key=lambda item: item[1].created_at, reverse=True)

            parts: list[str] = []
            used_chars = 0

            for label, entry in entries:
                serialized = json.dumps(
                    entry.value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                block = (
                    f"Source type: {label}\n"
                    f"Original query: {entry.query}\n"
                    f"Previously fetched result: {serialized}\n"
                )

                remaining = max_chars - used_chars
                if remaining <= 0:
                    break

                block = block[:remaining]
                parts.append(block)
                used_chars += len(block)

            return "\n".join(parts).strip()

    def clear(self) -> None:
        with self._lock:
            self._retrieval_cache.clear()
            self._web_cache.clear()
            self._latest_retrieval_key = None
            self._latest_web_key = None

    def _select_cache(self, kind: CacheKind) -> dict[str, CacheEntry]:
        if kind == "retrieval":
            return self._retrieval_cache
        if kind == "web":
            return self._web_cache
        raise ValueError(f"Unsupported cache kind: {kind}")

    def _prune_expired(self, cache: dict[str, CacheEntry]) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, entry in cache.items()
            if now - entry.created_at > self.cache_ttl_seconds
        ]

        for key in expired:
            cache.pop(key, None)

        if (
            self._latest_retrieval_key
            and self._latest_retrieval_key not in self._retrieval_cache
        ):
            self._latest_retrieval_key = None

        if self._latest_web_key and self._latest_web_key not in self._web_cache:
            self._latest_web_key = None
