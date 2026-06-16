"""Retrieval module: Qdrant vector search + context fetch."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """Return retrieved context as a single string."""
        raise NotImplementedError


# ── Qdrant backend ────────────────────────────────────────────────────────────

@dataclass
class QdrantRetriever(BaseRetriever):
    url: str = field(default_factory=lambda: os.environ["QDRANT_URL"])
    collection: str = field(default_factory=lambda: "documents")
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    embed_model: str = field(default="text-embedding-3-small")

    def __post_init__(self) -> None:
        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key or None,
        )

        self._openai = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )

        logger.info(
            "QdrantRetriever connected | collection=%s",
            self.collection,
        )

    def _embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            input=text,
            model=self.embed_model,
        )

        return response.data[0].embedding

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
    
        vector = self._embed(query)

        results = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
        ).points

        docs = [
            hit.payload["text"]
            for hit in results
            if hit.payload and "text" in hit.payload
        ]

        if not docs:
            return ""

        return "\n\n---\n\n".join(docs)


# ── Node ──────────────────────────────────────────────────────────────────────

class RetrieverNode:
    """LangGraph-compatible Qdrant retriever node."""

    def __init__(self) -> None:
        self._retriever = QdrantRetriever()

    def __call__(self, state) -> dict:
        retrieved = self._retriever.retrieve(state.StudentQuery)

        return {
            "RetrievedInfo": retrieved,
        }


def retriever_node(state) -> dict:
    return RetrieverNode()(state)