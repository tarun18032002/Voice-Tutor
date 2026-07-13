"""Retrieval module: Qdrant vector search + context fetch."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langsmith import traceable
from langchain.tools import tool
import json
import asyncio


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
    
    @traceable(name="Qdrant Retrieval", run_type="retriever")
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
    

@tool
async def retrieve_knowledge(query: str) -> str:
    """
    Search the tutor's private knowledge base.

    Use this tool when the student asks about uploaded documents,
    course material, textbook content, or private tutor information.

    Do not use it when the question can be answered from the existing
    conversation or from common general knowledge.
    """

    try:
        # If retrieve() is synchronous, run it in a worker thread so that
        # it does not block LiveKit's asyncio event loop.
        retrieved = await asyncio.to_thread(
            QdrantRetriever().retrieve,
            query,
        )

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "retrieved_info": retrieved,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("Knowledge retrieval failed")

        return json.dumps(
            {
                "status": "error",
                "query": query,
                "message": str(exc),
            }
        )
