"""
retriver.py
-----------
RAG Retriever node.

Connects to your existing vector store (Pinecone / Chroma / Qdrant etc.)
and retrieves the top-k most relevant chunks for the student query.

Configure via environment variables:
    RETRIEVER_BACKEND  = "pinecone" | "chroma" | "qdrant"   (default: chroma)
    PINECONE_API_KEY   / PINECONE_INDEX_NAME
    CHROMA_HOST        / CHROMA_PORT / CHROMA_COLLECTION
    QDRANT_URL         / QDRANT_COLLECTION / QDRANT_API_KEY

The node is backend-agnostic: swap _build_retriever() for your own
implementation without touching any other file.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .models import TutorAgentState

logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """Return retrieved context as a single string."""


# ── Chroma backend ────────────────────────────────────────────────────────────

@dataclass
class ChromaRetriever(BaseRetriever):
    host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "tutor_docs"))

    def __post_init__(self) -> None:
        import chromadb  # lazy import – only required if this backend is used
        self._client = chromadb.HttpClient(host=self.host, port=self.port)
        self._collection = self._client.get_collection(self.collection_name)
        logger.info("ChromaRetriever connected | collection=%s", self.collection_name)

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        results = self._collection.query(query_texts=[query], n_results=top_k)
        docs: list[str] = results.get("documents", [[]])[0]
        if not docs:
            logger.warning("ChromaRetriever: no documents found for query.")
            return ""
        combined = "\n\n---\n\n".join(docs)
        logger.debug("ChromaRetriever: retrieved %d chunks.", len(docs))
        return combined


# ── Pinecone backend ──────────────────────────────────────────────────────────

@dataclass
class PineconeRetriever(BaseRetriever):
    api_key: str = field(default_factory=lambda: os.environ["PINECONE_API_KEY"])
    index_name: str = field(default_factory=lambda: os.environ["PINECONE_INDEX_NAME"])
    embed_model: str = field(default="text-embedding-3-small")

    def __post_init__(self) -> None:
        from pinecone import Pinecone  # lazy import
        from openai import OpenAI

        self._pc = Pinecone(api_key=self.api_key)
        self._index = self._pc.Index(self.index_name)
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        logger.info("PineconeRetriever connected | index=%s", self.index_name)

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(input=text, model=self.embed_model)
        return resp.data[0].embedding

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        vector = self._embed(query)
        results = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = results.get("matches", [])
        if not matches:
            logger.warning("PineconeRetriever: no matches found for query.")
            return ""
        docs = [m["metadata"].get("text", "") for m in matches if m.get("metadata")]
        combined = "\n\n---\n\n".join(filter(None, docs))
        logger.debug("PineconeRetriever: retrieved %d chunks.", len(docs))
        return combined


# ── Qdrant backend ────────────────────────────────────────────────────────────

@dataclass
class QdrantRetriever(BaseRetriever):
    url: str = field(default_factory=lambda: os.environ["QDRANT_URL"])
    collection: str = field(default_factory=lambda: os.environ["QDRANT_COLLECTION"])
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    embed_model: str = field(default="text-embedding-3-small")

    def __post_init__(self) -> None:
        from qdrant_client import QdrantClient  # lazy import
        from openai import OpenAI

        self._client = QdrantClient(url=self.url, api_key=self.api_key or None)
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        logger.info("QdrantRetriever connected | collection=%s", self.collection)

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(input=text, model=self.embed_model)
        return resp.data[0].embedding

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        vector = self._embed(query)
        results = self._client.search(
            collection_name=self.collection, query_vector=vector, limit=top_k
        )
        docs = [hit.payload.get("text", "") for hit in results if hit.payload]
        combined = "\n\n---\n\n".join(filter(None, docs))
        logger.debug("QdrantRetriever: retrieved %d chunks.", len(docs))
        return combined


# ── Factory ───────────────────────────────────────────────────────────────────

def _build_retriever() -> BaseRetriever:
    backend = os.getenv("RETRIEVER_BACKEND", "chroma").lower()
    if backend == "pinecone":
        return PineconeRetriever()
    if backend == "qdrant":
        return QdrantRetriever()
    return ChromaRetriever()  # default


# ── Node ──────────────────────────────────────────────────────────────────────

class RetrieverNode:
    """LangGraph-compatible RAG retrieval node."""

    def __init__(self) -> None:
        self._retriever: BaseRetriever = _build_retriever()

    def __call__(self, state: TutorAgentState) -> dict:
        logger.info("RetrieverNode: querying for '%s'", state.StudentQuery[:80])
        retrieved = self._retriever.retrieve(state.StudentQuery)
        if not retrieved:
            logger.warning("RetrieverNode: nothing retrieved – orchestrator will re-route.")
        return {"RetrievedInfo": retrieved}


# ── Module-level function alias ───────────────────────────────────────────────

def retriever_node(state: TutorAgentState) -> dict:
    return RetrieverNode()(state)