"""Retrieval module: vector search + parent context fetch + LLM answer generation."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class Retriever:
    """
    Searches child nodes by embedding similarity, then fetches their parent
    chunks for richer context before passing everything to the LLM.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
    ):
        self.client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
        self.collection_name = collection_name
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._use_query_points: bool | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            logger.warning(
                "Collection '%s' not found or unreachable: %s. "
                "Run ingestion first: python ingestion_pipeline.py",
                self.collection_name, e,
            )
            return False

    def _embed_query(self, query: str) -> list[float]:
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _search_children(self, query_vector: list[float]) -> list[dict]:
        raw_hits = self._vector_search(query_vector)

        chunks = []
        for hit in raw_hits:
            payload = hit.payload or {}
            meta = payload.get("metadata", {})
            chunks.append(
                {
                    "text":      payload.get("text", ""),
                    "score":     hit.score,
                    "parent_id": meta.get("parent_id"),
                    "source":    meta.get("source", "unknown"),
                    "type":      meta.get("type", "child"),
                }
            )
        return chunks

    def _vector_search(self, query_vector: list[float]):
        if self._use_query_points is True:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.top_k,
                with_payload=True,
            ).points

        if self._use_query_points is False:
            return self._legacy_search(query_vector)

        # First call — probe query_points
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.top_k,
                with_payload=True,
            ).points
            self._use_query_points = True
            logger.debug("Retriever: using query_points API")
            return result
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.info("Falling back to legacy /search endpoint.")
                self._use_query_points = False
                return self._legacy_search(query_vector)
            raise

    def _legacy_search(self, query_vector: list[float]):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.top_k,
            with_payload=True,
        )

    # ------------------------------------------------------------------
    # Parent fetch
    # ------------------------------------------------------------------

    def _fetch_parents(self, parent_ids: list[str]) -> dict[str, str]:
        if not parent_ids:
            return {}

        parent_map: dict[str, str] = {}
        for pid in set(parent_ids):
            try:
                scroll_result, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.type",
                                match=MatchValue(value="parent"),
                            ),
                            FieldCondition(
                                key="metadata.parent_id",
                                match=MatchValue(value=pid),
                            ),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                )
                if scroll_result:
                    parent_map[pid] = scroll_result[0].payload.get("text", "")
            except Exception as e:
                logger.warning("Could not fetch parent %s: %s", pid, e)

        return parent_map

    # ------------------------------------------------------------------
    # Public retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> dict:
        if not self._collection_exists():
            logger.warning(
                "Retriever: collection '%s' missing – returning empty context.",
                self.collection_name,
            )
            return {"query": query, "context": "", "child_hits": []}

        query_vector = self._embed_query(query)
        child_hits   = self._search_children(query_vector)

        parent_ids = [h["parent_id"] for h in child_hits if h.get("parent_id")]
        parent_map = self._fetch_parents(parent_ids)

        seen_parents: set[str] = set()
        context_parts: list[str] = []

        for hit in child_hits:
            pid = hit.get("parent_id")
            if pid and pid in parent_map and pid not in seen_parents:
                context_parts.append(parent_map[pid])
                seen_parents.add(pid)
            elif not pid or pid not in parent_map:
                context_parts.append(hit["text"])

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "Retriever: %d chunks retrieved, context=%d chars",
            len(child_hits), len(context),
        )

        return {"query": query, "context": context, "child_hits": child_hits}


# ── LangGraph node wrapper ────────────────────────────────────────────────────

class RetrieverNode:
    """LangGraph-compatible retriever node."""

    def __init__(self) -> None:
        url     = os.environ["VECTOR_STORE_HOST"]
        api_key = os.environ["VECTOR_STORE_API_KEY"]
        self._retriever = Retriever(url=url, api_key=api_key)

    def __call__(self, state) -> dict:
        retrieved = self._retriever.retrieve(state.StudentQuery)
        return {"RetrievedInfo": retrieved["context"]}


def retriever_node(state) -> dict:
    return RetrieverNode()(state)



