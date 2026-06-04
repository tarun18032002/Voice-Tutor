"""Knowledge Agent — fetches accurate context from the vector store via RAG."""

from __future__ import annotations

import json
import os

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.agents.schemas import KnowledgeAgentOutput, StudentProfile


class KnowledgeAgent:
    """
    Retrieves relevant concepts, examples, and facts for a student query
    using semantic search over the Qdrant vector store.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        top_k: int = 5,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            url=os.getenv("VECTOR_STORE_HOST"),
            api_key=os.getenv("VECTOR_STORE_API_KEY"),
        )
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        student_profile: StudentProfile,
    ) -> KnowledgeAgentOutput:
        """Embed the query, retrieve chunks, parse into structured output."""

        query_vector = self._embed(query)
        hits = self._search(query_vector)
        return self._parse_hits(hits)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _search(self, vector: list[float]) -> list[dict]:
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=self.top_k,
            with_payload=True,
        )
        return [hit.payload for hit in results]

    def _parse_hits(self, hits: list[dict]) -> KnowledgeAgentOutput:
        """
        Naively bucket retrieved chunks into concepts / examples / facts.
        A more advanced version would ask the LLM to classify each chunk.
        """
        concepts: list[str] = []
        examples: list[str] = []
        facts: list[str] = []
        raw_parts: list[str] = []

        for hit in hits:
            text: str = hit.get("text", "")
            raw_parts.append(text)

            lower = text.lower()
            if any(kw in lower for kw in ["for example", "e.g.", "such as", "instance"]):
                examples.append(text)
            elif any(kw in lower for kw in ["is defined as", "refers to", "is called", "means"]):
                concepts.append(text)
            else:
                facts.append(text)

        return KnowledgeAgentOutput(
            concepts=concepts,
            examples=examples,
            facts=facts,
            raw_context="\n\n".join(raw_parts),
        )