import os
from openai import OpenAI
from qdrant_client import QdrantClient


class QueryService:

    def __init__(self):
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.qdrant_client = QdrantClient(
            url=os.getenv("VECTOR_STORE_HOST"),
            api_key=os.getenv("VECTOR_STORE_API_KEY")
        )

        self.collection_name = "documents"

    def search(self, query: str, limit: int = 5):
        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        query_vector = embedding_response.data[0].embedding

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

        return results.points