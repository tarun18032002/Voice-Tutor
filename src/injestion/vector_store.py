from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid


class VectorStoreManager:

    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(
            url=url,
            api_key=api_key
        )

        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )

    def store_nodes(self, nodes):
        points = []

        for node in nodes:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=node.embedding,
                    payload={
                        "text": node.text,
                        "metadata": node.metadata
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )