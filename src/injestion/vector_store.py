from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid


class VectorStoreManager:

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str
    ):
        self.client = QdrantClient(
            host=host,
            port=port
        )

        self.collection_name = collection_name

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