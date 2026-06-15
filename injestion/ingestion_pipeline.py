"""
Ingestion pipeline.

Run directly to ingest the chapter PDF:
    python ingestion_pipeline.py
"""

import os
import logging
from document_loader import DocumentLoader
from chunking import HierarchicalParentChildChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

FILE_PATH = "../data/chapter-1(10th std).pdf"


class IngestionPipeline:

    def __init__(self):
        self.loader = DocumentLoader()

        self.chunker = HierarchicalParentChildChunker(
            child_chunk_size=1024,
            child_chunk_overlap=200,
        )

        self.embedder = EmbeddingGenerator()

        self.vector_store = VectorStoreManager(
            url=os.getenv("VECTOR_STORE_HOST"),
            api_key=os.getenv("VECTOR_STORE_API_KEY"),
            collection_name="documents",
        )

    def ingest(self, file_path: str) -> dict:
        """
        Full ingestion pass for a single document.

        Returns a summary dict with counts for easy verification.
        """
        logger.info(f"Loading document: {file_path}")
        document = self.loader.load_document(file_path)

        logger.info("Creating chunks...")
        chunks = self.chunker.create_chunks(document)

        parent_nodes = chunks["parents"]
        child_nodes = chunks["children"]

        logger.info(
            f"Created {len(parent_nodes)} parent sections "
            f"and {len(child_nodes)} child chunks."
        )

        logger.info("Embedding child nodes...")
        embedded_children = self.embedder.generate_embeddings(child_nodes)

        logger.info("Storing child vectors...")
        self.vector_store.store_nodes(embedded_children)

        logger.info("Storing parent nodes (for context expansion)...")
        self._store_parents_without_embedding(parent_nodes)

        logger.info("Ingestion complete.")
        return {
            "file": document["file_name"],
            "parents_stored": len(parent_nodes),
            "children_stored": len(child_nodes),
        }

    def _store_parents_without_embedding(self, parent_nodes):
        from qdrant_client.models import PointStruct
        import uuid

        vector_size = 1536

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0] * vector_size,
                payload={
                    "text": node.text,
                    "metadata": node.metadata,
                },
            )
            for node in parent_nodes
        ]

        self.vector_store.client.upsert(
            collection_name=self.vector_store.collection_name,
            points=points,
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(
            f"PDF not found at '{FILE_PATH}'. "
            "Make sure the file is inside a 'data/' folder."
        )

    print(f"\n📄 Ingesting: {FILE_PATH}\n")
    pipeline = IngestionPipeline()
    result = pipeline.ingest(FILE_PATH)

    print("\n✅ Done!")
    print(f"   File            : {result['file']}")
    print(f"   Parents stored  : {result['parents_stored']}")
    print(f"   Children stored : {result['children_stored']}")