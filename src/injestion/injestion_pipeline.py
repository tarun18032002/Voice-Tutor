from src.injestion.document_loader import DocumentLoader
from src.injestion.chunking import HierarchicalParentChildChunker
from src.injestion.embeddings import EmbeddingGenerator
from src.injestion.vector_store import VectorStoreManager

import os

class IngestionPipeline:

    def __init__(self):

        self.loader = DocumentLoader()

        self.chunker = HierarchicalParentChildChunker(
            child_chunk_size=1024,
            child_chunk_overlap=200
        )

        self.embedder = EmbeddingGenerator()

        self.vector_store = VectorStoreManager(
            url=os.getenv("VECTOR_STORE_HOST"),
            api_key=os.getenv("VECTOR_STORE_API_KEY"),
            collection_name="documents"
        )

    def ingest(self, file_path: str):

        print("Loading document...")
        document = self.loader.load_document(file_path)

        print("Creating chunks...")
        chunks = self.chunker.create_chunks(document)

        print("Generating embeddings...")

        child_nodes = chunks["children"]

        embedded_chunks = self.embedder.generate_embeddings(child_nodes)

        print("Storing vectors...")
        self.vector_store.store_nodes(
            embedded_chunks
        )

        print("Ingestion completed.")