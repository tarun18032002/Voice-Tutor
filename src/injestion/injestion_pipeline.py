from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingGenerator
from ingestion.vector_store import VectorStoreManager


class IngestionPipeline:

    def __init__(self):

        self.loader = DocumentLoader()

        self.chunker = DocumentChunker(
            chunk_size=1024,
            chunk_overlap=200
        )

        self.embedder = EmbeddingGenerator()

        self.vector_store = VectorStoreManager(
            host="localhost",
            port=6333,
            collection_name="documents"
        )

    def ingest(self, file_path: str):

        print("Loading document...")
        document = self.loader.load_document(file_path)

        print("Creating chunks...")
        chunks = self.chunker.create_chunks(document)

        print("Generating embeddings...")
        embedded_chunks = self.embedder.generate_embeddings(
            chunks
        )

        print("Storing vectors...")
        self.vector_store.store_nodes(
            embedded_chunks
        )

        print("Ingestion completed.")