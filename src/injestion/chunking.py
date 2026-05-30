import re
import uuid

from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter


class HierarchicalParentChildChunker:

    def __init__(
        self,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 100
    ):
        self.child_splitter = SentenceSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )

    def _extract_sections(self, markdown_text: str):
        """
        Split document using markdown headings.
        """

        pattern = r"(?=^#{1,6}\s+)"
        sections = re.split(
            pattern,
            markdown_text,
            flags=re.MULTILINE
        )

        return [
            section.strip()
            for section in sections
            if section.strip()
        ]

    def create_chunks(self, document_data: dict):

        markdown_content = document_data["content"]
        print(f"Document length: {len(markdown_content)} characters and data in document_data: {markdown_content[:500]}...")  # Print first 500 characters for verification
        sections = self._extract_sections(
            markdown_content
        )

        parent_nodes = []
        child_nodes = []

        for section in sections:

            parent_id = str(uuid.uuid4())

            parent_document = Document(
                text=section,
                metadata={
                    "type": "parent",
                    "parent_id": parent_id,
                    "source": document_data["file_name"]
                }
            )

            parent_nodes.append(parent_document)

            children = self.child_splitter.get_nodes_from_documents(
                [parent_document]
            )

            for child in children:

                child.metadata.update(
                    {
                        "type": "child",
                        "parent_id": parent_id,
                        "source": document_data["file_name"]
                    }
                )

                child_nodes.append(child)

        return {
            "parents": parent_nodes,
            "children": child_nodes
        }