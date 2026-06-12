from pathlib import Path
from docling.document_converter import DocumentConverter


class DocumentLoader:

    def __init__(self):
        self.converter = DocumentConverter()

    def load_document(self, file_path: str) -> dict:

        result = self.converter.convert(file_path)

        return {
            "file_name": Path(file_path).name,
            "file_path": file_path,
            "content": result.document.export_to_markdown()
        }