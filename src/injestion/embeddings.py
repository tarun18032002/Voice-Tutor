import os
from openai import OpenAI


class EmbeddingGenerator:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "text-embedding-3-small"

    def generate_embeddings(self, nodes):
        print(f"Generating embeddings for {len(nodes)} nodes...")

        texts = [node.text for node in nodes]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        for node, item in zip(nodes, response.data):
            node.embedding = item.embedding

        return nodes