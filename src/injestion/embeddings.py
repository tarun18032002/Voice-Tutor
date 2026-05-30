import os

from google import genai


class EmbeddingGenerator:

    def __init__(self):

        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.model_name = "text-embedding-004"

    def generate_embeddings(self, nodes):

        texts = [node.text for node in nodes]

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )

        for node, embedding in zip(
            nodes,
            response.embeddings
        ):
            node.embedding = embedding.values

        return nodes