from openai import OpenAI
from dataclasses import dataclass

@dataclass
class LLM_Client:
    """
    A wrapper around the OpenAI API to provide a consistent interface for generating responses.
    """
    api_key: str

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, model: str, reasoning: dict, input: list, text_format: type) -> str:
        response = self.client.responses.parse(
            model=model,
            reasoning=reasoning,
            input=input,
            text_format=text_format,
        )
        return response.output_parsed