from openai import OpenAI, AsyncOpenAI
from dataclasses import dataclass
from langsmith import traceable


@dataclass
class LLMClient:
    api_key: str

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    @traceable(name="OpenAI Structured Response", run_type="llm")
    def generate_response(self, input: list, text_format: type, model: str = "gpt-5-mini"):
        response = self.client.responses.parse(
            model=model,
            input=input,
            text_format=text_format,
        )
        return response.output_parsed
    
    @traceable(name="OpenAI Structured Response", run_type="llm")
    async def agenerate_response(self, input: list, text_format: type, model: str = "gpt-5-mini"):
        response = await self.async_client.responses.parse(
            model=model,
            input=input,
            text_format=text_format,
        )
        return response.output_parsed