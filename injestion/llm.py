"""LLM Client for production-level inference and structured output."""

import os
import logging
from typing import Optional, Type, TypeVar, Any
from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """
    Production-grade wrapper around the OpenAI API for consistent interface.
    
    Supports structured outputs, batch processing, and error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: int = 30,
    ) -> None:
        """
        Initialize LLM Client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY.
            model: Model to use. Defaults to gpt-4o-mini.
            timeout: Request timeout in seconds.
            
        Raises:
            ValueError: If API key is not provided or found.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        logger.info(f"Initialized LLM_Client with model: {model}")
    
    def generate_response(
        self,
        model: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        input: Optional[list[dict]] = None,
        text_format: Optional[Type[T]] = None,
        reasoning: Optional[dict] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Generate a response from OpenAI API.
        
        Args:
            model: Model name. If None, uses default.
            messages: Messages list (new format).
            input: Legacy input format (converted to messages).
            text_format: Pydantic model for structured output.
            reasoning: Extended thinking config.
            temperature: Sampling temperature.
            max_tokens: Max tokens in response.
            **kwargs: Additional parameters.
            
        Returns:
            Parsed output if text_format provided, otherwise raw text.
            
        Raises:
            RateLimitError: If rate limited.
            APIError: If API call fails.
            ValueError: If response is invalid.
        """
        try:
            model = model or self.model
            # Convert legacy input format to messages
            if input is not None and messages is None:
                messages = input
            
            if not messages:
                raise ValueError("Either 'messages' or 'input' must be provided.")
            
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if reasoning:
                request_params["reasoning"] = reasoning
            
            # Use structured output if text_format is provided
            if text_format:
                # For structured output, add response_format to request
                schema = text_format.model_json_schema()
                # OpenAI requires additionalProperties to be false for strict mode
                schema["additionalProperties"] = False
                
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": text_format.__name__,
                        "schema": schema,
                        "strict": True,
                    }
                }
                response = self.client.chat.completions.create(**request_params)
                text = response.choices[0].message.content
                
                # Parse the JSON response
                import json
                parsed_data = json.loads(text)
                parsed = text_format(**parsed_data)
                logger.debug(f"Generated structured response: {type(parsed).__name__}")
                return parsed
            else:
                response = self.client.chat.completions.create(**request_params)
                text = response.choices[0].message.content
                logger.debug(f"Generated text response ({len(text)} chars)")
                return text
        
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise
        except APIError as e:
            logger.error(f"API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: list[str],
        text_format: Optional[Type[T]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> list[Any]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts.
            text_format: Pydantic model for structured output.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.
            
        Returns:
            List of responses.
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                response = self.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    text_format=text_format,
                    temperature=temperature,
                    **kwargs
                )
                results.append(response)
                logger.info(f"Processed batch item {i + 1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"Failed to process prompt {i}: {e}")
                results.append(None)
        
        return results
    
    def parse(
        self,
        messages: list[dict],
        response_format: Optional[Type[T]] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Alias for generate_response with structured output.
        Called by all agent nodes as client.parse(..., response_format=SomeModel).

        Args:
            messages: Chat messages list.
            response_format: Pydantic model for structured output.
            reasoning_effort: Ignored (kept for API compatibility).
            **kwargs: Additional parameters passed to generate_response.

        Returns:
            Parsed Pydantic model instance.
        """
        # reasoning_effort is not a standard OpenAI chat param — drop it silently
        return self.generate_response(
            messages=messages,
            text_format=response_format,
            **kwargs,
        )

    def change_model(self, model: str) -> None:
        """
        Change the default model.
        
        Args:
            model: New model name.
        """
        self.model = model
        logger.info(f"Changed model to: {model}")