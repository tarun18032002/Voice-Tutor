"""Tutor chat model configuration."""

from __future__ import annotations

from langchain.chat_models import init_chat_model

from agent.config import get_settings

settings = get_settings()

model = init_chat_model(
    settings.tutor_model,
    temperature=0,
    api_key=settings.openai_api_key or None,
    timeout=settings.llm_timeout_seconds,
    max_retries=settings.llm_max_retries,
    max_tokens=settings.llm_max_output_tokens,
)
