"""Central configuration for the Voice Tutor application."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def load_environment() -> None:
    """Load a local env file without overriding deployed environment values."""

    env_file = os.getenv("ENV_FILE", ".env.local").strip()
    if not env_file:
        return

    path = Path(env_file)
    if path.is_file():
        load_dotenv(path, override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Environment variable {name} must be one of: "
        "1, 0, true, false, yes, no, on, off."
    )


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)

    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}.")

    return value


def _env_float(
    name: str,
    default: float,
    minimum: float | None = None,
) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)

    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}.")

    return value


load_environment()


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_env: str
    log_level: str

    openai_api_key: str = field(repr=False)
    tutor_model: str = "gpt-5.4-mini"
    llm_timeout_seconds: float = 10.0
    llm_total_timeout_seconds: float = 12.0
    llm_max_retries: int = 1
    llm_max_output_tokens: int = 350
    max_llm_calls_per_turn: int = 3
    graph_recursion_limit: int = 8

    enable_retrieval: bool = True
    qdrant_url: str = ""
    qdrant_api_key: str = field(default="", repr=False)
    qdrant_collection: str = "documents"
    qdrant_text_field: str = "text"
    qdrant_timeout_seconds: float = 3.0
    embedding_model: str = "text-embedding-3-small"
    embedding_timeout_seconds: float = 5.0
    retriever_timeout_seconds: float = 9.0
    retriever_top_k: int = 5
    retriever_min_score: float = 0.0
    retriever_max_context_chars: int = 10000

    enable_web_search: bool = True
    tavily_api_key: str = field(default="", repr=False)
    web_search_timeout_seconds: float = 6.0
    web_search_max_results: int = 3
    web_search_max_context_chars: int = 6000
    web_search_depth: str = "basic"

    session_cache_ttl_seconds: float = 600.0
    session_cache_max_entries: int = 20
    session_reference_context_chars: int = 7000
    enable_tool_status_messages: bool = True

    livekit_url: str = ""
    livekit_api_key: str = field(default="", repr=False)
    livekit_api_secret: str = field(default="", repr=False)
    livekit_agent_name: str = "voice-tutor"

    stt_model: str = "gpt-4o-mini-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "ash"

    enable_noise_cancellation: bool = True
    endpointing_mode: str = "fixed"
    endpointing_min_delay: float = 0.5
    endpointing_max_delay: float = 3.0
    interruption_mode: str = "adaptive"
    interruption_min_duration: float = 0.5
    interruption_min_words: int = 0
    resume_false_interruption: bool = True

    preemptive_generation_enabled: bool = True
    preemptive_tts_enabled: bool = False
    preemptive_max_speech_duration: float = 8.0
    preemptive_max_retries: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings object for the worker process."""

    return Settings(
        app_env=os.getenv("APP_ENV", "development").strip().lower(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        tutor_model=os.getenv(
            "TUTOR_MODEL",
            "gpt-5.4-mini",
        ).strip(),
        llm_timeout_seconds=_env_float(
            "LLM_TIMEOUT_SECONDS",
            10.0,
            minimum=1.0,
        ),
        llm_total_timeout_seconds=_env_float(
            "LLM_TOTAL_TIMEOUT_SECONDS",
            12.0,
            minimum=1.0,
        ),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 1, minimum=0),
        llm_max_output_tokens=_env_int(
            "LLM_MAX_OUTPUT_TOKENS",
            350,
            minimum=64,
        ),
        max_llm_calls_per_turn=_env_int(
            "MAX_LLM_CALLS_PER_TURN",
            3,
            minimum=1,
        ),
        graph_recursion_limit=_env_int(
            "GRAPH_RECURSION_LIMIT",
            8,
            minimum=4,
        ),
        enable_retrieval=_env_bool("ENABLE_RETRIEVAL", True),
        qdrant_url=os.getenv("QDRANT_URL", "").strip(),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        qdrant_collection=os.getenv(
            "QDRANT_COLLECTION",
            "documents",
        ).strip(),
        qdrant_text_field=os.getenv(
            "QDRANT_TEXT_FIELD",
            "text",
        ).strip(),
        qdrant_timeout_seconds=_env_float(
            "QDRANT_TIMEOUT_SECONDS",
            3.0,
            minimum=0.5,
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
        ).strip(),
        embedding_timeout_seconds=_env_float(
            "EMBEDDING_TIMEOUT_SECONDS",
            5.0,
            minimum=0.5,
        ),
        retriever_timeout_seconds=_env_float(
            "RETRIEVER_TIMEOUT_SECONDS",
            9.0,
            minimum=1.0,
        ),
        retriever_top_k=_env_int("RETRIEVER_TOP_K", 5, minimum=1),
        retriever_min_score=_env_float(
            "RETRIEVER_MIN_SCORE",
            0.0,
        ),
        retriever_max_context_chars=_env_int(
            "RETRIEVER_MAX_CONTEXT_CHARS",
            10000,
            minimum=1000,
        ),
        enable_web_search=_env_bool("ENABLE_WEB_SEARCH", True),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        web_search_timeout_seconds=_env_float(
            "WEB_SEARCH_TIMEOUT_SECONDS",
            6.0,
            minimum=1.0,
        ),
        web_search_max_results=_env_int(
            "WEB_SEARCH_MAX_RESULTS",
            3,
            minimum=1,
        ),
        web_search_max_context_chars=_env_int(
            "WEB_SEARCH_MAX_CONTEXT_CHARS",
            6000,
            minimum=1000,
        ),
        web_search_depth=os.getenv(
            "WEB_SEARCH_DEPTH",
            "basic",
        ).strip().lower(),
        session_cache_ttl_seconds=_env_float(
            "SESSION_CACHE_TTL_SECONDS",
            600.0,
            minimum=0.0,
        ),
        session_cache_max_entries=_env_int(
            "SESSION_CACHE_MAX_ENTRIES",
            20,
            minimum=1,
        ),
        session_reference_context_chars=_env_int(
            "SESSION_REFERENCE_CONTEXT_CHARS",
            7000,
            minimum=1000,
        ),
        enable_tool_status_messages=_env_bool(
            "ENABLE_TOOL_STATUS_MESSAGES",
            True,
        ),
        livekit_url=os.getenv("LIVEKIT_URL", "").strip(),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", "").strip(),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "").strip(),
        livekit_agent_name=os.getenv(
            "LIVEKIT_AGENT_NAME",
            "voice-tutor",
        ).strip(),
        stt_model=os.getenv(
            "STT_MODEL",
            "gpt-4o-mini-transcribe",
        ).strip(),
        tts_model=os.getenv(
            "TTS_MODEL",
            "gpt-4o-mini-tts",
        ).strip(),
        tts_voice=os.getenv("TTS_VOICE", "ash").strip(),
        enable_noise_cancellation=_env_bool(
            "ENABLE_NOISE_CANCELLATION",
            True,
        ),
        endpointing_mode=os.getenv(
            "ENDPOINTING_MODE",
            "fixed",
        ).strip().lower(),
        endpointing_min_delay=_env_float(
            "ENDPOINTING_MIN_DELAY",
            0.5,
            minimum=0.0,
        ),
        endpointing_max_delay=_env_float(
            "ENDPOINTING_MAX_DELAY",
            3.0,
            minimum=0.0,
        ),
        interruption_mode=os.getenv(
            "INTERRUPTION_MODE",
            "adaptive",
        ).strip().lower(),
        interruption_min_duration=_env_float(
            "INTERRUPTION_MIN_DURATION",
            0.5,
            minimum=0.0,
        ),
        interruption_min_words=_env_int(
            "INTERRUPTION_MIN_WORDS",
            0,
            minimum=0,
        ),
        resume_false_interruption=_env_bool(
            "RESUME_FALSE_INTERRUPTION",
            True,
        ),
        preemptive_generation_enabled=_env_bool(
            "PREEMPTIVE_GENERATION_ENABLED",
            True,
        ),
        preemptive_tts_enabled=_env_bool(
            "PREEMPTIVE_TTS_ENABLED",
            False,
        ),
        preemptive_max_speech_duration=_env_float(
            "PREEMPTIVE_MAX_SPEECH_DURATION",
            8.0,
            minimum=1.0,
        ),
        preemptive_max_retries=_env_int(
            "PREEMPTIVE_MAX_RETRIES",
            2,
            minimum=0,
        ),
    )


def validate_environment(*, include_livekit: bool) -> None:
    """Fail fast when a required production value is missing or invalid."""

    settings = get_settings()
    missing: list[str] = []

    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")

    if settings.enable_retrieval:
        if not settings.qdrant_url:
            missing.append("QDRANT_URL")
        if not settings.qdrant_collection:
            missing.append("QDRANT_COLLECTION")

    if settings.enable_web_search and not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")

    if include_livekit:
        if not settings.livekit_url:
            missing.append("LIVEKIT_URL")
        if not settings.livekit_api_key:
            missing.append("LIVEKIT_API_KEY")
        if not settings.livekit_api_secret:
            missing.append("LIVEKIT_API_SECRET")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(sorted(set(missing)))
        )

    if settings.web_search_depth not in {
        "basic",
        "advanced",
        "fast",
        "ultra-fast",
    }:
        raise RuntimeError(
            "WEB_SEARCH_DEPTH must be one of: "
            "advanced, basic, fast, ultra-fast"
        )

    if settings.endpointing_mode not in {"fixed", "dynamic"}:
        raise RuntimeError("ENDPOINTING_MODE must be fixed or dynamic.")

    if settings.interruption_mode not in {"adaptive", "vad"}:
        raise RuntimeError("INTERRUPTION_MODE must be adaptive or vad.")

    if settings.endpointing_max_delay < settings.endpointing_min_delay:
        raise RuntimeError(
            "ENDPOINTING_MAX_DELAY must be greater than or equal to "
            "ENDPOINTING_MIN_DELAY."
        )


def configure_logging() -> None:
    """Configure process logging once."""

    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
