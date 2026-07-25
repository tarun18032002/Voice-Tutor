"""LiveKit voice tutor backed by the LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Any

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    PreemptiveGenerationOptions,
    TurnHandlingOptions,
    room_io,
)
from livekit.plugins import ai_coustics, langchain, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.config import (
    configure_logging,
    get_settings,
    validate_environment,
)
from agent.models import TutorRuntimeContext

configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)
server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Validate configuration and initialize reusable worker resources."""

    validate_environment(include_livekit=True)

    # Importing here makes validation happen before graph/model/client startup.
    from agent.graph import get_tutor_graph
    from agent.retriver import get_retriever
    from agent.websearch import get_web_search_client

    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["tutor_graph"] = get_tutor_graph()

    if settings.enable_retrieval:
        proc.userdata["retriever"] = get_retriever()

    if settings.enable_web_search:
        proc.userdata["web_search"] = get_web_search_client()

    logger.info("Voice Tutor worker prewarmed environment=%s", settings.app_env)


server.setup_fnc = prewarm


def _build_audio_input_options() -> room_io.AudioInputOptions:
    options: dict[str, Any] = {}

    if settings.enable_noise_cancellation:
        options["noise_cancellation"] = ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_S,
        )

    return room_io.AudioInputOptions(**options)


def _stream_mode() -> str | list[str]:
    if settings.enable_tool_status_messages:
        return ["messages", "custom"]
    return "messages"


@server.rtc_session(agent_name=settings.livekit_agent_name)
async def my_agent(ctx: JobContext) -> None:
    """Create one isolated tutor session for a LiveKit room."""

    room_id = ctx.room.name
    ctx.log_context_fields = {"room": room_id}

    runtime_context = TutorRuntimeContext(
        room_id=room_id,
        cache_ttl_seconds=settings.session_cache_ttl_seconds,
        max_cache_entries=settings.session_cache_max_entries,
    )

    graph_llm = langchain.LLMAdapter(
        graph=ctx.proc.userdata["tutor_graph"],
        config={
            "recursion_limit": settings.graph_recursion_limit,
            "tags": ["voice-tutor", settings.app_env],
            "metadata": {"room_id": room_id},
        },
        context=runtime_context,
        stream_mode=_stream_mode(),
    )

    session = AgentSession(
        stt=openai.STT(model=settings.stt_model),
        llm=graph_llm,
        tts=openai.TTS(
            model=settings.tts_model,
            voice=settings.tts_voice,
            instructions=(
                "Speak in a friendly, patient, clear, and conversational tone."
            ),
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            endpointing={
                "mode": settings.endpointing_mode,
                "min_delay": settings.endpointing_min_delay,
                "max_delay": settings.endpointing_max_delay,
            },
            interruption={
                "mode": settings.interruption_mode,
                "min_duration": settings.interruption_min_duration,
                "min_words": settings.interruption_min_words,
                "resume_false_interruption": (
                    settings.resume_false_interruption
                ),
            },
            preemptive_generation=PreemptiveGenerationOptions(
                enabled=settings.preemptive_generation_enabled,
                preemptive_tts=settings.preemptive_tts_enabled,
                max_speech_duration=(
                    settings.preemptive_max_speech_duration
                ),
                max_retries=settings.preemptive_max_retries,
            ),
        ),
    )

    await session.start(
        room=ctx.room,
        # The graph contains the single source of truth for tutor instructions.
        agent=Agent(instructions=""),
        room_options=room_io.RoomOptions(
            audio_input=_build_audio_input_options(),
        ),
    )

    await session.generate_reply(
        instructions=(
            "Greet the student in one short sentence and ask what topic "
            "they would like help with."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
