"""
voice_agent.py
--------------
LiveKit voice tutor backed by a LangGraph workflow.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
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
from livekit.plugins import (
    ai_coustics,
    langchain,
    openai,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.test import get_tutor_graph


load_dotenv(".env.local")

logger = logging.getLogger(__name__)


ASSISTANT_INSTRUCTIONS = """
You are a friendly AI tutor speaking with a student by voice.

Explain concepts clearly.
Keep responses concise, simple, and conversational.
Use short sentences.
Ask one short follow-up question when it supports learning.

The underlying tutor workflow may use private learning material or web
search when required. Never mention internal tools, graphs, agents, or
routing decisions.
"""


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """
    Load expensive local resources once when the worker process starts.
    """

    proc.userdata["vad"] = silero.VAD.load()

    # The graph itself is already cached globally in agent.graph.
    # Accessing it here ensures graph imports and clients are initialized
    # before the first student turn.
    proc.userdata["tutor_graph"] = get_tutor_graph()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="voice-tutor")
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    tutor_graph = ctx.proc.userdata["tutor_graph"]

    graph_llm = langchain.LLMAdapter(
        graph=tutor_graph,
        stream_mode="messages",
    )

    session = AgentSession(
    stt=openai.STT(
        model="gpt-4o-mini-transcribe",
    ),
    llm=langchain.LLMAdapter(
        graph=tutor_graph,
        stream_mode="messages",
    ),
    tts=openai.TTS(
        model="gpt-4o-mini-tts",
        voice="ash",
        instructions=(
            "Speak in a friendly, patient, and conversational tone."
        ),
    ),
    vad=ctx.proc.userdata["vad"],
    turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            preemptive_generation=PreemptiveGenerationOptions(
                enabled=True,
                preemptive_tts=True,
            ),
        ),
)
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=ASSISTANT_INSTRUCTIONS,
        ),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S,
                ),
            ),
        ),
    )

    await session.generate_reply(
        instructions=(
            "Greet the student briefly and ask what topic they would "
            "like help with."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)