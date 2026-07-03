import asyncio

from dotenv import load_dotenv, main

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    room_io,
)
from livekit.plugins import ai_coustics, langchain, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.graph import build_tutor_graph

load_dotenv(".env.local")


ASSISTANT_INSTRUCTIONS = """
You are a friendly AI tutor.
The student is interacting with you by voice.
Explain concepts clearly and help students learn.
Keep responses concise, simple, and conversational.
Ask short follow-up questions when useful.
"""


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm




@server.rtc_session(agent_name="voice-tutor")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    tutor_graph = build_tutor_graph()
    session = AgentSession(
        stt=openai.STT(
            model="gpt-4o-mini-transcribe",
        ),
        llm=langchain.LLMAdapter(
            graph=tutor_graph
        ),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="Speak in a friendly and conversational tone.",
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
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

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the student and ask what topic they want help with."
    )

async def test():
    load_dotenv(".env.local")

    from openai import OpenAI
    from agent.graph import build_tutor_graph
    import time

    # Preload OpenAI resources before graph starts
    preload_client = OpenAI()
    _ = preload_client.responses
    _ = preload_client.embeddings

    print("Building tutor graph...")

    agent = build_tutor_graph()

    start_time = time.time()
    result = await agent.ainvoke({
        "query": "What is CO2? Explain it like I am 10 years old."
    })
    end_time = time.time()
    print(f"Tutor graph built and executed in {end_time - start_time:.2f} seconds.")

    print(result["TutorAnswer"])
if __name__ == "__main__":
    # agents.cli.run_app(server)
    asyncio.run(test()) 