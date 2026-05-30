import logging

from livekit.agents import JobContext
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam,silero
from livekit.agents import stt

from .retriever import QueryService



logger = logging.getLogger("sarvam-voice-agent")
logger.setLevel(logging.INFO)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a helpful voice assistant.
            Use the RAG context when it is relevant.
            Keep answers short and conversational.
            """,

            # stt=sarvam.STT(
            #     language="unknown",
            #     model="saaras:v3",
            #     mode="transcribe",
            #     flush_signal=True,
            # ),
            stt = stt.StreamAdapter(
                    stt=openai.STT(),
                    vad=silero.VAD.load()
                ),

            llm=openai.LLM(model="gpt-4o-mini",max_tokens=250),

            # tts=sarvam.TTS(
            #     target_language_code="en-IN",
            #     model="bulbul:v3",
            #     speaker="shubh",
            # ),
            tts = openai.TTS(model="gpt-4o-mini-tts"),
            
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        user_query = new_message.text_content
        logger.info(f"User query: {user_query}")
        query_service = QueryService()
        results = query_service.search(user_query)
        logger.info(f"RAG context: {results}")

        rag_context = "\n\n".join([
                result.payload["text"]
                for result in results
            ])


        turn_ctx.add_message(
            role="system",
            content=f"""
Use this RAG context to answer the user.

RAG Context:
{rag_context}
""",
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and ask how you can help."
        )


async def entrypoint(ctx: JobContext):
    logger.info(f"User connected to room: {ctx.room.name}")

    # Sarvam docs recommend no VAD here.
    # Sarvam STT handles speech start/end internally.
    session = AgentSession(
        turn_detection="stt",
        min_endpointing_delay=0.07,
    )

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
    )


