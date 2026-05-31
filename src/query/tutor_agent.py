import json
import logging

from livekit.agents import JobContext
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero
from livekit.agents import stt

from .retriever import QueryService


logger = logging.getLogger("sarvam-voice-agent")
logger.setLevel(logging.INFO)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        self.query_service = QueryService()
        super().__init__(
            instructions="""
            You are a helpful, friendly voice assistant for children aged 8-14.
            Use the RAG context when it is relevant.
            Keep answers short, clear, and encouraging.
            Use simple words. Add a fun fact when possible!
            """,

            stt=stt.StreamAdapter(
                stt=openai.STT(),
                vad=silero.VAD.load()
            ),

            llm=openai.LLM(model="gpt-4o-mini"),

            tts=openai.TTS(model="gpt-4o-mini-tts"),
        )

    # ── Broadcast helpers (send text to frontend via DataChannel) ──────────
    async def _broadcast(self, room, payload: dict):
        """Send JSON to all participants (including the browser frontend)."""
        try:
            data = json.dumps(payload).encode()
            await room.local_participant.publish_data(data, reliable=True)
        except Exception as e:
            logger.warning(f"DataChannel broadcast failed: {e}")

    # ── Called when the user finishes speaking ─────────────────────────────
    async def on_user_turn_completed(self, turn_ctx, new_message):
        user_query = new_message.text_content
        logger.info(f"User query: {user_query}")

        # 📡 Send user transcript to browser
        await self._broadcast(self.session.room, {
            "type": "user_transcript",
            "text": user_query,
        })

        # RAG lookup
        results = self.query_service.search(user_query)
        logger.info(f"RAG context: {results}")

        rag_context = "\n\n".join([
            result.payload["text"]
            for result in results
        ])

        turn_ctx.add_message(
            role="system",
            content=f"Use this RAG context to answer the user.\n\nRAG Context:\n{rag_context}",
        )

        # 📡 Tell browser agent is thinking
        await self._broadcast(self.session.room, {"type": "agent_thinking"})

    # ── Called when the agent finishes its reply ───────────────────────────
    async def on_agent_turn_completed(self, turn_ctx, new_message):
        agent_reply = new_message.text_content
        logger.info(f"Agent reply: {agent_reply}")

        # 📡 Send agent reply text to browser
        await self._broadcast(self.session.room, {
            "type": "agent_response",
            "text": agent_reply,
        })

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the child warmly and ask what they would like to learn today."
        )


async def entrypoint(ctx: JobContext):
    logger.info(f"User connected to room: {ctx.room.name}")

    session = AgentSession()

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
    )