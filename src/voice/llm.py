import logging
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import groq, sarvam

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful voice assistant.
                Be friendly, concise, and conversational.
                Speak naturally as if you're having a real conversation.
            """,

            # Sarvam STT
            stt=sarvam.STT(
                language="unknown",
                model="saaras:v3",
                mode="transcribe"
            ),

            # Groq Llama Model
            llm=groq.LLM(
                model="llama-3.3-70b-versatile"
            ),

            # Sarvam TTS
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="shubh"
            ),
        )

    async def on_enter(self):
        """Called when user joins"""
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    logger.info(f"User connected to room: {ctx.room.name}")

    session = AgentSession()

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )