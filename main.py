from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import silero, openai

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tutor-agent")


# -----------------------------------------------------------------------------
# Graph Runner
# -----------------------------------------------------------------------------

def _run_graph(graph, query: str) -> str:
    from agent.models import TutorAgentState

    state = TutorAgentState(
        StudentQuery=query
    )

    result = graph.invoke(state)

    answer = result["TutorAnswer"]

    if not answer:
        answer = "I'm sorry, I couldn't generate an answer."

    logger.info("Graph finished. Answer length=%d", len(answer))

    return answer


# -----------------------------------------------------------------------------
# Tutor Agent
# -----------------------------------------------------------------------------

class TutorAgent(Agent):

    def __init__(self, graph):
        super().__init__(
            instructions=(
                "You are a friendly AI tutor. "
                "Explain concepts clearly and help students learn."
            )
        )

        self.graph = graph
        self.processing = False

    async def on_user_turn_completed(self, turn_ctx=None, new_message=None):

        if self.processing:
            logger.info("Already processing another request")
            return ""

        self.processing = True

        try:
            query = self._extract_query(new_message)

            if not query:
                return (
                    "Sorry, I didn't catch that. "
                    "Could you repeat your question?"
                )

            logger.info("Student: %s", query)

            loop = asyncio.get_running_loop()

            answer = await loop.run_in_executor(
                None,
                _run_graph,
                self.graph,
                query
            )

            logger.info("Tutor answer ready")

            return answer

        except Exception:
            logger.exception("Error while generating answer")

            return (
                "Sorry, something went wrong while processing your question."
            )

        finally:
            self.processing = False

    @staticmethod
    def _extract_query(new_message):

        if new_message is None:
            return ""

        content = getattr(new_message, "content", new_message)

        if isinstance(content, list):
            parts = []

            for item in content:

                if isinstance(item, str):
                    parts.append(item)

                elif hasattr(item, "text"):
                    parts.append(item.text)

                elif hasattr(item, "content"):
                    parts.append(str(item.content))

            return " ".join(parts).strip()

        return str(content).strip()


# -----------------------------------------------------------------------------
# LiveKit Entry Point
# -----------------------------------------------------------------------------

async def entrypoint(ctx: JobContext):

    from agent.graph import get_tutor_graph

    graph = get_tutor_graph()

    await ctx.connect()

    logger.info("Connected to room: %s", ctx.room.name)

    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.10,
            min_silence_duration=0.35,
            activation_threshold=0.50,
            prefix_padding_duration=0.15,
        ),

        # OpenAI Speech-to-Text
        stt=openai.STT(
            model="gpt-4o-transcribe",
        ),

        # OpenAI Text-to-Speech
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="alloy",
        ),

        # llm=openai.LLM(
        #     model="gpt-5.4-mini",
        # ),

        allow_interruptions=True,
        min_interruption_duration=0.30,
        min_interruption_words=2,
        min_endpointing_delay=0.40,
        max_endpointing_delay=3.0,
        min_consecutive_speech_delay=0.0,
        discard_audio_if_uninterruptible=True,
    )

    agent = TutorAgent(graph)

    _handled = set()

    def _handle_final_transcript(user_text: str):

        user_text = (user_text or "").strip()

        if not user_text:
            return

        if user_text in _handled:
            return

        _handled.add(user_text)

        if len(_handled) > 50:
            _handled.clear()

        logger.info("USER: %s", user_text)

    def _on_user_input_transcribed(ev):

        is_final = getattr(ev, "is_final", True)

        user_text = (
            getattr(ev, "transcript", "")
            or getattr(ev, "text", "")
        )

        logger.info(
            "[stt] transcribed | final=%s text=%r",
            is_final,
            user_text,
        )

        if is_final:
            _handle_final_transcript(user_text)

    def _on_speech_committed(ev):

        user_text = (
            getattr(ev, "transcript", "")
            or getattr(ev, "text", "")
            or str(ev)
        )

        logger.info("[stt] committed | text=%r", user_text)

        _handle_final_transcript(user_text)

    session.on(
        "user_input_transcribed",
        _on_user_input_transcribed
    )

    session.on(
        "user_speech_committed",
        _on_speech_committed
    )

    await session.start(
        room=ctx.room,
        agent=agent
    )

    await asyncio.Event().wait()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )