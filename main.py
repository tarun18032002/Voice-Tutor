"""
main.py
--------
Entry point for the Tutor Agent service.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# IMPORTANT: LiveKit plugins must be imported on the main thread
from livekit.agents import Agent, AgentSession, WorkerOptions, cli, function_tool, RunContext
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables
load_dotenv(dotenv_path=".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingestion_job(file_path: str = "test/chapter-1(10th std).pdf") -> None:
    """
    Run the ingestion pipeline.
    """
    from injestion.injestion_pipeline import IngestionPipeline

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    logger.info("Starting ingestion...")
    pipeline = IngestionPipeline()
    pipeline.ingest(file_path)
    logger.info("Ingestion completed.")


# ──────────────────────────────────────────────────────────────────────────────
# Graph execution
# ──────────────────────────────────────────────────────────────────────────────

def _run_graph(graph, query: str) -> str:
    from agent.models import TutorAgentState

    initial_state = TutorAgentState(StudentQuery=query)

    result: TutorAgentState = graph.invoke(initial_state)
    print("result:", result)

    answer = result["TutorAnswer"] or "I'm sorry, I couldn't generate an answer."

    logger.info("Final answer generated.")

    return answer


# ──────────────────────────────────────────────────────────────────────────────
# CLI testing
# ──────────────────────────────────────────────────────────────────────────────

def cli_test(query: str) -> None:
    from agent.graph import get_tutor_graph

    graph = get_tutor_graph()
    answer = _run_graph(graph, query)

    print("\n" + "=" * 60)
    print("TUTOR ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# LiveKit Voice Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def entrypoint(job_ctx):
    """
    LiveKit voice agent entrypoint using LangGraph as the tutor brain.
    """

    print("=" * 60)
    print("ENTRYPOINT STARTED")
    print("=" * 60)

    from agent.graph import get_tutor_graph

    graph = get_tutor_graph()

    class LangGraphTutorAgent(Agent):
        def __init__(self):
            super().__init__(
                instructions=(
                    "You are a voice tutor agent. "
                    "For every student question, call the answer_student_question tool. "
                    "After the tool returns, speak the answer naturally and clearly. "
                    "Do not answer from your own knowledge unless the tool fails."
                )
            )

        @function_tool()
        async def answer_student_question(
            self,
            context: RunContext,
            question: str,
        ) -> str:
            """
            Answer a student's question using the LangGraph multi-agent tutor system.

            Args:
                question: The student's full question.
            """
            logger.info("LangGraph question: %s", question)

            try:
                return _run_graph(graph, question)
            except Exception:
                logger.exception("LangGraph failed.")
                return "Sorry, I had trouble generating the tutor answer."

    await job_ctx.connect()

    logger.info("Connected to room: %s", job_ctx.room.name)
    print("CONNECTED TO VOICE ROOM")

    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),

        # OpenAI Speech-to-Text
        stt=openai.STT(
            model="gpt-4o-transcribe",
        ),

        # OpenAI LLM used only to manage the voice agent/tool call
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),

        # OpenAI Text-to-Speech
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="alloy",
        ),
    )

    await session.start(
        room=job_ctx.room,
        agent=LangGraphTutorAgent(),
    )

    print("SESSION STARTED")

    await session.generate_reply(
        instructions="Greet the student and ask what they want to learn today."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tutor Agent")

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Quick CLI test query",
    )

    args, unknown = parser.parse_known_args()

    # verify env vars
    for var in (
        "OPENAI_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "TAVILY_API_KEY",
    ):
        status = "✓" if os.getenv(var) else "✗ MISSING"
        print(f"{var} = {status}")

    if args.ingest:
        ingestion_job()

    elif args.query:
        cli_test(args.query)

    else:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="voice-tutor",
            )
        )