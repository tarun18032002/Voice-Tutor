"""
main.py — Tutor Agent entry point.
livekit-agents 1.6.0: Agent subclass with on_user_turn_completed override.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero


# ──────────────────────────────────────────────────────────────────────────────
# Graph runner (blocking — called via run_in_executor)
# ──────────────────────────────────────────────────────────────────────────────

def _run_graph(graph, query: str) -> str:
    from agent.models import TutorAgentState

    initial_state = TutorAgentState(StudentQuery=query)
    result = graph.invoke(initial_state)

    if isinstance(result, dict):
        answer = result.get("TutorAnswer") or "I'm sorry, I couldn't generate an answer."
    else:
        answer = result.TutorAnswer or "I'm sorry, I couldn't generate an answer."

    logger.info("Graph finished. Answer length: %d chars", len(answer))
    return answer


# ──────────────────────────────────────────────────────────────────────────────
# Tutor Agent — subclass of Agent (correct 1.6.0 pattern)
# ──────────────────────────────────────────────────────────────────────────────



class TutorAgent(Agent):
    """
    Overrides on_user_turn_completed to intercept STT transcript,
    run the LangGraph tutor pipeline, and speak the answer via TTS.
    """

    def __init__(self, graph) -> None:
        super().__init__(
            instructions=(
                "You are a friendly AI tutor. "
                "Listen to the student's question and answer clearly and helpfully."
            )
        )
        self._graph = graph
        self.processing = False

    async def on_user_turn_completed(self, turn_ctx=None, new_message=None) -> str:
        # Prevent overlapping requests
        if self.processing:
            logger.info("Already processing another question. Ignoring new turn.")
            return ""

        self.processing = True

        try:
            query = self._extract_query(new_message)

            if not query:
                return (
                    "Sorry, I didn't catch that. "
                    "Could you please repeat your question?"
                )

            logger.info("Student said: %s", query)

            loop = asyncio.get_running_loop()

            answer = await loop.run_in_executor(
                None,
                _run_graph,
                self._graph,
                query,
            )

            logger.info("Tutor answer ready (%d chars)", len(answer))

            return ""

        except Exception:
            logger.exception("Error while processing student query.")
            return (
                "Sorry, something went wrong while generating the answer. "
                "Please try again."
            )

        finally:
            self.processing = False

    @staticmethod
    def _extract_query(new_message) -> str:
        if new_message is None:
            return ""

        content = new_message
        if hasattr(new_message, "content"):
            content = new_message.content

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

        return str(content).strip() if content else ""




# ──────────────────────────────────────────────────────────────────────────────
# LiveKit entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    from agent.graph import get_tutor_graph

    graph = get_tutor_graph()
    await ctx.connect()
    logger.info("Connected to room: %s", ctx.room.name)

    session = AgentSession(
        stt=lk_openai.STT(model="whisper-1"),
        tts=lk_openai.TTS(model="tts-1", voice="alloy"),
        vad=silero.VAD.load(),
    )
    await session.start(
        room=ctx.room,
        agent=TutorAgent(graph),
        
    )

    # await session.generate_reply(
    #     instructions="Greet the student warmly and ask what they'd like to learn today."
    # )

    await asyncio.Event().wait()


# ──────────────────────────────────────────────────────────────────────────────
# CLI quick-test (text only)
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
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingestion_job(file_path: str = "data/chapter-1(10th std).pdf") -> None:
    from injestion.ingestion_pipeline import IngestionPipeline

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    logger.info("Starting ingestion for: %s", file_path)
    pipeline = IngestionPipeline()
    pipeline.ingest(file_path)
    logger.info("Ingestion completed.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tutor Agent")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--file", type=str, default="data/chapter-1(10th std).pdf")
    parser.add_argument("--query", type=str, default="")
    args, unknown = parser.parse_known_args()

    for var in ("OPENAI_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY",
                "VECTOR_STORE_HOST", "VECTOR_STORE_API_KEY"):
        status = "✓" if os.getenv(var) else "✗ MISSING"
        print(f"{var} = {status}")

    if args.ingest:
        ingestion_job(args.file)
    elif args.query:
        cli_test(args.query)
    else:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))