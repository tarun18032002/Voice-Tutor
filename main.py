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

# Load environment variables
load_dotenv()

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

    initial_state = TutorAgentState(
        StudentQuery=query
    )

    result: TutorAgentState = graph.invoke(initial_state)

    answer = result.TutorAnswer or "I'm sorry, I couldn't generate an answer."

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
# LiveKit Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def entrypoint(job_ctx):
    """
    LiveKit text sandbox entrypoint.
    """

    import asyncio

    from livekit import rtc
    from agent.graph import get_tutor_graph

    graph = get_tutor_graph()

    # Connect to room
    await job_ctx.connect()

    logger.info("Connected to room: %s", job_ctx.room.name)

    # Create ChatManager
    chat = rtc.ChatManager(job_ctx.room)

    @chat.on("message_received")
    async def on_message(msg: rtc.ChatMessage):
        try:
            query = msg.message.strip()

            logger.info("Received query: %s", query)

            answer = _run_graph(graph, query)

            logger.info("Answer:\n%s", answer)

            await chat.send_message(answer)

        except Exception:
            logger.exception("Error while processing query.")

            await chat.send_message(
                "Sorry, something went wrong while processing your question."
            )

    # Keep the agent alive
    await asyncio.Event().wait()


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
        "TAVILY_API_KEY",
    ):
        status = "✓" if os.getenv(var) else "✗ MISSING"
        print(f"{var} = {status}")

    if args.ingest:
        ingestion_job()

    elif args.query:
        cli_test(args.query)

    else:
        from livekit.agents import WorkerOptions, cli

        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint
            )
        )