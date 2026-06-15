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

# Load environment variables
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

    initial_state = TutorAgentState(
        StudentQuery=query
    )

    result: TutorAgentState = graph.invoke(initial_state)

    answer = result['TutorAnswer'] or "I'm sorry, I couldn't generate an answer."

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
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--file", type=str, default="data/chapter-1(10th std).pdf")
    parser.add_argument("--query", type=str, default="")
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
        ingestion_job(args.file)
    elif args.query:
        cli_test(args.query)
    else:
        from livekit.agents import WorkerOptions, cli

        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint
            )
        )