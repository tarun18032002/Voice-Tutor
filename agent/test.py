"""
graph.py
--------
Single-LLM tutor graph for LiveKit.

Flow
----

START
  ↓
tutor
  ├── direct answer → END
  └── tool calls
        ├── retrieve_knowledge
        └── web_search
              ↓
            tutor
              ↓
             END
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import (
    AnyMessage,
    SystemMessage,
)
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
import asyncio

from agent.retriver import QdrantRetriever
from agent.websearch import WebSearchTool


load_dotenv(".env.local")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Initialize expensive resources once
# ---------------------------------------------------------------------

retriever_client = QdrantRetriever()
web_search_client = WebSearchTool()


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

@tool
async def retrieve_knowledge(query: str) -> str:
    """
    Search the tutor's private knowledge base.

    Use this tool when the student asks about uploaded documents,
    course material, textbook content, or private tutor information.

    Do not use it when the question can be answered from the existing
    conversation or from common general knowledge.
    """

    try:
        # If retrieve() is synchronous, run it in a worker thread so that
        # it does not block LiveKit's asyncio event loop.
        retrieved = await asyncio.to_thread(
            retriever_client.retrieve,
            query,
        )

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "retrieved_info": retrieved,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("Knowledge retrieval failed")

        return json.dumps(
            {
                "status": "error",
                "query": query,
                "message": str(exc),
            }
        )


@tool
async def web_search(query: str) -> str:
    """
    Search the public web.

    Use this tool only when the student's question requires current,
    recent, changing, or externally verified information.

    Do not use web search for stable general-knowledge questions.
    """

    try:
        # If search() is synchronous, keep it off the event loop.
        results = await asyncio.to_thread(
            web_search_client.search,
            query,
        )

        return json.dumps(
            {
                "status": "success",
                "query": query,
                "web_search_results": results,
            },
            default=str,
        )

    except Exception as exc:
        logger.exception("Web search failed")

        return json.dumps(
            {
                "status": "error",
                "query": query,
                "message": str(exc),
            }
        )


tools = [
    retrieve_knowledge,
    web_search,
]


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

class TutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


# ---------------------------------------------------------------------
# Tutor model
# ---------------------------------------------------------------------

model = init_chat_model(
    "gpt-5.4-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

model_with_tools = model.bind_tools(
    tools,
    parallel_tool_calls=True,
)


TUTOR_SYSTEM_PROMPT = """
You are a friendly AI tutor speaking to a student through a voice interface.

Teaching style:
- Explain concepts clearly and conversationally.
- Use short sentences that sound natural when spoken.
- Start with simple language.
- Give one relatable example when helpful.
- Avoid unnecessarily long lists.
- Ask at most one short follow-up question.

Tool rules:

1. Existing conversation
Use information already available in the conversation.
Do not retrieve the same information again for follow-up questions.

2. General knowledge
Answer stable general-knowledge questions directly.
For example, definitions of CO2 or photosynthesis do not require tools.

3. Private tutor content
Use retrieve_knowledge when the student refers to uploaded documents,
textbooks, lessons, chapters, notes, or private learning content.

4. Web search
Use web_search only for recent, current, changing, or externally verified
information.

5. Tool results
After receiving a tool result, explain the answer naturally.
Do not mention tool names, graph nodes, JSON, internal routing, or agents.

6. Voice output
Do not use markdown tables.
Do not read URLs aloud.
Keep the initial answer concise and expand only when the student asks.
"""


async def tutor_node(state: TutorState) -> dict:
    """The only node that invokes an LLM."""

    response = await model_with_tools.ainvoke(
        [
            SystemMessage(content=TUTOR_SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

def build_tutor_graph():
    builder = StateGraph(TutorState)

    builder.add_node("tutor", tutor_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tutor")

    builder.add_conditional_edges(
        "tutor",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "tutor")

    return builder.compile()


# Build one reusable graph.
tutor_graph = build_tutor_graph()


def get_tutor_graph():
    return tutor_graph