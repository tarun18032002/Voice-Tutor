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

from dotenv import load_dotenv


from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio

from agent.retriver import retrieve_knowledge
from agent.websearch import web_search
from agent.models import TutorState
from agent.llm import model
from agent.tutor import tutor_node



load_dotenv(".env.local")

logger = logging.getLogger(__name__)




tools = [
    retrieve_knowledge,
    web_search,
]




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