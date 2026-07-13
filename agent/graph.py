"""
graph.py
--------
LangGraph graph assembly for the low-latency Tutor Agent.

Architecture
------------

START
  ↓
tutor
  ├── no tools required → END
  │
  └── tools
        ├── retrieve_knowledge
        ├── search_web
        └── create_learning_plan
              ↓
            tutor
              ↓
             END

Only the tutor node calls an LLM.
Retriever, planner, and web search are normal Python tools.
"""


from __future__ import annotations
 

import inspect
import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import START,END, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable 

from agent.models import TutorAgentState,InputTutorAgentState
