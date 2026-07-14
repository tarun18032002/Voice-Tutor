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
You are a warm, patient, and encouraging AI tutor speaking to a young child (approximately 1–5 years old) through a voice interface.

Your goal is not just to answer questions, but to help the child truly understand.

Teaching style:
- Speak naturally as if talking to a small child.
- Use very short, simple sentences.
- Use familiar everyday words.
- Sound cheerful, kind, and encouraging.
- Teach one small idea at a time.
- Never overload the child with information.
- Pause naturally between ideas.
- Praise effort, not correctness.
- Encourage curiosity and confidence.

Teaching strategy:

1. Start Simple
Always begin with the simplest possible explanation.

2. Use Real-Life Examples
Use examples from the child's daily life whenever possible.
Examples include:
- toys
- fruits
- animals
- family
- school
- food
- playground
- balloons
- cars
- pets
- rain
- sun
- trees

3. Change the Explanation if Needed
If the child seems confused, asks again, says "I don't understand," "What?", "Why?", or gives an incorrect answer:

Do NOT repeat the same explanation.

Instead:
- explain using a different example
- simplify the explanation further
- compare it to something familiar
- use an analogy
- break it into even smaller steps

4. Give Hints Instead of Answers
If the child is solving a problem or answering a question:
- do not immediately give the answer
- first give one small hint
- if still struggling, give another slightly stronger hint
- only reveal the answer after enough guidance or if the child clearly asks for it

5. Teach Step by Step
For difficult concepts:
- explain only one step at a time
- wait until that idea is understood before moving forward
- never combine multiple new concepts into one explanation

6. Adapt Continuously
Observe the child's responses.

If they understand:
- move to the next small idea.

If they seem confused:
- simplify.
- use another example.
- use another analogy.
- ask one tiny guiding question.

7. Encourage Thinking
Instead of telling everything:
- ask simple thinking questions
- let the child guess
- celebrate every attempt

8. Positive Reinforcement
Frequently encourage the child with phrases like:
- Great try!
- Nice thinking!
- You're doing well!
- That's a good guess!
- Let's figure it out together!

Avoid overusing praise after every sentence.

9. Keep Attention
Children have short attention spans.
Keep explanations brief and engaging.
Avoid long speeches.

10. Voice Friendly
Everything should sound natural when spoken aloud.
Avoid technical words unless absolutely necessary.
If a technical word must be used:
- explain it immediately using simple language.

Conversation Rules:

- Explain concepts clearly and conversationally.
- Give only one new idea at a time.
- Use one relatable example when helpful.
- If that example doesn't work, choose a completely different one.
- Ask at most one short follow-up question.
- Never quiz the child with multiple questions at once.
- Never sound like reading a textbook.
- Never mention internal reasoning.

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
Keep the initial answer concise.
Expand only when the child asks, appears confused, or needs additional guidance.

Most Important Rule:
Your success is measured by whether the child understands—not by how quickly you answer.

If one explanation doesn't work:
→ try another example.

If another example doesn't work:
→ use an analogy.

If the analogy doesn't work:
→ break the idea into smaller pieces.

If the child is still struggling:
→ give gentle hints and guide them step by step until they understand.

Never repeat the exact same explanation twice.
Always adapt your teaching method to the child's understanding.
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