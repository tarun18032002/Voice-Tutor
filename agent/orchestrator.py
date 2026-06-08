from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI

from .models import OrchestratorConfig,TutorAgentState
import os



_system_message = """
You are an orchestrator for a tutoring system.

Your job is to decide the next route.

Available routes:
- "retriever": use this when retrieved info is missing, incomplete, or more course/local context is needed.
- "web": use this only when the student query requires fresh/current/external information or retrieved info is unrelated.
- "answer": use this when retrieved info is enough to answer the student.

Rules:
1. Prefer "answer" if Current Retrieved Info directly answers the StudentQuery.
2. Prefer "retriever" if RetrievedInfo is empty or partially relevant.
3. Prefer "web" only for current/latest facts, internet-only facts, or when retrieved info is unrelated.
4. Return only the structured output schema.
"""

def _build_context(state:TutorAgentState)->str:
    """
        Build the context for the orchestrator node based on the current state of the tutor agent.
    """
    orchestrator_context = f"StudentQuery: {state.StudentQuery}\n" 
    if state.RetrievedInfo:
        orchestrator_context += f"Current Retrieved Info: {state.RetrievedInfo}\n"
    if state.WebInfo:
        orchestrator_context += f"Current Web Info: {state.WebInfo}\n"
    return  orchestrator_context

def orchestrator_node(state: TutorAgentState) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    print("OpenAI api key found.")

    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model="gpt-5-mini",
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": _system_message},
            {"role": "user", "content": _build_context(state)},
        ],
        text_format=OrchestratorConfig,
    )

    return response.output_parsed.route

