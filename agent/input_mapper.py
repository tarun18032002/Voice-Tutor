from __future__ import annotations

from langchain_core.messages import HumanMessage
from agent.models import TutorAgentState


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def input_mapper_node(state: TutorAgentState) -> dict:
    latest_user_text = ""

    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            latest_user_text = _content_to_text(msg.content).strip()
            break

    if not latest_user_text:
        return {
            "StudentQuery": "Greet the student and ask what topic they want help with."
        }

    return {
        "StudentQuery": latest_user_text,
    }