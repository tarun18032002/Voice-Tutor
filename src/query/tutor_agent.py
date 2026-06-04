"""Tutor Agent — generates the final child-facing tutoring response."""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from src.agents.schemas import (
    StudentProfile,
    KnowledgeAgentOutput,
    TeachingStrategyOutput,
    TutorResponse,
)


STRATEGY_INSTRUCTIONS = {
    "socratic": (
        "Use the Socratic method. Ask probing questions to guide the child toward "
        "the answer themselves. NEVER state the answer directly."
    ),
    "story": (
        "Wrap the concept in a short, engaging story or narrative that a child "
        "of this age would enjoy. Make it vivid and relatable."
    ),
    "analogy": (
        "Explain using a simple, everyday analogy the child will instantly recognise. "
        "Make the abstract concrete."
    ),
    "hints": (
        "Give progressive hints — start very subtle and increase specificity only if "
        "the child asks for more help. Do NOT reveal the full answer."
    ),
    "quiz": (
        "Turn this into a fun quiz moment. Ask one multiple-choice or open question, "
        "then give encouraging feedback based on the child's answer."
    ),
}


class TutorAgent:
    """
    Generates tutoring responses for the student.
    Does NOT give direct answers. Instead it guides the child through
    the strategy chosen by the TeachingStrategyAgent.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = "TutorAgent"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        student_query: str,
        student_profile: StudentProfile,
        knowledge: Optional[KnowledgeAgentOutput],
        strategy: Optional[TeachingStrategyOutput],
        conversation_history: list[dict],
    ) -> TutorResponse:
        """Produce a child-facing tutoring message."""

        system_prompt = self._build_system_prompt(student_profile, strategy)
        messages = self._build_messages(student_query, knowledge, conversation_history)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )

        return self._parse_response(response.choices[0].message.content, strategy)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        profile: StudentProfile,
        strategy: Optional[TeachingStrategyOutput],
    ) -> str:
        strategy_name = strategy.strategy if strategy else "socratic"
        strategy_instruction = STRATEGY_INSTRUCTIONS.get(
            strategy_name, STRATEGY_INSTRUCTIONS["socratic"]
        )
        return (
            f"You are a warm, encouraging AI tutor for a {profile.age}-year-old "
            f"child in grade {profile.grade}.\n\n"
            f"TEACHING STRATEGY: {strategy_instruction}\n\n"
            "RULES:\n"
            "- Speak at an age-appropriate level. Use simple words.\n"
            "- Always be encouraging and patient.\n"
            "- End every response with either a question or a short understanding check.\n"
            "- NEVER give direct answers unprompted.\n"
        )

    def _build_messages(
        self,
        student_query: str,
        knowledge: Optional[KnowledgeAgentOutput],
        history: list[dict],
    ) -> list[dict]:
        messages: list[dict] = list(history)

        if knowledge and knowledge.raw_context.strip():
            context_note = (
                "[CONTEXT FOR TUTOR — do not quote verbatim, use to guide teaching]\n"
                + knowledge.raw_context[:1200]
            )
            messages.append({"role": "user", "content": context_note})
            messages.append({
                "role": "assistant",
                "content": "Understood. I'll use this context to guide my teaching.",
            })

        messages.append({"role": "user", "content": student_query})
        return messages

    def _parse_response(
        self, raw_text: str, strategy: Optional[TeachingStrategyOutput]
    ) -> TutorResponse:
        lines = raw_text.strip().splitlines()
        follow_up: Optional[str] = None

        for line in reversed(lines):
            line = line.strip()
            if line.endswith("?"):
                follow_up = line
                break

        return TutorResponse(
            message=raw_text.strip(),
            strategy_used=strategy.strategy if strategy else "socratic",
            follow_up_question=follow_up,
        )