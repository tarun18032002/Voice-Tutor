"""Teaching Strategy Agent — decides *how* to teach based on student profile."""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from src.agents.schemas import (
    StudentProfile,
    KnowledgeAgentOutput,
    TeachingStrategyOutput,
)


class TeachingStrategyAgent:
    """
    Chooses a teaching strategy (Socratic, Story, Analogy, Hints, Quiz)
    based on the child's age, difficulty level, topic type, and learning mode.
    """

    VALID_STRATEGIES = {"socratic", "story", "analogy", "hints", "quiz"}

    # Map the student's preferred learning_mode → default strategy
    MODE_TO_STRATEGY = {
        "discovery": "socratic",
        "hint":      "hints",
        "story":     "story",
        "quiz":      "quiz",
    }

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        student_query: str,
        student_profile: StudentProfile,
        knowledge: Optional[KnowledgeAgentOutput] = None,
    ) -> TeachingStrategyOutput:
        """Return the best teaching strategy for this interaction."""

        context_snippet = ""
        if knowledge:
            context_snippet = knowledge.raw_context[:800]  # keep prompt short

        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(
            student_query, student_profile, context_snippet
        )

        if self.client is not None:
            try:
                message = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                raw_text = message.choices[0].message.content or ""
                return self._parse_response(raw_text, student_profile)
            except Exception:
                pass

        fallback = self.MODE_TO_STRATEGY.get(student_profile.learning_mode, "socratic")
        return TeachingStrategyOutput(strategy=fallback, reasoning="Fallback to default strategy due to OpenAI availability.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert pedagogy advisor for a children's AI tutor. "
            "Given the student's profile and their question, choose the best teaching strategy. "
            "Respond ONLY with valid JSON:\n"
            '{"strategy": "<socratic|story|analogy|hints|quiz>", '
            '"reasoning": "<short explanation>", '
            '"suggested_questions": ["<q1>", "<q2>"]}'
        )

    def _build_user_message(
        self,
        query: str,
        profile: StudentProfile,
        context: str,
    ) -> str:
        return (
            f"Student: {profile.name}, Age: {profile.age}, Grade: {profile.grade}\n"
            f"Preferred learning mode: {profile.learning_mode}\n"
            f"Difficulty level: {profile.difficulty_level}\n"
            f"Query: {query}\n"
            f"Relevant knowledge snippet:\n{context}"
        )

    def _parse_response(
        self, raw: str, profile: StudentProfile
    ) -> TeachingStrategyOutput:
        try:
            data = json.loads(raw.strip())
            strategy = data.get("strategy", "").lower()
            if strategy not in self.VALID_STRATEGIES:
                strategy = self.MODE_TO_STRATEGY.get(profile.learning_mode, "socratic")
            return TeachingStrategyOutput(
                strategy=strategy,
                reasoning=data.get("reasoning", ""),
                suggested_questions=data.get("suggested_questions", []),
            )
        except json.JSONDecodeError:
            # Graceful fallback: use the student's preferred mode
            fallback = self.MODE_TO_STRATEGY.get(profile.learning_mode, "socratic")
            return TeachingStrategyOutput(
                strategy=fallback,
                reasoning="Fallback to default strategy due to parse error.",
            )