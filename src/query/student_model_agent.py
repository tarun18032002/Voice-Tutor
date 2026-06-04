"""Student Model Agent — maintains and updates the student's learning profile."""

from __future__ import annotations

import os

from openai import OpenAI

from src.agents.schemas import AssessmentOutput, StudentProfile


class StudentModelAgent:
    """
    Tracks the student's knowledge state and adapts their profile over time.

    Responsibilities:
    - Increment/decrement difficulty based on assessment results.
    - Suggest a learning mode switch if the current one isn't working.
    - Keep a lightweight log of mastered vs struggling topics.
    """

    DIFFICULTY_LADDER = ["easy", "medium", "hard"]
    MODE_ROTATION = ["discovery", "hint", "story", "quiz"]

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self._mastered_topics: list[str] = []
        self._struggling_topics: list[str] = []

    def update(
        self,
        profile: StudentProfile,
        assessment: AssessmentOutput,
        topic: str,
    ) -> StudentProfile:
        """
        Return an (optionally updated) StudentProfile based on the latest assessment.
        """
        updated = profile.model_copy(deep=True)

        if assessment.understood and assessment.confidence_score >= 0.75:
            updated = self._level_up(updated, topic)
        elif not assessment.understood or assessment.confidence_score < 0.4:
            updated = self._level_down(updated, topic)
            updated = self._suggest_mode_switch(updated, assessment)

        return updated

    def get_mastered_topics(self) -> list[str]:
        return list(self._mastered_topics)

    def get_struggling_topics(self) -> list[str]:
        return list(self._struggling_topics)

    def _level_up(self, profile: StudentProfile, topic: str) -> StudentProfile:
        """Increase difficulty if not already at max."""
        if topic not in self._mastered_topics:
            self._mastered_topics.append(topic)

        current_idx = self.DIFFICULTY_LADDER.index(profile.difficulty_level)
        if current_idx < len(self.DIFFICULTY_LADDER) - 1:
            profile.difficulty_level = self.DIFFICULTY_LADDER[current_idx + 1]

        return profile

    def _level_down(self, profile: StudentProfile, topic: str) -> StudentProfile:
        """Decrease difficulty if not already at min."""
        if topic not in self._struggling_topics:
            self._struggling_topics.append(topic)

        current_idx = self.DIFFICULTY_LADDER.index(profile.difficulty_level)
        if current_idx > 0:
            profile.difficulty_level = self.DIFFICULTY_LADDER[current_idx - 1]

        return profile

    def _suggest_mode_switch(
        self, profile: StudentProfile, assessment: AssessmentOutput
    ) -> StudentProfile:
        """
        Rotate the learning mode to try a fresh approach when the child is confused.
        """
        current_idx = self.MODE_ROTATION.index(profile.learning_mode)
        next_idx = (current_idx + 1) % len(self.MODE_ROTATION)
        profile.learning_mode = self.MODE_ROTATION[next_idx]
        return profile
