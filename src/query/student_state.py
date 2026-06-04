"""Stores lightweight student state for tutoring sessions."""

from __future__ import annotations


class StudentState:
    """Represents session-level student context."""

    def __init__(self, class_level: int = 5) -> None:
        self.class_level = class_level
        self.history: list[str] = []

    def add_interaction(self, text: str) -> None:
        """Append an interaction to the session history."""
        self.history.append(text)

    def summary(self) -> dict:
        """Return a small summary of current student state."""
        return {
            "class_level": self.class_level,
            "interaction_count": len(self.history),
        }
