"""Conversation memory stub."""


class ConversationMemory:
    """Stores recent conversation history."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, text: str) -> None:
        self.messages.append(text)
