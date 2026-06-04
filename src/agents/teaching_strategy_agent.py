"""Teaching strategy agent with simple, question-aware fallback."""


class TeachingStrategyAgent:
    """Chooses how to explain or guide the student."""

    def __init__(self) -> None:
        self.name = "TeachingStrategyAgent"

    def choose_strategy(self, query: str) -> str:
        text = query.lower()
        if "photosynthesis" in text:
            return "explanation"
        if "why" in text or "what is" in text:
            return "guided_learning"
        return "guided_learning"
