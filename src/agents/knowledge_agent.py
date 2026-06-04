"""Knowledge retrieval agent with simple topic-aware fallback."""


class KnowledgeAgent:
    """Provides knowledge context for tutoring."""

    def __init__(self) -> None:
        self.name = "KnowledgeAgent"

    def retrieve(self, query: str) -> list[str]:
        text = query.lower()

        if "photosynthesis" in text:
            return [
                "Photosynthesis is the process plants use to make food using sunlight, water, and carbon dioxide.",
                "It happens mainly in leaves and produces glucose and oxygen.",
            ]

        if "plant" in text or "leaf" in text:
            return ["Plants make their own food using sunlight, water, and air."]

        return ["Use a simple explanation with one example and one follow-up question."]
