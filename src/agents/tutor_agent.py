"""Tutor agent with simple question-aware responses."""


class TutorAgent:
    """Generates tutoring responses for the student."""

    def __init__(self) -> None:
        self.name = "TutorAgent"

    def generate(self, query: str, profile, strategy: str = "guided_learning") -> str:
        text = query.lower()

        if "photosynthesis" in text:
            return (
                f"Photosynthesis is how plants make their own food. "
                f"They use sunlight, water, and carbon dioxide to make glucose (food) and oxygen. "
                f"This mainly happens in the leaves. Would you like a tiny example using a plant?"
            )

        if "plant" in text:
            return f"Plants need sunlight, water, and air to grow. They make their own food in their leaves."

        return (
            f"Sure, {profile.name}! I can help with that. "
            f"Let’s explain it in a simple way and check your understanding after."
        )
