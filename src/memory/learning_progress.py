"""Learning progress memory stub."""


class LearningProgress:
    """Stores progress metrics for the student."""

    def __init__(self) -> None:
        self.topics_mastered: list[str] = []
        self.score = 0
