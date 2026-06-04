"""Student profile memory stub."""


class StudentProfile:
    """Stores persistent student profile data."""

    def __init__(self, name: str = "Student") -> None:
        self.name = name
        self.preferences: dict = {}
