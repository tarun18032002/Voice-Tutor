"""Schema definitions for agents and sessions."""

import copy
from dataclasses import dataclass, field, replace


@dataclass
class StudentProfile:
    student_id: str
    name: str
    age: int
    grade: int
    learning_mode: str = "discovery"
    difficulty_level: str = "medium"
    mastered_topics: list[str] = field(default_factory=list)
    struggling_topics: list[str] = field(default_factory=list)

    def model_copy(self, deep: bool = True) -> "StudentProfile":
        return copy.deepcopy(self) if deep else replace(self)


@dataclass
class AssessmentOutput:
    understood: bool
    confidence_score: float
    recommended_action: str
    rationale: str = ""


@dataclass
class OrchestratorInput:
    student_query: str
    student_profile: StudentProfile
    conversation_history: list[dict] = field(default_factory=list)


@dataclass
class TutorRequest:
    query: str
    class_level: int = 5
    student_name: str | None = None


@dataclass
class TutorResponse:
    message: str
    strategy_used: str = "guided_learning"
    confidence_score: float = 0.5
    metadata: dict = field(default_factory=dict)
