"""Student model agent stub."""

from src.agents.schemas import AssessmentOutput, StudentProfile


class StudentModelAgent:
    """Tracks student learning profile and needs."""

    DIFFICULTY_LADDER = ["easy", "medium", "hard"]
    MODE_ROTATION = ["discovery", "hint", "story", "quiz"]

    def __init__(self) -> None:
        self.name = "StudentModelAgent"

    def update(self, profile: StudentProfile, assessment: AssessmentOutput, topic: str) -> StudentProfile:
        updated = profile.model_copy(deep=True)

        if assessment.understood and assessment.confidence_score >= 0.75:
            current_idx = self.DIFFICULTY_LADDER.index(updated.difficulty_level)
            if current_idx < len(self.DIFFICULTY_LADDER) - 1:
                updated.difficulty_level = self.DIFFICULTY_LADDER[current_idx + 1]
            if topic not in updated.mastered_topics:
                updated.mastered_topics.append(topic)
        else:
            current_idx = self.DIFFICULTY_LADDER.index(updated.difficulty_level)
            if current_idx > 0:
                updated.difficulty_level = self.DIFFICULTY_LADDER[current_idx - 1]
            if topic not in updated.struggling_topics:
                updated.struggling_topics.append(topic)

            current_idx_mode = self.MODE_ROTATION.index(updated.learning_mode)
            updated.learning_mode = self.MODE_ROTATION[(current_idx_mode + 1) % len(self.MODE_ROTATION)]

        return updated
