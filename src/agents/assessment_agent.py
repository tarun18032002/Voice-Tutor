"""Assessment agent stub."""

from src.agents.schemas import AssessmentOutput


class AssessmentAgent:
    """Evaluates student progress and understanding."""

    def __init__(self) -> None:
        self.name = "AssessmentAgent"

    def run(self, student_reply: str, tutor_last_response, student_profile, conversation_history) -> AssessmentOutput:
        understood = len(student_reply.split()) >= 2
        confidence = 0.8 if understood else 0.35
        recommended_action = "continue" if understood else "simplify"
        return AssessmentOutput(
            understood=understood,
            confidence_score=confidence,
            recommended_action=recommended_action,
            rationale="Simple heuristic assessment for test mode.",
        )
