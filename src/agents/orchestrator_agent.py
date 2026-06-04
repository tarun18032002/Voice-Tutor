"""Main orchestrator agent stub."""

from src.agents.schemas import TutorResponse


class OrchestratorAgent:
    """Coordinates the teaching workflow."""

    def __init__(self, knowledge_agent=None, teaching_strategy_agent=None, tutor_agent=None) -> None:
        self.name = "OrchestratorAgent"
        self.knowledge_agent = knowledge_agent
        self.teaching_strategy_agent = teaching_strategy_agent
        self.tutor_agent = tutor_agent

    def run(self, input_data) -> TutorResponse:
        strategy = "guided_learning"
        if self.teaching_strategy_agent is not None and hasattr(self.teaching_strategy_agent, "choose_strategy"):
            strategy = self.teaching_strategy_agent.choose_strategy(input_data.student_query)

        message = input_data.student_query
        if self.tutor_agent is not None and hasattr(self.tutor_agent, "generate"):
            message = self.tutor_agent.generate(input_data.student_query, input_data.student_profile, strategy)

        return TutorResponse(message=message, strategy_used=strategy)
