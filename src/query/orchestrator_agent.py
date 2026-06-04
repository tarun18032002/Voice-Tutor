"""Orchestrator Agent — understands the query and decides what is needed."""

from __future__ import annotations

from src.agents.schemas import (
    OrchestratorInput,
    OrchestratorDecision,
    KnowledgeAgentOutput,
    TeachingStrategyOutput,
    TutorResponse,
)


CHITCHAT_TRIGGERS = {"hi", "hello", "hey", "thanks", "thank you", "bye", "ok", "okay", "good", "great"}


class OrchestratorAgent:
    """
    Central coordinator. Given a student query it:
      1. Calls the KnowledgeAgent to fetch relevant context.
      2. Calls the TeachingStrategyAgent to decide *how* to teach.
      3. Calls the TutorAgent to generate the final response.
    """

    def __init__(
        self,
        knowledge_agent,
        teaching_strategy_agent,
        tutor_agent,
    ) -> None:
        self.knowledge_agent = knowledge_agent
        self.teaching_strategy_agent = teaching_strategy_agent
        self.tutor_agent = tutor_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_data: OrchestratorInput) -> TutorResponse:
        """Full pipeline: knowledge → strategy → tutor response."""

        decision = self._decide(input_data)

        knowledge: KnowledgeAgentOutput | None = None
        if decision.fetch_knowledge:
            knowledge = self.knowledge_agent.run(
                query=input_data.student_query,
                student_profile=input_data.student_profile,
            )

        strategy: TeachingStrategyOutput | None = None
        if decision.choose_strategy:
            strategy = self.teaching_strategy_agent.run(
                student_query=input_data.student_query,
                student_profile=input_data.student_profile,
                knowledge=knowledge,
            )

        response = self.tutor_agent.run(
            student_query=input_data.student_query,
            student_profile=input_data.student_profile,
            knowledge=knowledge,
            strategy=strategy,
            conversation_history=input_data.conversation_history,
        )

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide(self, input_data: OrchestratorInput) -> OrchestratorDecision:
        """Rule-based routing — no LLM needed."""

        query_words = set(input_data.student_query.lower().split())
        is_chitchat = query_words.issubset(CHITCHAT_TRIGGERS)

        return OrchestratorDecision(
            fetch_knowledge=not is_chitchat,
            choose_strategy=not is_chitchat,
            student_query=input_data.student_query,
            student_profile=input_data.student_profile,
            conversation_history=input_data.conversation_history,
        )