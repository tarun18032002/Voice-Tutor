"""
models.py
---------
All Pydantic state schemas and LLM structured-output configs for the
Tutor Agent graph.
"""

from __future__ import annotations

from typing import Annotated, Literal,Optional
from pydantic import BaseModel, Field
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# ── LangGraph state ───────────────────────────────────────────────────────────

class TutorInputState(BaseModel):
    """Public input surface: only what the caller provides."""
    StudentQuery: str = Field(..., description="The raw query from the student.")


class TutorAgentState_test(BaseModel):
    """Full internal state threaded through every node."""

    # ── Input ──────────────────────────────────────────────────────────────
    # messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    StudentQuery: str = Field(default="", description="The query from the student.")

    # ── Routing / orchestration ────────────────────────────────────────────
    AgentAction: str = Field(default="", description="Last routing decision made by the orchestrator.")
    OrchestratorReason: str = Field(default="", description="Reason for the routing decision.")
    
    TeachingMode: str = Field(default="story")

    # ── Knowledge sources ─────────────────────────────────────────────────
    RetrievedInfo: str = Field(default="", description="Information from the RAG retriever.")
    WebInfo: str = Field(default="", description="Information from web search.")

    # ── Downstream processing ─────────────────────────────────────────────
    # SynthesisInfo: str = Field(default="", description="Synthesized knowledge from knowledge agent.")
    # PlanInfo: str = Field(default="", description="Step-by-step plan from the planner.")
    TutorAnswer: str = Field(default="", description="Final answer from the tutor node.")

    # ── Comprehension loop ────────────────────────────────────────────────
    ComprehensionPassed: bool = Field(default=False, description="Whether the student understood the answer.")
    ComprehensionFeedback: str = Field(default="", description="Feedback/follow-up from comprehension checker.")
    RetryCount: int = Field(default=0, description="Number of re-plan attempts so far.")

    # ── Escalation ────────────────────────────────────────────────────────
    Escalate: bool = Field(default=False, description="Escalate to a human tutor if True.")


# ── Structured-output schemas (LLM response shapes) ──────────────────────────

class OrchestratorConfig_test(BaseModel):
    route: Literal["web", "answer"] = Field(description="If retrieved info is sufficient to answer the student query, choose 'answer'.  If the student query requires fresh/current/external information or retrieved info is unrelated or need more information on the user query, choose 'web'.")
    reason: str


class KnowledgeAgentConfig(BaseModel):
    """Synthesised knowledge from all available sources."""
    synthesis: str = Field(
        description=(
            "A coherent synthesis of retrieved info and/or web info that "
            "directly addresses the student query. Preserve all key facts."
        )
    )


class PlannerConfig(BaseModel):
    """Answer plan produced by the planner."""
    plan: str = Field(
        description=(
            "A step-by-step plan for answering the student query. "
            "For simple queries one step is fine; for complex ones list explicit steps."
        )
    )


class TutorConfig(BaseModel):
    """Final answer produced by the tutor."""
    answer: str = Field(
        description=(
            "The answer to the student query following the plan exactly. "
            "If the plan cannot be satisfied, state that clearly."
        )
    )


class ComprehensionConfig(BaseModel):
    """Comprehension check result."""
    passed: bool = Field(
        description=(
            "True if the tutor's answer is clear, complete, and a student "
            "reading it would likely understand the concept. False otherwise."
        )
    )
    feedback: str = Field(
        description=(
            "If passed is False, describe specifically what is unclear or "
            "missing. If passed is True, a short positive confirmation."
        )
    )









# -----------------
# new 
class InputTutorAgentState(BaseModel):
    """Public input surface for the graph."""
    query: str = Field(..., description="The raw query from the student.")


class TutorAgentState(BaseModel):
    query: str

    Retriver_info: list[str] = []
    Planner_info: str = ""
    planner_strategy: str = ""

    Orchestartor_info: str = ""
    orchestrator_action: str = ""

    search_queries: list[str] = []
    WebSearch_info: Optional[str] = None

    TutorAnswer: str = ""

    voice_status: str = ""


class OrchestratorConfig(BaseModel):
    reasoning: str = Field(
        description="Briefly explain what is missing, or why the current info is sufficient."
    )
    action: Literal["SEARCH", "ANSWER","TUTOR"]
    search_queries: list[str] = Field(
        default_factory=list,
        description="Only include if action is SEARCH. Provide 1-3 optimized search strings.",
    )
    final_response: Optional[str] = Field(
        default=None,
        description="Optional direct answer if action is ANSWER. Usually the Tutor Agent writes the final answer.",
    )
