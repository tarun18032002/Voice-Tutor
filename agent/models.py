from pydantic import BaseModel, Field
from typing import Literal


class TutorAgentState(BaseModel):
    StudentQuery: str = Field(default="", description="The query from the student.")
    AgentAction: str = Field(default="", description="The action taken by the agent.")
    RetrievedInfo: str = Field(default="", description="The information retrieved from the retriever node.")
    WebInfo: str = Field(default="", description="The information retrieved from the web node.")
    Escalate: bool = Field(default=False, description="Whether to escalate to a human tutor.")
    



class OrchestratorConfig(BaseModel):
    route: Literal["retriever", "web", "answer"]
    reason: str
