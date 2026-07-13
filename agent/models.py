
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage

# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

class TutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int