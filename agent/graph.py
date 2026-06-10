from langgraph.graph import StateGraph, START, END

from .models import TutorAgentState, TutorInputState

from .orchestrator import orchestrator_node


# ── Build graph ───────────────────────────────────────────────────────────────

def build_tutor_graph() -> StateGraph:
    builder = StateGraph(TutorAgentState, input_schema=TutorInputState)

    # ── Register nodes (one per diagram block) ───────────────────────────────
    builder.add_node("retriver", retriver_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("web", web_node)