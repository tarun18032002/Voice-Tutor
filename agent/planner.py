import os


from .models import TutorAgentState,PlannerConfig
from .llm import LLM_Client



_system_message = """
You are an planner for a tutoring system.
Your job is to make a plan according to the student query and retrieved info.
Your task is plan how to answer the student query based on the retrieved info.


Rules:
1. If the student query is straightforward and can be directly answered by retrieved info, return a plan to answer the student query.
2. If the student query is complex and requires multiple steps of reasoning, return a plan with multiple steps to answer the student query.


"""

def _build_context(state:TutorAgentState)->str:
    """
        Build the context for the planner node based on the current state of the tutor agent.
    """
    planner_context = f"StudentQuery: {state.StudentQuery}\n" 
    if state.RetrievedInfo:
        planner_context += f"Current Retrieved Info: {state.RetrievedInfo}\n"
    if state.WebInfo:
        planner_context += f"Current Web Info: {state.WebInfo}\n"
    return  planner_context

def planner_node(state: TutorAgentState) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    client = LLM_Client(api_key=api_key)
    response = client.generate_response(
        model="gpt-5-mini",
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content":  _system_message},
            {"role": "user", "content":  _build_context(state)},
        ],
        text_format=PlannerConfig,
    )
    
    print(f"Planner output: {response}")
    if not response.plan:
        raise ValueError("Invalid response from planner node. Missing 'plan'.")
    else:
        return {"plan": response.plan}

    
