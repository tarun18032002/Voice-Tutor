import os


from .models import TutorAgentState,TutorConfig
from .llm import LLM_Client



_system_message = """
You are an tutor agent , Your job is to answer to the student query based on the plan.

Rules:
1. Follow the plan step by step to answer the student query.
2. If the plan is straightforward and can be directly answered, answer the student query directly.
3. If the plan is complex and requires multiple steps of reasoning, follow the plan step by step to answer the student query.   
4. If the plan is not helpful to answer the student query, DO NOT answer reply mention that you don't have the necessary information.


"""

def _build_context(state:TutorAgentState)->str:
    """
        Build the context for the planner node based on the current state of the tutor agent.
    """
    tutor_context = f"StudentQuery: {state.StudentQuery}\n" 
    if state.planInfo:
        tutor_context += f"Plan: {state.planInfo}\n"
    return  tutor_context

def tutor_node(state: TutorAgentState) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    client = LLM_Client(api_key=api_key)
    response = client.generate_response(
        model="gpt-5-mini",
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content":  _system_message},
            {"role": "user", "content":  _build_context(state)},
        ],
        text_format=TutorConfig,
    )
    
    print(f"Tutor output: {response}")
    if not response.answer:
        raise ValueError("Invalid response from tutor node. Missing 'answer'.")
    else:
        return {"plan": response.plan}

    
