"""
test_pipeline.py — text-based test for the full agent pipeline (no voice).

Usage:
    python test_pipeline.py
"""

from dotenv import load_dotenv
load_dotenv()

from src.query.knowledge_agent import KnowledgeAgent
from src.query.teaching_strategy_agent import TeachingStrategyAgent
from src.query.tutor_agent import TutorAgent
from src.query.orchestrator_agent import OrchestratorAgent
from src.query.assessment_agent import AssessmentAgent
from src.query.student_model_agent import StudentModelAgent
from src.query.schemas import OrchestratorInput, StudentProfile


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

knowledge_agent         = KnowledgeAgent()
teaching_strategy_agent = TeachingStrategyAgent()
tutor_agent             = TutorAgent()
orchestrator            = OrchestratorAgent(
                              knowledge_agent=knowledge_agent,
                              teaching_strategy_agent=teaching_strategy_agent,
                              tutor_agent=tutor_agent,
                          )
assessment_agent        = AssessmentAgent()
student_model_agent     = StudentModelAgent()


# ---------------------------------------------------------------------------
# Student profile — edit to test different students
# ---------------------------------------------------------------------------

student_profile = StudentProfile(
    student_id="test_001",
    name="Arjun",
    age=10,
    grade=5,
    learning_mode="discovery",
    difficulty_level="medium",
)

conversation_history: list[dict] = []


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Voice Tutor — Text Test Mode")
print(f"  Student: {student_profile.name}, Age {student_profile.age}, Grade {student_profile.grade}")
print("  Type 'quit' to exit")
print("=" * 60)

while True:
    student_query = input("\nYou: ").strip()

    if not student_query:
        continue
    if student_query.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break

    # 1. Run orchestrator → knowledge → strategy → tutor
    input_data = OrchestratorInput(
        student_query=student_query,
        student_profile=student_profile,
        conversation_history=conversation_history,
    )
    tutor_response = orchestrator.run(input_data)

    print(f"\nTutor [{tutor_response.strategy_used}]: {tutor_response.message}")

    # 2. Update history
    conversation_history.append({"role": "user",      "content": student_query})
    conversation_history.append({"role": "assistant", "content": tutor_response.message})

    # 3. Assess after at least one exchange
    if len(conversation_history) >= 4:
        assessment = assessment_agent.run(
            student_reply=student_query,
            tutor_last_response=tutor_response,
            student_profile=student_profile,
            conversation_history=conversation_history,
        )

        student_profile = student_model_agent.update(
            profile=student_profile,
            assessment=assessment,
            topic=student_query,
        )

        print(f"\n  [debug] understood={assessment.understood} | "
              f"confidence={assessment.confidence_score:.2f} | "
              f"action={assessment.recommended_action} | "
              f"difficulty={student_profile.difficulty_level} | "
              f"mode={student_profile.learning_mode}")