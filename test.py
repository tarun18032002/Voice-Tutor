"""
test.py — Interactive text-based test for the full agent pipeline (no voice).

Usage:
    python test.py

The script first asks the child for their name, age, and grade,
then derives sensible defaults for learning_mode and difficulty_level,
and starts the tutoring chat loop.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.teaching_strategy_agent import TeachingStrategyAgent
from src.agents.tutor_agent import TutorAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.assessment_agent import AssessmentAgent
from src.agents.student_model_agent import StudentModelAgent
from src.agents.schemas import OrchestratorInput, StudentProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_int(prompt: str, min_val: int, max_val: int) -> int:
    """Keep asking until the user enters a valid integer in [min_val, max_val]."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            value = int(raw)
            if min_val <= value <= max_val:
                return value
        print(f"  ⚠  Please enter a number between {min_val} and {max_val}.")


def _grade_to_difficulty(grade: int) -> str:
    """Map school grade to a starting difficulty level."""
    if grade <= 3:
        return "easy"
    elif grade <= 7:
        return "medium"
    else:
        return "hard"


def _grade_to_default_mode(grade: int) -> str:
    """Younger kids do better with stories; older with discovery."""
    if grade <= 3:
        return "story"
    elif grade <= 6:
        return "hint"
    else:
        return "discovery"


def _ask_learning_mode() -> str:
    """Let the child (or parent/teacher) pick a preferred learning style."""
    modes = {
        "1": ("discovery", "I like figuring things out myself 🔍"),
        "2": ("hint",      "I like getting little hints 💡"),
        "3": ("story",     "I love learning through stories 📖"),
        "4": ("quiz",      "I like quizzes and challenges 🏆"),
    }
    print("\n  How do you like to learn?")
    for key, (_, label) in modes.items():
        print(f"    [{key}] {label}")
    while True:
        choice = input("  Your choice (1-4, or press Enter to skip): ").strip()
        if choice == "":
            return ""          # caller will use grade-based default
        if choice in modes:
            return modes[choice][0]
        print("  ⚠  Please enter 1, 2, 3, or 4.")


# ---------------------------------------------------------------------------
# On-boarding: collect student profile interactively
# ---------------------------------------------------------------------------

def collect_student_profile() -> StudentProfile:
    print("\n" + "=" * 60)
    print("  Welcome to AI Tutor! 🎓")
    print("  Let's get to know you before we start.")
    print("=" * 60)

    # Name
    while True:
        name = input("\n  What's your name? ").strip()
        if name:
            break
        print("  ⚠  Name cannot be empty.")

    # Age
    age = _prompt_int(f"\n  Hi {name}! How old are you? (5–18): ", 5, 18)

    # Grade
    grade = _prompt_int(
        f"\n  Which grade / class are you in? (1–12): ", 1, 12
    )

    # Learning mode (optional)
    mode = _ask_learning_mode()
    if not mode:
        mode = _grade_to_default_mode(grade)
        print(f"  → Using default learning mode: '{mode}'")

    difficulty = _grade_to_difficulty(grade)

    student_id = f"{name.lower().replace(' ', '_')}_{age}_{grade}"

    profile = StudentProfile(
        student_id=student_id,
        name=name,
        age=age,
        grade=grade,
        learning_mode=mode,
        difficulty_level=difficulty,
    )

    print("\n" + "-" * 60)
    print(f"  ✅  Profile created!")
    print(f"      Name      : {profile.name}")
    print(f"      Age       : {profile.age}")
    print(f"      Grade     : {profile.grade}")
    print(f"      Mode      : {profile.learning_mode}")
    print(f"      Difficulty: {profile.difficulty_level}")
    print("-" * 60)

    return profile


# ---------------------------------------------------------------------------
# Build agent pipeline
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    student_profile = collect_student_profile()
    conversation_history: list[dict] = []

    print(f"\n  Ask me anything, {student_profile.name}! Type 'quit' to exit.\n")
    print("=" * 60)

    while True:
        try:
            student_query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not student_query:
            continue
        if student_query.lower() in {"quit", "exit", "bye"}:
            print(f"\nGoodbye, {student_profile.name}! Keep learning! 👋")
            break

        # ── 1. Orchestrator → knowledge → strategy → tutor ──────────────
        input_data = OrchestratorInput(
            student_query=student_query,
            student_profile=student_profile,
            conversation_history=conversation_history,
        )

        try:
            tutor_response = orchestrator.run(input_data)
        except Exception as exc:
            print(f"\n  ⚠  Pipeline error: {exc}")
            continue

        print(f"\nTutor [{tutor_response.strategy_used}]: {tutor_response.message}")

        # ── 2. Update conversation history ───────────────────────────────
        conversation_history.append({"role": "user",      "content": student_query})
        conversation_history.append({"role": "assistant", "content": tutor_response.message})

        # ── 3. Assess + update student model after first real exchange ───
        if len(conversation_history) >= 4:
            try:
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

                print(
                    f"\n  [debug] understood={assessment.understood} | "
                    f"confidence={assessment.confidence_score:.2f} | "
                    f"action={assessment.recommended_action} | "
                    f"difficulty={student_profile.difficulty_level} | "
                    f"mode={student_profile.learning_mode}"
                )
            except Exception as exc:
                print(f"\n  [debug] Assessment error (non-fatal): {exc}")


if __name__ == "__main__":
    main()