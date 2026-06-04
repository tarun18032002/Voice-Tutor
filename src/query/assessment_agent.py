"""Assessment Agent — evaluates whether the child understood and decides what to do next."""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from src.agents.schemas import AssessmentOutput, StudentProfile, TutorResponse


class AssessmentAgent:
    """
    Analyses the child's last reply and the tutor's previous message to decide:
      - Did the child understand?
      - How confident are we (0 – 1)?
      - What areas are still confusing?
      - Should we proceed, re-teach with a different approach, or quiz?
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        student_reply: str,
        tutor_last_response: TutorResponse,
        student_profile: StudentProfile,
        conversation_history: Optional[list[dict]] = None,
    ) -> AssessmentOutput:
        """Return a structured assessment of the student's understanding."""

        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(
            student_reply, tutor_last_response, student_profile, conversation_history or []
        )

        if self.client is not None:
            try:
                message = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                raw_text = message.choices[0].message.content or ""
                return self._parse_response(raw_text)
            except Exception:
                pass

        return AssessmentOutput(
            understood=False,
            confidence_score=0.5,
            recommended_action="reteach",
            rationale="Fallback assessment due to OpenAI availability.",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert children's learning assessor. "
            "Given the tutor's last message and the child's reply, "
            "assess understanding and respond ONLY with valid JSON:\n"
            '{"understood": true/false, '
            '"confidence_score": 0.0-1.0, '
            '"confusion_areas": ["<area1>", "..."], '
            '"recommended_action": "<proceed|reteach|quiz>"}'
        )

    def _build_user_message(
        self,
        student_reply: str,
        tutor_response: TutorResponse,
        profile: StudentProfile,
        history: list[dict],
    ) -> str:
        recent_history = history[-4:] if len(history) > 4 else history
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent_history
        )
        return (
            f"Student: {profile.name}, Age: {profile.age}, Grade: {profile.grade}\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"Tutor's last message: {tutor_response.message}\n\n"
            f"Student's reply: {student_reply}"
        )

    def _parse_response(self, raw: str) -> AssessmentOutput:
        try:
            data = json.loads(raw.strip())
            action = data.get("recommended_action", "proceed")
            if action not in {"proceed", "reteach", "quiz"}:
                action = "proceed"
            return AssessmentOutput(
                understood=bool(data.get("understood", False)),
                confidence_score=float(data.get("confidence_score", 0.5)),
                confusion_areas=data.get("confusion_areas", []),
                recommended_action=action,
            )
        except (json.JSONDecodeError, ValueError):
            return AssessmentOutput(
                understood=False,
                confidence_score=0.5,
                recommended_action="reteach",
            )