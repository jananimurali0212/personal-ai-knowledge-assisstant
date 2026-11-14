"""Pydantic domain models for AI-powered quiz generation, grading, and review."""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, ConfigDict

QuizDifficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal["multiple_choice", "true_false", "short_answer"]


class QuizQuestionBase(BaseModel):
    """Base schema for a quiz question."""
    question_index: int
    question_type: QuestionType
    question_text: str = Field(..., min_length=5, description="The test question text")
    options: List[str] = Field(
        default_factory=list,
        description="List of choices for multiple_choice and true_false questions",
    )


class QuizQuestionStudentView(QuizQuestionBase):
    """
    Public student view of a quiz question presented to user prior to submission.
    Omits correct answers, explanations, and source quotes to prevent cheating.
    """
    id: UUID
    quiz_id: UUID
    page_number: Optional[int] = None


class QuizQuestion(QuizQuestionBase):
    """Full question entity containing ground-truth answers, explanations, and source citations."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    quiz_id: UUID
    correct_answer: str = Field(..., description="The verified factual correct answer")
    explanation: str = Field(..., description="Explanation grounded in the source text")
    source_excerpt: str = Field(..., description="Verbatim text quote from the document supporting the answer")
    page_number: Optional[int] = None
    chunk_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: List[str], info) -> List[str]:
        q_type = info.data.get("question_type")
        if q_type == "multiple_choice" and len(v) < 2:
            raise ValueError("Multiple-choice questions must have at least 2 options.")
        if q_type == "true_false":
            # Normalize true/false options if missing
            if not v:
                return ["True", "False"]
        return v


class GenerateQuizRequest(BaseModel):
    """Request payload to generate a quiz from an uploaded document."""
    difficulty: QuizDifficulty = Field(
        default="medium",
        description="'easy' (recall/definitions), 'medium' (application/concepts), 'hard' (analysis/edge cases)",
    )
    num_questions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate (1 to 20)",
    )
    question_types: List[QuestionType] = Field(
        default=["multiple_choice", "true_false", "short_answer"],
        description="Allowed question types in the quiz",
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional custom quiz title",
    )


class QuizSummary(BaseModel):
    """Metadata summary of a generated quiz."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    document_name: str
    user_id: UUID
    title: str
    difficulty: QuizDifficulty
    total_questions: int
    question_types: List[QuestionType]
    created_at: datetime


class QuizResponse(BaseModel):
    """Interactive quiz session payload delivered to frontend for taking the quiz."""
    quiz: QuizSummary
    questions: List[QuizQuestionStudentView]


class QuizSubmitRequest(BaseModel):
    """Payload submitted by user containing their answers."""
    answers: Dict[str, str] = Field(
        ...,
        description="Map of question_id (UUID string) to the user's submitted answer text/option",
    )


class QuestionReviewItem(BaseModel):
    """Detailed review item for a submitted question including grading, answer, and quote."""
    question_id: UUID
    question_index: int
    question_type: QuestionType
    question_text: str
    options: List[str]
    user_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str
    source_excerpt: str
    page_number: Optional[int] = None


class QuizSubmitResponse(BaseModel):
    """Scoring and feedback results delivered after quiz submission."""
    attempt_id: UUID
    quiz_id: UUID
    score: float
    total_points: float
    percentage: float
    passed: bool
    completed_at: datetime
    questions: List[QuestionReviewItem]


class QuizAttemptHistoryItem(BaseModel):
    """Historical quiz attempt summary."""
    id: UUID
    quiz_id: UUID
    score: float
    total_points: float
    percentage: float
    completed_at: datetime
    review_details: List[QuestionReviewItem]
