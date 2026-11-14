"""API routes for AI-powered quiz generation, retrieval, submission, and grading."""
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Header, HTTPException, status
import logging

from ..models.quiz import (
    GenerateQuizRequest,
    QuizResponse,
    QuizSummary,
    QuizSubmitRequest,
    QuizSubmitResponse,
    QuizAttemptHistoryItem,
)
from ..services.quiz import (
    default_quiz_service,
    DocumentNotFoundError,
    EmptyDocumentError,
    QuizNotFoundError,
    QuizServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Quizzes"])


def get_current_user_id(
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
) -> UUID:
    """Extracts and validates user ID from X-User-ID header or query param."""
    raw_uid = x_user_id or user_id
    if not raw_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide X-User-ID header.",
        )
    try:
        return UUID(raw_uid)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format. Must be a valid UUID.",
        )


@router.post(
    "/documents/{document_id}/generate-quiz",
    response_model=QuizResponse,
    summary="Generate an AI-powered quiz from an uploaded document",
)
def generate_quiz_endpoint(
    document_id: UUID,
    request: GenerateQuizRequest = GenerateQuizRequest(),
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """
    Generates a quiz from the uploaded document text.
    - Configurable difficulty (easy, medium, hard).
    - Question formats: multiple-choice, true/false, short-answer.
    - Questions are strictly grounded in document content with source citations.
    - Returns student view (withholding answers and explanations before submission).
    """
    user_uuid = get_current_user_id(x_user_id, user_id)

    try:
        quiz_response = default_quiz_service.generate_quiz(
            document_id=document_id,
            user_id=user_uuid,
            request=request,
        )
        return quiz_response

    except DocumentNotFoundError as e:
        logger.warning(f"Document not found during quiz generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except EmptyDocumentError as e:
        logger.warning(f"Empty document for quiz generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except QuizServiceError as e:
        logger.error(f"Quiz service error for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate quiz: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during quiz generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during quiz generation.",
        )


@router.get(
    "/documents/{document_id}/quizzes",
    response_model=List[QuizSummary],
    summary="List all quizzes generated for a document",
)
def list_document_quizzes_endpoint(
    document_id: UUID,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """Lists previous quizzes generated for the specified document."""
    user_uuid = get_current_user_id(x_user_id, user_id)

    try:
        return default_quiz_service.list_quizzes(document_id=document_id, user_id=user_uuid)
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error listing quizzes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list document quizzes.",
        )


@router.get(
    "/quizzes/{quiz_id}",
    response_model=QuizResponse,
    summary="Get interactive quiz session (student view)",
)
def get_quiz_endpoint(
    quiz_id: UUID,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """
    Retrieves the interactive quiz session. Answers and explanations are withheld.
    """
    user_uuid = get_current_user_id(x_user_id, user_id)

    try:
        return default_quiz_service.get_quiz_student_view(quiz_id=quiz_id, user_id=user_uuid)
    except QuizNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error getting quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve quiz.",
        )


@router.post(
    "/quizzes/{quiz_id}/submit",
    response_model=QuizSubmitResponse,
    summary="Submit answers and get scored review with explanations",
)
def submit_quiz_endpoint(
    quiz_id: UUID,
    request: QuizSubmitRequest,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """
    Submits user answers for grading.
    Returns:
    - Overall score, total points, percentage, passed status.
    - Question-by-question breakdown with correctness, correct answers, explanations, and verbatim source excerpts.
    """
    user_uuid = get_current_user_id(x_user_id, user_id)

    try:
        return default_quiz_service.grade_quiz(
            quiz_id=quiz_id,
            user_id=user_uuid,
            request=request,
        )
    except QuizNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error submitting quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit and grade quiz.",
        )


@router.get(
    "/quizzes/{quiz_id}/attempts",
    response_model=List[QuizAttemptHistoryItem],
    summary="Get attempt history for a quiz",
)
def get_quiz_attempts_endpoint(
    quiz_id: UUID,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """Retrieves all past attempts for a given quiz."""
    user_uuid = get_current_user_id(x_user_id, user_id)

    try:
        return default_quiz_service.get_attempts(quiz_id=quiz_id, user_id=user_uuid)
    except QuizNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error getting quiz attempts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve quiz attempts.",
        )
