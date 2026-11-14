"""Unit and integration tests for AI-powered quiz generation, grading, and anti-cheating separation."""
import json
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.quiz import (
    GenerateQuizRequest,
    QuizSubmitRequest,
    QuizQuestion,
)
from app.services.chunk_repository import ChunkRepository
from app.services.quiz_repository import QuizRepository
from app.services.quiz import (
    QuizGenerationService,
    DocumentNotFoundError,
    EmptyDocumentError,
    QuizNotFoundError,
)
from app.services.processor import _processing_status_registry


@pytest.fixture
def test_user_id():
    return uuid4()


@pytest.fixture
def other_user_id():
    return uuid4()


@pytest.fixture
def test_document_id():
    return uuid4()


@pytest.fixture
def mock_chunk_repository(test_document_id, test_user_id):
    repo = ChunkRepository(supabase_url="", supabase_key="")
    chunks = [
        {
            "id": str(uuid4()),
            "document_id": str(test_document_id),
            "user_id": str(test_user_id),
            "chunk_index": 0,
            "content": "Polymorphism in Java allows methods to perform different tasks based on the object that invokes them. Runtime polymorphism is achieved via method overriding.",
            "page_number": 1,
        },
        {
            "id": str(uuid4()),
            "document_id": str(test_document_id),
            "user_id": str(test_user_id),
            "chunk_index": 1,
            "content": "Inheritance enables a subclass to acquire all properties and behaviors from a superclass using the 'extends' keyword. Java supports single class inheritance.",
            "page_number": 2,
        },
        {
            "id": str(uuid4()),
            "document_id": str(test_document_id),
            "user_id": str(test_user_id),
            "chunk_index": 2,
            "content": "Encapsulation restricts direct access to some of an object's components by keeping fields private and providing public getter and setter methods.",
            "page_number": 3,
        },
    ]
    repo.get_chunks_by_document = MagicMock(return_value=chunks)
    return repo


@pytest.fixture
def mock_quiz_repository():
    return QuizRepository(supabase_url="", supabase_key="")


@pytest.fixture
def registered_doc(test_document_id, test_user_id):
    _processing_status_registry[str(test_document_id)] = {
        "status": "ready",
        "user_id": str(test_user_id),
        "file_name": "java_oop_guide.pdf",
    }
    yield test_document_id
    _processing_status_registry.pop(str(test_document_id), None)


class TestQuizGenerationService:
    """Tests for the QuizGenerationService domain logic."""

    def test_generate_quiz_student_view_withholds_answers(
        self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository
    ):
        """Verify student view omits correct_answer, explanation, and source_excerpt to prevent cheating."""
        service = QuizGenerationService(
            api_key="mock_key",
            chunk_repository=mock_chunk_repository,
            quiz_repository=mock_quiz_repository,
        )

        req = GenerateQuizRequest(
            difficulty="medium",
            num_questions=3,
            question_types=["multiple_choice", "true_false", "short_answer"],
        )

        quiz_resp = service.generate_quiz(
            document_id=registered_doc,
            user_id=test_user_id,
            request=req,
        )

        assert quiz_resp.quiz.title is not None
        assert quiz_resp.quiz.difficulty == "medium"
        assert len(quiz_resp.questions) == 3

        # Anti-cheating verification: QuizQuestionStudentView must NOT have correct_answer or explanation
        for q in quiz_resp.questions:
            assert not hasattr(q, "correct_answer")
            assert not hasattr(q, "explanation")
            assert not hasattr(q, "source_excerpt")
            assert q.question_text
            assert q.question_type in ["multiple_choice", "true_false", "short_answer"]

    def test_configurable_difficulties(
        self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository
    ):
        """Verify easy, medium, and hard difficulty levels are properly configured."""
        service = QuizGenerationService(
            api_key="mock_key",
            chunk_repository=mock_chunk_repository,
            quiz_repository=mock_quiz_repository,
        )

        for diff in ["easy", "medium", "hard"]:
            req = GenerateQuizRequest(difficulty=diff, num_questions=2)
            quiz_resp = service.generate_quiz(
                document_id=registered_doc,
                user_id=test_user_id,
                request=req,
            )
            assert quiz_resp.quiz.difficulty == diff
            assert len(quiz_resp.questions) == 2

    def test_empty_document_raises_bad_request(
        self, registered_doc, test_user_id, mock_quiz_repository
    ):
        """Verify empty document without chunks raises EmptyDocumentError."""
        empty_chunk_repo = ChunkRepository(supabase_url="", supabase_key="")
        empty_chunk_repo.get_chunks_by_document = MagicMock(return_value=[])

        service = QuizGenerationService(
            api_key="mock_key",
            chunk_repository=empty_chunk_repo,
            quiz_repository=mock_quiz_repository,
        )

        with pytest.raises(EmptyDocumentError) as exc_info:
            service.generate_quiz(
                document_id=registered_doc,
                user_id=test_user_id,
                request=GenerateQuizRequest(),
            )
        assert "has no extracted text chunks" in str(exc_info.value)

    def test_user_isolation_document_access(
        self, registered_doc, other_user_id, mock_chunk_repository, mock_quiz_repository
    ):
        """User B cannot generate quiz from User A's document."""
        service = QuizGenerationService(
            api_key="mock_key",
            chunk_repository=mock_chunk_repository,
            quiz_repository=mock_quiz_repository,
        )

        with pytest.raises(DocumentNotFoundError):
            service.generate_quiz(
                document_id=registered_doc,
                user_id=other_user_id,
                request=GenerateQuizRequest(),
            )

    def test_grading_mc_tf_and_short_answer(
        self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository
    ):
        """Verify grading accuracy across multiple choice, true/false, and short answer questions."""
        service = QuizGenerationService(
            api_key="mock_key",
            chunk_repository=mock_chunk_repository,
            quiz_repository=mock_quiz_repository,
        )

        quiz_resp = service.generate_quiz(
            document_id=registered_doc,
            user_id=test_user_id,
            request=GenerateQuizRequest(
                num_questions=3,
                question_types=["multiple_choice", "true_false", "short_answer"],
            ),
        )

        quiz_id = quiz_resp.quiz.id
        # Access internal stored questions to construct valid user answers
        stored = mock_quiz_repository.get_quiz(quiz_id, test_user_id)
        assert stored is not None
        questions = stored["questions"]

        user_answers = {}
        for q in questions:
            qid = str(q["id"])
            q_type = q["question_type"]
            correct = q["correct_answer"]
            if q_type == "multiple_choice":
                # Supply exact correct option
                user_answers[qid] = correct
            elif q_type == "true_false":
                # Supply correct boolean
                user_answers[qid] = correct
            elif q_type == "short_answer":
                # Supply answer with high keyword overlap
                user_answers[qid] = correct

        submit_req = QuizSubmitRequest(answers=user_answers)
        result = service.grade_quiz(quiz_id=quiz_id, user_id=test_user_id, request=submit_req)

        assert result.score == 3.0
        assert result.total_points == 3.0
        assert result.percentage == 100.0
        assert result.passed is True
        assert len(result.questions) == 3

        # Grounding and explanation review checks
        for item in result.questions:
            assert item.is_correct is True
            assert item.correct_answer != ""
            assert item.explanation != ""
            assert item.source_excerpt != ""
            assert item.page_number is not None

    def test_grading_wrong_answers(
        self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository
    ):
        """Verify 0 score when completely incorrect answers are submitted."""
        service = QuizGenerationService(
            api_key="mock_key",
            chunk_repository=mock_chunk_repository,
            quiz_repository=mock_quiz_repository,
        )

        quiz_resp = service.generate_quiz(
            document_id=registered_doc,
            user_id=test_user_id,
            request=GenerateQuizRequest(
                num_questions=2,
                question_types=["true_false", "multiple_choice"],
            ),
        )

        quiz_id = quiz_resp.quiz.id
        stored = mock_quiz_repository.get_quiz(quiz_id, test_user_id)
        questions = stored["questions"]

        # Intentionally give opposite/wrong answers
        user_answers = {}
        for q in questions:
            qid = str(q["id"])
            if q["question_type"] == "true_false":
                user_answers[qid] = "False" if q["correct_answer"] == "True" else "True"
            else:
                user_answers[qid] = "Completely Incorrect Option Z"

        result = service.grade_quiz(
            quiz_id=quiz_id, user_id=test_user_id, request=QuizSubmitRequest(answers=user_answers)
        )

        assert result.score == 0.0
        assert result.percentage == 0.0
        assert result.passed is False
        for item in result.questions:
            assert item.is_correct is False


class TestQuizAPIEndpoints:
    """Direct API endpoint tests for quiz routes."""

    def test_api_generate_quiz_success(self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository):
        from app.routes.quizzes import generate_quiz_endpoint, default_quiz_service
        default_quiz_service.chunk_repository = mock_chunk_repository
        default_quiz_service.quiz_repository = mock_quiz_repository
        default_quiz_service.api_key = "mock_key"
        default_quiz_service._client = None

        payload = GenerateQuizRequest(
            difficulty="hard",
            num_questions=3,
            question_types=["multiple_choice", "true_false", "short_answer"],
            title="Java OOP Advanced Exam",
        )

        resp = generate_quiz_endpoint(
            document_id=registered_doc,
            request=payload,
            x_user_id=str(test_user_id),
        )

        assert resp.quiz.title == "Java OOP Advanced Exam"
        assert resp.quiz.difficulty == "hard"
        assert len(resp.questions) == 3

        # Verify no answers in student view response
        for q in resp.questions:
            assert not hasattr(q, "correct_answer")
            assert not hasattr(q, "explanation")
            assert not hasattr(q, "source_excerpt")

    def test_api_unauthorized_user_blocked(self, registered_doc):
        from app.routes.quizzes import generate_quiz_endpoint
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            generate_quiz_endpoint(
                document_id=registered_doc,
                request=GenerateQuizRequest(),
                x_user_id=None,
            )
        assert exc_info.value.status_code == 401

    def test_api_submit_quiz_and_review(self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository):
        from app.routes.quizzes import (
            generate_quiz_endpoint,
            get_quiz_endpoint,
            submit_quiz_endpoint,
            get_quiz_attempts_endpoint,
            default_quiz_service,
        )
        default_quiz_service.chunk_repository = mock_chunk_repository
        default_quiz_service.quiz_repository = mock_quiz_repository
        default_quiz_service.api_key = "mock_key"
        default_quiz_service._client = None

        # 1. Generate quiz
        gen_resp = generate_quiz_endpoint(
            document_id=registered_doc,
            request=GenerateQuizRequest(difficulty="medium", num_questions=2),
            x_user_id=str(test_user_id),
        )
        quiz_id = gen_resp.quiz.id
        q1_id = gen_resp.questions[0].id
        q2_id = gen_resp.questions[1].id

        # 2. Get student view
        get_resp = get_quiz_endpoint(quiz_id=quiz_id, x_user_id=str(test_user_id))
        assert len(get_resp.questions) == 2

        # 3. Submit answers
        sub_req = QuizSubmitRequest(answers={str(q1_id): "Sample Answer", str(q2_id): "True"})
        sub_resp = submit_quiz_endpoint(quiz_id=quiz_id, request=sub_req, x_user_id=str(test_user_id))
        assert sub_resp.score >= 0.0
        assert sub_resp.percentage >= 0.0
        assert len(sub_resp.questions) == 2

        # 4. Check attempts history
        att_resp = get_quiz_attempts_endpoint(quiz_id=quiz_id, x_user_id=str(test_user_id))
        assert len(att_resp) >= 1

    def test_api_list_document_quizzes(self, registered_doc, test_user_id, mock_chunk_repository, mock_quiz_repository):
        from app.routes.quizzes import list_document_quizzes_endpoint, default_quiz_service
        default_quiz_service.chunk_repository = mock_chunk_repository
        default_quiz_service.quiz_repository = mock_quiz_repository
        default_quiz_service.api_key = "mock_key"
        default_quiz_service._client = None

        list_resp = list_document_quizzes_endpoint(
            document_id=registered_doc,
            x_user_id=str(test_user_id),
        )
        assert isinstance(list_resp, list)

