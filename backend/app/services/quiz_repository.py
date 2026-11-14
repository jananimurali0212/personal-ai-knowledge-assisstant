"""Supabase repository for storing and managing quizzes, questions, and attempts."""
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone
import httpx

from ..config.settings import settings
from ..models.quiz import QuizQuestion

logger = logging.getLogger(__name__)

# In-memory storage caches for test environments and offline fallback
_in_memory_quizzes: Dict[str, Dict[str, Any]] = {}  # quiz_id -> quiz dict
_in_memory_questions: Dict[str, List[Dict[str, Any]]] = {}  # quiz_id -> list of questions
_in_memory_attempts: Dict[str, List[Dict[str, Any]]] = {}  # quiz_id -> list of attempts


class QuizRepositoryError(Exception):
    """Base exception for quiz storage operations."""
    pass


class QuizRepository:
    """
    Manages persistence of quizzes, questions, and attempts to Supabase.
    Enforces multi-tenant ownership via user_id.
    Provides robust in-memory fallback for testing and local dev.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.supabase_url = (
            supabase_url
            if supabase_url is not None
            else settings.SUPABASE_URL
        ).rstrip("/")
        self.supabase_key = (
            supabase_key
            if supabase_key is not None
            else (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY)
        )

    def is_configured(self) -> bool:
        """Checks if valid Supabase connection credentials are present."""
        return bool(
            self.supabase_url
            and self.supabase_key
            and self.supabase_key not in ("your_supabase_anon_key_here", "")
            and len(self.supabase_key) > 20
        )

    def create_quiz(
        self,
        document_id: UUID,
        user_id: UUID,
        title: str,
        difficulty: str,
        total_questions: int,
        question_types: List[str],
        quiz_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Creates a new quiz entry in the database."""
        qid = quiz_id or uuid4()
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": str(qid),
            "document_id": str(document_id),
            "user_id": str(user_id),
            "title": title,
            "difficulty": difficulty,
            "total_questions": total_questions,
            "question_types": question_types,
            "created_at": now,
            "updated_at": now,
        }

        # Cache locally
        _in_memory_quizzes[str(qid)] = dict(record)
        _in_memory_questions[str(qid)] = []
        _in_memory_attempts[str(qid)] = []

        if not self.is_configured():
            logger.info(f"Quiz {qid} stored in in-memory store (Supabase not configured).")
            return record

        try:
            endpoint = f"{self.supabase_url}/rest/v1/quizzes"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(endpoint, json=record, headers=headers)
                if res.status_code in (200, 201):
                    saved = res.json()
                    if saved and len(saved) > 0:
                        _in_memory_quizzes[str(qid)] = saved[0]
                        return saved[0]
                else:
                    logger.warning(
                        f"Failed to insert quiz to Supabase ({res.status_code}): {res.text}. Using local copy."
                    )
        except Exception as e:
            logger.warning(f"Error persisting quiz to Supabase: {e}. Using local copy.")

        return record

    def save_questions(
        self,
        quiz_id: UUID,
        questions: List[QuizQuestion],
    ) -> List[Dict[str, Any]]:
        """Saves generated questions associated with a quiz."""
        qid_str = str(quiz_id)
        records = []
        for q in questions:
            rec = {
                "id": str(q.id),
                "quiz_id": qid_str,
                "question_index": q.question_index,
                "question_type": q.question_type,
                "question_text": q.question_text,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "source_excerpt": q.source_excerpt,
                "page_number": q.page_number,
                "chunk_id": str(q.chunk_id) if q.chunk_id else None,
                "created_at": q.created_at.isoformat() if hasattr(q.created_at, "isoformat") else str(q.created_at),
            }
            records.append(rec)

        # Store in local memory
        _in_memory_questions[qid_str] = records

        if not self.is_configured():
            return records

        try:
            endpoint = f"{self.supabase_url}/rest/v1/quiz_questions"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }
            with httpx.Client(timeout=20.0) as client:
                res = client.post(endpoint, json=records, headers=headers)
                if res.status_code in (200, 201):
                    saved = res.json()
                    _in_memory_questions[qid_str] = saved
                    return saved
                else:
                    logger.warning(
                        f"Failed to insert quiz questions to Supabase ({res.status_code}): {res.text}."
                    )
        except Exception as e:
            logger.warning(f"Error persisting questions to Supabase: {e}.")

        return records

    def get_quiz(self, quiz_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieves quiz and all its questions, enforcing user ownership."""
        qid_str = str(quiz_id)
        uid_str = str(user_id)

        # In-memory lookup
        quiz = _in_memory_quizzes.get(qid_str)
        if quiz and str(quiz.get("user_id")) != uid_str:
            return None

        questions = _in_memory_questions.get(qid_str, [])

        if quiz and questions:
            return {**quiz, "questions": questions}

        if not self.is_configured():
            if quiz:
                return {**quiz, "questions": questions}
            return None

        # Fetch from Supabase
        try:
            endpoint = f"{self.supabase_url}/rest/v1/quizzes"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            params = {
                "id": f"eq.{qid_str}",
                "user_id": f"eq.{uid_str}",
                "select": "*",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code == 200:
                    rows = res.json()
                    if rows:
                        quiz = rows[0]
                        _in_memory_quizzes[qid_str] = quiz
                    else:
                        return None
                else:
                    return None

            # Fetch questions for this quiz
            q_endpoint = f"{self.supabase_url}/rest/v1/quiz_questions"
            q_params = {
                "quiz_id": f"eq.{qid_str}",
                "select": "*",
                "order": "question_index.asc",
            }
            with httpx.Client(timeout=10.0) as client:
                q_res = client.get(q_endpoint, headers=headers, params=q_params)
                if q_res.status_code == 200:
                    questions = q_res.json()
                    _in_memory_questions[qid_str] = questions

            return {**quiz, "questions": questions}
        except Exception as e:
            logger.warning(f"Error fetching quiz from Supabase: {e}")
            if quiz:
                return {**quiz, "questions": questions}
            return None

    def list_quizzes_by_document(
        self, document_id: UUID, user_id: UUID
    ) -> List[Dict[str, Any]]:
        """Lists all quizzes created for a specific document by user."""
        doc_str = str(document_id)
        uid_str = str(user_id)

        # Check in-memory store
        local_quizzes = [
            q for q in _in_memory_quizzes.values()
            if str(q.get("document_id")) == doc_str and str(q.get("user_id")) == uid_str
        ]

        if not self.is_configured():
            return sorted(local_quizzes, key=lambda x: x.get("created_at", ""), reverse=True)

        try:
            endpoint = f"{self.supabase_url}/rest/v1/quizzes"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            params = {
                "document_id": f"eq.{doc_str}",
                "user_id": f"eq.{uid_str}",
                "select": "*",
                "order": "created_at.desc",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code == 200:
                    remote = res.json()
                    # Sync local cache
                    for r in remote:
                        _in_memory_quizzes[r["id"]] = r
                    return remote
            return local_quizzes
        except Exception as e:
            logger.warning(f"Error querying quizzes from Supabase: {e}")
            return local_quizzes

    def save_quiz_attempt(
        self,
        quiz_id: UUID,
        user_id: UUID,
        user_answers: Dict[str, str],
        score: float,
        total_points: float,
        percentage: float,
        review_details: List[Dict[str, Any]],
        attempt_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Saves a student's completed quiz attempt and scores."""
        aid = attempt_id or uuid4()
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": str(aid),
            "quiz_id": str(quiz_id),
            "user_id": str(user_id),
            "user_answers": user_answers,
            "score": score,
            "total_points": total_points,
            "percentage": percentage,
            "review_details": review_details,
            "completed_at": now,
        }

        # Cache locally
        qid_str = str(quiz_id)
        if qid_str not in _in_memory_attempts:
            _in_memory_attempts[qid_str] = []
        _in_memory_attempts[qid_str].append(record)

        if not self.is_configured():
            return record

        try:
            endpoint = f"{self.supabase_url}/rest/v1/quiz_attempts"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(endpoint, json=record, headers=headers)
                if res.status_code in (200, 201):
                    saved = res.json()
                    if saved:
                        return saved[0]
                else:
                    logger.warning(
                        f"Failed to save quiz attempt in Supabase ({res.status_code}): {res.text}."
                    )
        except Exception as e:
            logger.warning(f"Error saving quiz attempt in Supabase: {e}.")

        return record

    def get_quiz_attempts(
        self, quiz_id: UUID, user_id: UUID
    ) -> List[Dict[str, Any]]:
        """Retrieves attempt history for a specific quiz, enforcing user ownership."""
        qid_str = str(quiz_id)
        uid_str = str(user_id)

        local_attempts = [
            a for a in _in_memory_attempts.get(qid_str, [])
            if str(a.get("user_id")) == uid_str
        ]

        if not self.is_configured():
            return sorted(local_attempts, key=lambda x: x.get("completed_at", ""), reverse=True)

        try:
            endpoint = f"{self.supabase_url}/rest/v1/quiz_attempts"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            params = {
                "quiz_id": f"eq.{qid_str}",
                "user_id": f"eq.{uid_str}",
                "select": "*",
                "order": "completed_at.desc",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code == 200:
                    remote = res.json()
                    _in_memory_attempts[qid_str] = remote
                    return remote
            return local_attempts
        except Exception as e:
            logger.warning(f"Error fetching attempts from Supabase: {e}")
            return local_attempts

    def delete_quiz(self, quiz_id: UUID, user_id: UUID) -> bool:
        """Deletes a quiz and all associated questions and attempts."""
        qid_str = str(quiz_id)
        uid_str = str(user_id)

        # In-memory remove
        quiz = _in_memory_quizzes.get(qid_str)
        if quiz and str(quiz.get("user_id")) != uid_str:
            return False

        _in_memory_quizzes.pop(qid_str, None)
        _in_memory_questions.pop(qid_str, None)
        _in_memory_attempts.pop(qid_str, None)

        if not self.is_configured():
            return True

        try:
            endpoint = f"{self.supabase_url}/rest/v1/quizzes"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            params = {
                "id": f"eq.{qid_str}",
                "user_id": f"eq.{uid_str}",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.delete(endpoint, headers=headers, params=params)
                return res.status_code in (200, 204)
        except Exception as e:
            logger.warning(f"Error deleting quiz in Supabase: {e}")
            return True


default_quiz_repository = QuizRepository()
