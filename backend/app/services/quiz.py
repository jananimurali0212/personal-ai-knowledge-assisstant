"""AI-Powered Quiz Generation and Grading Service.
Generates grounded quizzes from uploaded documents with configurable difficulty,
multiple question formats (multiple choice, true/false, short answer),
source citations, anti-cheating separation, and automated grading.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone
import httpx

from google import genai
from google.genai import types
from google.genai.errors import APIError

from ..config.settings import settings
from ..models.quiz import (
    QuizDifficulty,
    QuestionType,
    QuizQuestion,
    QuizQuestionStudentView,
    GenerateQuizRequest,
    QuizSummary,
    QuizResponse,
    QuizSubmitRequest,
    QuestionReviewItem,
    QuizSubmitResponse,
    QuizAttemptHistoryItem,
)
from .chunk_repository import ChunkRepository, default_chunk_repository
from .quiz_repository import QuizRepository, default_quiz_repository
from .processor import _processing_status_registry

logger = logging.getLogger(__name__)


class QuizServiceError(Exception):
    """Base exception for quiz service errors."""
    pass


class DocumentNotFoundError(QuizServiceError):
    """Raised when document does not exist or user lacks access."""
    pass


class EmptyDocumentError(QuizServiceError):
    """Raised when document has no chunks to generate questions from."""
    pass


class QuizNotFoundError(QuizServiceError):
    """Raised when requested quiz does not exist or user lacks access."""
    pass


QUIZ_GENERATION_SYSTEM_INSTRUCTION = """You are an expert educational assessment creator and professor.
Your mission is to generate a comprehensive, challenging, and strictly factual quiz from the provided document excerpts.

DIFFICULTY GUIDELINES:
- 'easy': Test direct factual recall, definitions, terminology, and clearly stated facts.
- 'medium': Test conceptual understanding, cause-and-effect relationships, and practical applications.
- 'hard': Test in-depth analysis, synthesis across ideas, subtle edge cases, and comparative reasoning.

STRICT GROUNDING RULES:
1. FACTUAL ACCURACY: Every question, answer, and explanation MUST be 100% grounded in the provided document excerpts.
2. ZERO HALLUCINATION: Never assume, extrapolate, or invent facts that are absent from the text.
3. SOURCE CITATIONS: For every question, provide the EXACT verbatim quote from the text in 'source_excerpt' that proves the answer.
4. QUESTION FORMATS:
   - 'multiple_choice': Exactly 4 distinct options. Exactly one correct answer.
   - 'true_false': Options must be ["True", "False"]. Correct answer must be either "True" or "False".
   - 'short_answer': Open-ended prompt requiring a 1-2 sentence response. Provide a complete reference answer in 'correct_answer'.

OUTPUT FORMAT:
Return a single valid JSON object matching this schema exactly, with NO markdown backticks or extra text:
{
  "questions": [
    {
      "question_index": 1,
      "question_type": "multiple_choice",
      "question_text": "What is the primary function of ...?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option B",
      "explanation": "According to the document, ...",
      "source_excerpt": "Direct sentence quote from the document text.",
      "page_number": 1
    }
  ]
}
"""


class QuizGenerationService:
    """
    Orchestrates quiz creation from uploaded documents:
    - Verifies document ownership.
    - Gathers document chunks.
    - Calls Gemini with structured instructions tailored to difficulty and question types.
    - Saves quiz and questions in the database.
    - Delivers student view (withholding answers) to prevent cheating.
    - Grades submitted attempts and generates detailed answer reviews.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        chunk_repository: Optional[ChunkRepository] = None,
        quiz_repository: Optional[QuizRepository] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = self._clean_model_name(
            model_name or settings.GEMINI_GENERATION_MODEL or "gemini-2.5-flash"
        )
        self.chunk_repository = chunk_repository or default_chunk_repository
        self.quiz_repository = quiz_repository or default_quiz_repository
        self._client: Optional[genai.Client] = None

    @staticmethod
    def _clean_model_name(name: str) -> str:
        if name.startswith("models/"):
            return name[len("models/"):]
        return name

    def _get_client(self) -> genai.Client:
        """Initializes or returns cached Google Gen AI Client."""
        if self._client is None:
            if not self.api_key or self.api_key in ("your_gemini_api_key_here", "mock_key") or self.api_key.startswith("mock") or len(self.api_key) < 20:
                raise QuizServiceError(
                    "GEMINI_API_KEY is not configured or mock key provided."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _get_document_metadata(self, document_id: UUID, user_id: UUID) -> Dict[str, Any]:
        """Validates document ownership and returns metadata."""
        key = str(document_id)
        if key in _processing_status_registry:
            entry = _processing_status_registry[key]
            if str(entry.get("user_id")) != str(user_id):
                raise DocumentNotFoundError(
                    f"Unauthorized: Document {document_id} does not belong to user {user_id}"
                )
            return {
                "id": str(document_id),
                "user_id": str(user_id),
                "file_name": entry.get("file_name", "document"),
                "status": entry.get("status", "ready"),
            }

        # Check Supabase documents table
        supabase_url = settings.SUPABASE_URL
        api_key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
        if supabase_url and api_key and api_key not in ("your_supabase_anon_key_here", ""):
            try:
                endpoint = f"{supabase_url.rstrip('/')}/rest/v1/documents"
                headers = {
                    "apikey": api_key,
                    "Authorization": f"Bearer {api_key}",
                }
                params = {
                    "id": f"eq.{document_id}",
                    "user_id": f"eq.{user_id}",
                    "select": "id,user_id,file_name,status",
                }
                with httpx.Client(timeout=10.0) as client:
                    res = client.get(endpoint, headers=headers, params=params)
                    if res.status_code == 200:
                        rows = res.json()
                        if rows and len(rows) > 0:
                            return rows[0]
            except Exception as e:
                logger.warning(f"Error checking document ownership in Supabase: {e}")

        raise DocumentNotFoundError(
            f"Document {document_id} not found or you do not have permission to access it."
        )

    def generate_quiz(
        self,
        document_id: UUID,
        user_id: UUID,
        request: GenerateQuizRequest,
    ) -> QuizResponse:
        """
        Generates a new quiz for the given document and user.
        Validates ownership, retrieves content, prompts Gemini, saves records,
        and returns the student view (no answers exposed).
        """
        # 1. Validate document ownership
        doc_meta = self._get_document_metadata(document_id, user_id)
        doc_name = doc_meta.get("file_name", "Document")

        # 2. Retrieve document chunks
        chunks = self.chunk_repository.get_chunks_by_document(document_id, user_id)
        if not chunks:
            raise EmptyDocumentError(
                f"Document '{doc_name}' has no extracted text chunks. Please ensure it is processed."
            )

        # 3. Determine title
        quiz_title = request.title or f"{doc_name} Quiz ({request.difficulty.capitalize()})"

        # 4. Generate questions via LLM
        quiz_id = uuid4()
        questions = self._generate_questions_with_gemini(
            quiz_id=quiz_id,
            chunks=chunks,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            question_types=request.question_types,
            doc_name=doc_name,
        )

        # 5. Persist quiz and questions to database
        saved_quiz = self.quiz_repository.create_quiz(
            document_id=document_id,
            user_id=user_id,
            title=quiz_title,
            difficulty=request.difficulty,
            total_questions=len(questions),
            question_types=request.question_types,
            quiz_id=quiz_id,
        )

        self.quiz_repository.save_questions(quiz_id=quiz_id, questions=questions)

        # 6. Build student view (withhold answers)
        student_questions = [
            QuizQuestionStudentView(
                id=q.id,
                quiz_id=q.quiz_id,
                question_index=q.question_index,
                question_type=q.question_type,
                question_text=q.question_text,
                options=q.options,
                page_number=q.page_number,
            )
            for q in questions
        ]

        summary = QuizSummary(
            id=quiz_id,
            document_id=document_id,
            document_name=doc_name,
            user_id=user_id,
            title=quiz_title,
            difficulty=request.difficulty,
            total_questions=len(questions),
            question_types=request.question_types,
            created_at=datetime.now(timezone.utc),
        )

        return QuizResponse(quiz=summary, questions=student_questions)

    def _generate_questions_with_gemini(
        self,
        quiz_id: UUID,
        chunks: List[Dict[str, Any]],
        difficulty: QuizDifficulty,
        num_questions: int,
        question_types: List[QuestionType],
        doc_name: str,
    ) -> List[QuizQuestion]:
        """Calls Gemini with strict schema enforcement to generate grounded questions."""
        # Compile document context with chunk IDs and page numbers
        context_parts = []
        for idx, chunk in enumerate(chunks[:25]):  # limit to top 25 chunks to avoid token blowout
            p_num = chunk.get("page_number", 1)
            c_id = chunk.get("id", f"chunk_{idx}")
            content = chunk.get("content", "").strip()
            context_parts.append(
                f"[Chunk {idx+1} | Page {p_num} | ID: {c_id}]\n{content}"
            )
        document_text = "\n\n".join(context_parts)

        types_str = ", ".join(question_types)
        prompt = f"""Generate exactly {num_questions} quiz questions based on the following document excerpts from "{doc_name}".

DIFFICULTY LEVEL: {difficulty}
ALLOWED QUESTION TYPES: {types_str}
TOTAL QUESTIONS REQUESTED: {num_questions}

Distribute the questions evenly among the allowed question types: {types_str}.
Ensure each question has an exact factual source excerpt from the excerpts below.

DOCUMENT EXCERPTS:
\"\"\"
{document_text}
\"\"\"

Produce valid JSON matching the required schema with all {num_questions} questions.
"""
        raw_text = ""
        try:
            client = self._get_client()
            config = types.GenerateContentConfig(
                system_instruction=QUIZ_GENERATION_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or ""
        except Exception as e:
            logger.warning(f"Gemini API call failed for quiz generation: {e}. Falling back to rule-based generation.")
            # Fallback generation for testing or offline
            return self._generate_fallback_questions(
                quiz_id=quiz_id,
                chunks=chunks,
                difficulty=difficulty,
                num_questions=num_questions,
                question_types=question_types,
            )

        parsed_questions = self._parse_and_validate_questions_json(
            raw_text=raw_text,
            quiz_id=quiz_id,
            chunks=chunks,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
        )

        return parsed_questions

    def _parse_and_validate_questions_json(
        self,
        raw_text: str,
        quiz_id: UUID,
        chunks: List[Dict[str, Any]],
        num_questions: int,
        question_types: List[QuestionType],
        difficulty: QuizDifficulty,
    ) -> List[QuizQuestion]:
        """Parses LLM JSON and converts into validated QuizQuestion models."""
        cleaned = raw_text.strip()
        # Remove markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        data = {}
        try:
            data = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON for quiz: {e}. Raw: {cleaned[:200]}")
            return self._generate_fallback_questions(
                quiz_id=quiz_id,
                chunks=chunks,
                difficulty=difficulty,
                num_questions=num_questions,
                question_types=question_types,
            )

        items = data.get("questions", [])
        if not items or not isinstance(items, list):
            logger.warning("No 'questions' list found in JSON.")
            return self._generate_fallback_questions(
                quiz_id=quiz_id,
                chunks=chunks,
                difficulty=difficulty,
                num_questions=num_questions,
                question_types=question_types,
            )

        results: List[QuizQuestion] = []
        for idx, item in enumerate(items):
            try:
                q_type = item.get("question_type", "multiple_choice")
                if q_type not in ("multiple_choice", "true_false", "short_answer"):
                    q_type = "multiple_choice"

                options = item.get("options", [])
                correct_answer = str(item.get("correct_answer", "")).strip()

                if q_type == "true_false":
                    options = ["True", "False"]
                    if correct_answer.lower() in ("true", "t", "yes"):
                        correct_answer = "True"
                    else:
                        correct_answer = "False"
                elif q_type == "multiple_choice":
                    if not options or len(options) < 2:
                        options = ["Option A", "Option B", "Option C", "Option D"]
                        if not correct_answer:
                            correct_answer = options[0]
                elif q_type == "short_answer":
                    options = []

                page_num = item.get("page_number")
                if page_num is None and chunks:
                    page_num = chunks[0].get("page_number", 1)

                q = QuizQuestion(
                    id=uuid4(),
                    quiz_id=quiz_id,
                    question_index=idx + 1,
                    question_type=q_type,
                    question_text=item.get("question_text", f"Question {idx+1}"),
                    options=options,
                    correct_answer=correct_answer or "See explanation",
                    explanation=item.get("explanation", "Grounded in the document content."),
                    source_excerpt=item.get("source_excerpt", "From document text."),
                    page_number=page_num,
                    chunk_id=None,
                )
                results.append(q)
            except Exception as ex:
                logger.warning(f"Error constructing question item {idx}: {ex}")

        if not results:
            return self._generate_fallback_questions(
                quiz_id=quiz_id,
                chunks=chunks,
                difficulty=difficulty,
                num_questions=num_questions,
                question_types=question_types,
            )

        return results

    def _generate_fallback_questions(
        self,
        quiz_id: UUID,
        chunks: List[Dict[str, Any]],
        difficulty: QuizDifficulty,
        num_questions: int,
        question_types: List[QuestionType],
    ) -> List[QuizQuestion]:
        """
        Deterministic, grounded fallback question generator used when LLM API is unavailable.
        Extracts key sentences from document chunks to guarantee 100% grounded questions.
        """
        questions: List[QuizQuestion] = []
        target_types = question_types or ["multiple_choice", "true_false", "short_answer"]

        for i in range(num_questions):
            chunk = chunks[i % len(chunks)]
            content = chunk.get("content", "").strip()
            page_num = chunk.get("page_number", 1)
            chunk_id_val = chunk.get("id")

            # Extract a coherent sentence
            sentences = [s.strip() for s in re.split(r"[.!?]\s+", content) if len(s.strip()) > 20]
            sentence = sentences[0] if sentences else content[:120]

            q_type = target_types[i % len(target_types)]

            if q_type == "multiple_choice":
                q_text = f"According to the document ({difficulty} level): '{sentence[:60]}...' pertains to what fact?"
                options = [
                    sentence[:90],
                    f"Contradicts: {sentence[:40]} is disabled",
                    "It is an obsolete specification",
                    "None of the above",
                ]
                correct = options[0]
                explanation = f"Directly stated in the document on page {page_num}: '{sentence}'."
            elif q_type == "true_false":
                is_true = (i % 2 == 0)
                if is_true:
                    q_text = f"True or False: The text indicates that '{sentence}'."
                    correct = "True"
                else:
                    q_text = f"True or False: The text explicitly denies that '{sentence}'."
                    correct = "False"
                options = ["True", "False"]
                explanation = f"Verified from document text: '{sentence}'."
            else:  # short_answer
                q_text = f"Summarize what the document states regarding: '{sentence[:60]}...'"
                options = []
                correct = sentence
                explanation = f"Based on the excerpt: '{sentence}'."

            try:
                c_uuid = UUID(str(chunk_id_val)) if chunk_id_val else None
            except Exception:
                c_uuid = None

            questions.append(
                QuizQuestion(
                    id=uuid4(),
                    quiz_id=quiz_id,
                    question_index=i + 1,
                    question_type=q_type,
                    question_text=q_text,
                    options=options,
                    correct_answer=correct,
                    explanation=explanation,
                    source_excerpt=sentence,
                    page_number=page_num,
                    chunk_id=c_uuid,
                )
            )

        return questions

    def grade_quiz(
        self,
        quiz_id: UUID,
        user_id: UUID,
        request: QuizSubmitRequest,
    ) -> QuizSubmitResponse:
        """
        Grades submitted student answers against ground truth stored in database.
        Calculates per-question correctness, total score, and percentage,
        records the attempt, and returns the full review with explanations and source quotes.
        """
        quiz_data = self.quiz_repository.get_quiz(quiz_id=quiz_id, user_id=user_id)
        if not quiz_data:
            raise QuizNotFoundError(f"Quiz {quiz_id} not found or you do not have permission to access it.")

        stored_questions = quiz_data.get("questions", [])
        if not stored_questions:
            raise QuizNotFoundError(f"No questions found for quiz {quiz_id}.")

        review_items: List[QuestionReviewItem] = []
        total_score = 0.0
        total_points = float(len(stored_questions))

        for q in stored_questions:
            qid_str = str(q.get("id"))
            q_type = q.get("question_type", "multiple_choice")
            correct_ans = str(q.get("correct_answer", "")).strip()
            user_ans = str(request.answers.get(qid_str, "")).strip()

            is_correct, points_earned = self._evaluate_answer(
                question_type=q_type,
                user_answer=user_ans,
                correct_answer=correct_ans,
                options=q.get("options", []),
            )

            total_score += points_earned

            review_items.append(
                QuestionReviewItem(
                    question_id=UUID(qid_str),
                    question_index=q.get("question_index", 1),
                    question_type=q_type,
                    question_text=q.get("question_text", ""),
                    options=q.get("options", []),
                    user_answer=user_ans,
                    correct_answer=correct_ans,
                    is_correct=is_correct,
                    explanation=q.get("explanation", ""),
                    source_excerpt=q.get("source_excerpt", ""),
                    page_number=q.get("page_number"),
                )
            )

        percentage = round((total_score / total_points * 100.0), 1) if total_points > 0 else 0.0
        passed = percentage >= 70.0
        attempt_id = uuid4()
        now = datetime.now(timezone.utc)

        # Persist attempt record
        self.quiz_repository.save_quiz_attempt(
            quiz_id=quiz_id,
            user_id=user_id,
            user_answers=request.answers,
            score=total_score,
            total_points=total_points,
            percentage=percentage,
            review_details=[item.model_dump() for item in review_items],
            attempt_id=attempt_id,
        )

        return QuizSubmitResponse(
            attempt_id=attempt_id,
            quiz_id=quiz_id,
            score=total_score,
            total_points=total_points,
            percentage=percentage,
            passed=passed,
            completed_at=now,
            questions=review_items,
        )

    @staticmethod
    def _evaluate_answer(
        question_type: str,
        user_answer: str,
        correct_answer: str,
        options: List[str],
    ) -> (bool, float):
        """
        Evaluates an individual answer against ground truth.
        Supports normalized comparison, option matching, and short answer keyword overlap.
        """
        if not user_answer:
            return False, 0.0

        u_norm = user_answer.strip().lower()
        c_norm = correct_answer.strip().lower()

        if question_type == "true_false":
            # Normalize True/False
            u_bool = u_norm in ("true", "t", "yes", "1")
            c_bool = c_norm in ("true", "t", "yes", "1")
            return (u_bool == c_bool), (1.0 if u_bool == c_bool else 0.0)

        elif question_type == "multiple_choice":
            # Direct match
            if u_norm == c_norm:
                return True, 1.0

            # Match option index (e.g., student passed '0' or 'A')
            option_letters = ["a", "b", "c", "d", "e", "f"]
            if u_norm in option_letters:
                u_idx = option_letters.index(u_norm)
                if u_idx < len(options) and options[u_idx].strip().lower() == c_norm:
                    return True, 1.0

            # Check if student picked the exact text of one option that equals correct_answer
            for opt in options:
                if opt.strip().lower() == u_norm and opt.strip().lower() == c_norm:
                    return True, 1.0

            return False, 0.0

        elif question_type == "short_answer":
            # Direct containment
            if u_norm == c_norm or c_norm in u_norm or u_norm in c_norm:
                return True, 1.0

            # Word / token overlap evaluation
            stop_words = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "that", "it", "for", "on", "with"}
            u_tokens = set(re.findall(r"\w+", u_norm)) - stop_words
            c_tokens = set(re.findall(r"\w+", c_norm)) - stop_words

            if not c_tokens:
                return True, 1.0

            intersection = u_tokens & c_tokens
            recall = len(intersection) / len(c_tokens)

            if recall >= 0.5:
                return True, 1.0
            elif recall >= 0.25:
                return True, 0.5
            else:
                return False, 0.0

        # Default fallback
        is_match = (u_norm == c_norm)
        return is_match, (1.0 if is_match else 0.0)

    def get_quiz_student_view(self, quiz_id: UUID, user_id: UUID) -> QuizResponse:
        """Retrieves an existing quiz in student view mode (no answers exposed)."""
        quiz_data = self.quiz_repository.get_quiz(quiz_id=quiz_id, user_id=user_id)
        if not quiz_data:
            raise QuizNotFoundError(f"Quiz {quiz_id} not found or you do not have permission to access it.")

        stored_questions = quiz_data.get("questions", [])
        student_questions = [
            QuizQuestionStudentView(
                id=UUID(str(q["id"])),
                quiz_id=quiz_id,
                question_index=q["question_index"],
                question_type=q["question_type"],
                question_text=q["question_text"],
                options=q.get("options", []),
                page_number=q.get("page_number"),
            )
            for q in stored_questions
        ]

        # Fetch document name
        doc_id = UUID(str(quiz_data["document_id"]))
        doc_meta = {}
        try:
            doc_meta = self._get_document_metadata(doc_id, user_id)
        except Exception:
            pass

        summary = QuizSummary(
            id=quiz_id,
            document_id=doc_id,
            document_name=doc_meta.get("file_name", "Document"),
            user_id=user_id,
            title=quiz_data.get("title", "Quiz"),
            difficulty=quiz_data.get("difficulty", "medium"),
            total_questions=len(student_questions),
            question_types=quiz_data.get("question_types", ["multiple_choice"]),
            created_at=datetime.fromisoformat(str(quiz_data["created_at"]))
            if isinstance(quiz_data["created_at"], str)
            else quiz_data["created_at"],
        )

        return QuizResponse(quiz=summary, questions=student_questions)

    def list_quizzes(self, document_id: UUID, user_id: UUID) -> List[QuizSummary]:
        """Lists metadata of all quizzes created for a document."""
        self._get_document_metadata(document_id, user_id)  # verify access
        quizzes = self.quiz_repository.list_quizzes_by_document(document_id, user_id)

        results = []
        for q in quizzes:
            results.append(
                QuizSummary(
                    id=UUID(str(q["id"])),
                    document_id=document_id,
                    document_name=q.get("document_name", "Document"),
                    user_id=user_id,
                    title=q.get("title", "Quiz"),
                    difficulty=q.get("difficulty", "medium"),
                    total_questions=q.get("total_questions", 0),
                    question_types=q.get("question_types", []),
                    created_at=datetime.fromisoformat(str(q["created_at"]))
                    if isinstance(q["created_at"], str)
                    else q["created_at"],
                )
            )
        return results

    def get_attempts(self, quiz_id: UUID, user_id: UUID) -> List[QuizAttemptHistoryItem]:
        """Retrieves attempt history for a quiz."""
        quiz_data = self.quiz_repository.get_quiz(quiz_id=quiz_id, user_id=user_id)
        if not quiz_data:
            raise QuizNotFoundError(f"Quiz {quiz_id} not found or you do not have permission to access it.")

        raw_attempts = self.quiz_repository.get_quiz_attempts(quiz_id, user_id)
        items = []
        for a in raw_attempts:
            review_details = [
                QuestionReviewItem(**rd) for rd in a.get("review_details", [])
            ]
            items.append(
                QuizAttemptHistoryItem(
                    id=UUID(str(a["id"])),
                    quiz_id=quiz_id,
                    score=float(a.get("score", 0.0)),
                    total_points=float(a.get("total_points", 0.0)),
                    percentage=float(a.get("percentage", 0.0)),
                    completed_at=datetime.fromisoformat(str(a["completed_at"]))
                    if isinstance(a["completed_at"], str)
                    else a["completed_at"],
                    review_details=review_details,
                )
            )
        return items


default_quiz_service = QuizGenerationService()
