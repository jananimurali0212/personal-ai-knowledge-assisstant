"""
Demonstration & Verification Script: AI-Powered Quiz Generation.

Demonstrates:
1. Grounded quiz generation with configurable difficulty levels (Easy, Medium, Hard).
2. Support for multiple question formats: Multiple Choice, True/False, and Short Answer.
3. Anti-cheating security: Student view cleanly withholds answers, explanations, and source quotes.
4. Interactive submission: Automated grading, percentage calculation, and pass/fail evaluation.
5. Review breakdown: Explanations and verbatim document source excerpts with page numbers.
6. History tracking: Retrieval of recorded attempts.
"""
import sys
import os
from uuid import uuid4

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.quiz import (
    GenerateQuizRequest,
    QuizSubmitRequest,
)
from app.services.quiz import (
    QuizGenerationService,
)
from app.services.quiz_repository import (
    QuizRepository,
)
from app.services.chunk_repository import (
    ChunkRepository,
)
from app.services.processor import (
    update_document_status,
)


def run_demo():
    print("=" * 88)
    print("      PERSONAL AI KNOWLEDGE ASSISTANT: QUIZ GENERATION VERIFICATION & DEMO      ")
    print("=" * 88)

    user_id = uuid4()
    doc_id = uuid4()
    file_name = "Cloud_Native_Architecture_DeepDive.pdf"

    # Register document
    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": file_name, "file_type": "pdf"},
    )

    # Populate chunks
    chunk_repo = ChunkRepository(supabase_url="", supabase_key="")
    quiz_repo = QuizRepository(supabase_url="", supabase_key="")

    chunks = [
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "user_id": str(user_id),
            "chunk_index": 0,
            "content": (
                "Microservice architectures decompose complex monolithic systems into autonomous, loosely coupled services. "
                "Each service owns its own private data store to enforce strict domain boundaries and prevent accidental coupling."
            ),
            "page_number": 1,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "user_id": str(user_id),
            "chunk_index": 1,
            "content": (
                "The Circuit Breaker pattern prevents cascading failures across distributed systems by monitoring downstream call failures. "
                "When failure rates breach the threshold, the circuit transitions from Closed to Open, immediately rejecting subsequent requests without waiting for timeouts."
            ),
            "page_number": 2,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "user_id": str(user_id),
            "chunk_index": 2,
            "content": (
                "Idempotency keys ensure that network retries do not result in duplicate transactions. "
                "In payment processing, repeating a request with the exact same idempotency token returns the original cached response without re-executing billing logic."
            ),
            "page_number": 3,
        },
    ]
    chunk_repo.get_chunks_by_document = lambda d, u: chunks

    service = QuizGenerationService(
        api_key="mock_key",
        chunk_repository=chunk_repo,
        quiz_repository=quiz_repo,
    )

    # -------------------------------------------------------------------------
    # 1. Generate Quiz (Medium Difficulty, 3 Questions)
    # -------------------------------------------------------------------------
    print(f"\n[1] Generating Quiz from '{file_name}'...")
    quiz_req = GenerateQuizRequest(
        difficulty="medium",
        num_questions=3,
        question_types=["multiple_choice", "true_false", "short_answer"],
        title="Microservice Architecture & Resilience Exam",
    )

    quiz_resp = service.generate_quiz(
        document_id=doc_id,
        user_id=user_id,
        request=quiz_req,
    )

    print(f"   Quiz Title: {quiz_resp.quiz.title}")
    print(f"   Difficulty: {quiz_resp.quiz.difficulty.upper()}")
    print(f"   Total Questions: {quiz_resp.quiz.total_questions}")
    print(f"   Session ID: {quiz_resp.quiz.id}")

    # -------------------------------------------------------------------------
    # 2. Verify Anti-Cheating Student View
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Anti-Cheating Student View (Answers Withheld)...")
    for q in quiz_resp.questions:
        print(f"   Question {q.question_index} [{q.question_type.upper()}] (Page {q.page_number}):")
        print(f"     Text: {q.question_text}")
        if q.options:
            print(f"     Options: {q.options}")
        # Validate that sensitive fields are completely omitted
        assert not hasattr(q, "correct_answer"), "Security Breach: correct_answer exposed!"
        assert not hasattr(q, "explanation"), "Security Breach: explanation exposed!"
        assert not hasattr(q, "source_excerpt"), "Security Breach: source_excerpt exposed!"

    print("   ✓ Verified: All correct answers, explanations, and quotes are withheld.")

    # -------------------------------------------------------------------------
    # 3. Simulate Student Submission & Automated Grading
    # -------------------------------------------------------------------------
    print("\n[3] Submitting Student Answers & Automated Grading...")
    # Fetch ground truth internally for simulation
    stored_quiz = quiz_repo.get_quiz(quiz_resp.quiz.id, user_id)
    questions = stored_quiz["questions"]

    answers = {
        str(questions[0]["id"]): questions[0]["correct_answer"],  # Correct MC
        str(questions[1]["id"]): questions[1]["correct_answer"],  # Correct TF
        str(questions[2]["id"]): "Idempotency keys prevent duplicate transactions during network retries.",  # Correct Short Answer
    }

    submit_req = QuizSubmitRequest(answers=answers)
    graded_result = service.grade_quiz(
        quiz_id=quiz_resp.quiz.id,
        user_id=user_id,
        request=submit_req,
    )

    print(f"   Attempt ID: {graded_result.attempt_id}")
    print(f"   Score: {graded_result.score} / {graded_result.total_points} points")
    print(f"   Percentage: {graded_result.percentage}%")
    print(f"   Passed: {'YES' if graded_result.passed else 'NO'}")

    # -------------------------------------------------------------------------
    # 4. Review Breakdown & Grounded Excerpt Citations
    # -------------------------------------------------------------------------
    print("\n[4] Reviewing Graded Questions & Source Excerpts:")
    for rev in graded_result.questions:
        status_symbol = "✓" if rev.is_correct else "✗"
        print(f"\n   {status_symbol} Q{rev.question_index}: {rev.question_text}")
        print(f"      Student Answer: {rev.user_answer}")
        print(f"      Correct Answer: {rev.correct_answer}")
        print(f"      Explanation:    {rev.explanation}")
        print(f"      Source Excerpt: \"{rev.source_excerpt}\" (Page {rev.page_number})")

    # -------------------------------------------------------------------------
    # 5. History & Attempt Persistence
    # -------------------------------------------------------------------------
    print("\n[5] Fetching Attempt History:")
    history = service.get_attempts(quiz_resp.quiz.id, user_id)
    print(f"   Total Recorded Attempts: {len(history)}")
    for att in history:
        print(f"   - Attempt {att.id}: {att.percentage}% at {att.completed_at}")

    print("\n" + "=" * 88)
    print("           ALL QUIZ GENERATION VERIFICATIONS SUCCESSFULLY COMPLETED!           ")
    print("=" * 88)


if __name__ == "__main__":
    run_demo()
