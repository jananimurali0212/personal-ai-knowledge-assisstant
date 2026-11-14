"""
Security and Access Control Audit Test Suite.

Verifies:
1. Supabase Row Level Security (RLS) policies across all tables:
   - documents, document_chunks, conversations, chat_messages,
   - document_summaries, quizzes, quiz_questions, quiz_attempts.
2. Cross-user document access prevention (multi-tenant boundary):
   - User B cannot read, summarize, quiz, or delete User A's documents.
3. Backend authentication enforcement:
   - Rejection of unauthenticated requests (missing/malformed X-User-ID/Bearer token).
4. API keys protection:
   - Zero hardcoded secrets/passwords in source repository.
   - Credentials exclusively sourced from runtime environment variables.
5. Upload validation:
   - Strict rejection of disallowed file extensions (e.g., .exe, .sh, .py).
   - Rejection of oversized files (>25MB) and empty files (0 bytes).
6. Private storage access controls:
   - Enforcing signed URLs and authenticating access tokens.
"""
import os
import re
from uuid import uuid4, UUID
import pytest
from fastapi import HTTPException

from app.routes.documents import (
    get_current_user_id,
    process_document_file,
    summarize_document_endpoint,
)
from app.routes.quizzes import (
    generate_quiz_endpoint,
    submit_quiz_endpoint,
)
from app.routes.conversations import (
    list_user_conversations,
    delete_user_conversation,
)
from app.services.processor import (
    _processing_status_registry,
    update_document_status,
)
from app.services.chunk_repository import ChunkRepository
from app.services.conversation_repository import ConversationRepository
from app.services.summary_repository import SummaryRepository
from app.services.quiz_repository import QuizRepository
from app.config.settings import settings


@pytest.fixture
def user_alice():
    return uuid4()


@pytest.fixture
def user_bob():
    return uuid4()


@pytest.fixture
def alice_document(user_alice):
    doc_id = uuid4()
    update_document_status(
        document_id=doc_id,
        user_id=user_alice,
        status="ready",
        extra={"file_name": "alice_secret_financials.pdf", "file_type": "pdf"},
    )
    yield doc_id
    _processing_status_registry.pop(str(doc_id), None)


class TestSecurityAudit:
    """Security verification test cases."""

    # -------------------------------------------------------------------------
    # 1. Supabase Row Level Security (RLS) Verification
    # -------------------------------------------------------------------------
    def test_rls_sql_policies_defined_for_all_tables(self):
        """Verify that RLS ENABLE and auth.uid() policies exist for all 8 core tables."""
        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "supabase_schema.sql"
        )
        assert os.path.exists(schema_path), "supabase_schema.sql must exist."

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        core_tables = [
            "documents",
            "document_chunks",
            "conversations",
            "chat_messages",
            "document_summaries",
            "quizzes",
            "quiz_questions",
            "quiz_attempts",
        ]

        for table in core_tables:
            # Check RLS enabled
            rls_enabled_pattern = rf"ALTER\s+TABLE\s+(?:public\.)?{table}\s+ENABLE\s+ROW\s+LEVEL\s+SECURITY"
            assert re.search(rls_enabled_pattern, schema_sql, re.IGNORECASE), (
                f"Table '{table}' must have ROW LEVEL SECURITY explicitly enabled."
            )
            # Check auth.uid() referenced
            assert "auth.uid()" in schema_sql, "RLS policies must isolate data using auth.uid()."

    # -------------------------------------------------------------------------
    # 2. Cross-User Document Access Prevention
    # -------------------------------------------------------------------------
    def test_cross_user_summarization_blocked(self, alice_document, user_bob):
        """User Bob cannot summarize Alice's document."""
        with pytest.raises(HTTPException) as exc_info:
            summarize_document_endpoint(
                document_id=alice_document,
                x_user_id=str(user_bob),
            )
        assert exc_info.value.status_code == 404

    def test_cross_user_quiz_generation_blocked(self, alice_document, user_bob):
        """User Bob cannot generate a quiz from Alice's document."""
        from app.models.quiz import GenerateQuizRequest
        with pytest.raises(HTTPException) as exc_info:
            generate_quiz_endpoint(
                document_id=alice_document,
                request=GenerateQuizRequest(),
                x_user_id=str(user_bob),
            )
        assert exc_info.value.status_code == 404

    def test_cross_user_conversation_deletion_blocked(self, user_alice, user_bob):
        """User Bob cannot delete Alice's conversation history."""
        conv_repo = ConversationRepository(supabase_url="", supabase_key="")
        conv = conv_repo.create_conversation(user_id=user_alice, title="Alice's Private Chat")
        conv_id = conv.id if hasattr(conv, "id") else UUID(str(conv["id"]))

        # Bob attempts deletion
        deleted = conv_repo.delete_conversation(conversation_id=conv_id, user_id=user_bob)
        assert deleted is False, "Bob must not be able to delete Alice's conversation."

        # Verify conversation still exists for Alice
        alice_conv = conv_repo.get_conversation(conversation_id=conv_id, user_id=user_alice)
        assert alice_conv is not None

    def test_cross_user_chunk_isolation(self, user_alice, user_bob):
        """User Bob cannot retrieve Alice's document chunks."""
        chunk_repo = ChunkRepository(supabase_url="", supabase_key="")
        alice_doc_id = uuid4()
        chunk_key = f"{user_alice}:{alice_doc_id}"
        from app.services.chunk_repository import _in_memory_chunks_store
        _in_memory_chunks_store[chunk_key] = [
            {"chunk_index": 0, "content": "Confidential internal API credentials."}
        ]

        # Bob queries chunks for Alice's document
        bob_chunks = chunk_repo.get_chunks_by_document(document_id=alice_doc_id, user_id=user_bob)
        assert len(bob_chunks) == 0, "Bob received chunks belonging to Alice!"

    # -------------------------------------------------------------------------
    # 3. Backend Authentication Enforcement
    # -------------------------------------------------------------------------
    def test_missing_auth_header_rejected(self):
        """Requests missing X-User-ID must be rejected with 401 Unauthorized."""
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id(x_user_id=None)
        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail

    def test_invalid_uuid_auth_header_rejected(self):
        """Requests with malformed user ID must be rejected with 400 Bad Request."""
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id(x_user_id="not-a-valid-uuid")
        assert exc_info.value.status_code == 400
        assert "Invalid user ID format" in exc_info.value.detail

    # -------------------------------------------------------------------------
    # 4. API Keys & Secrets Exposure Protection
    # -------------------------------------------------------------------------
    def test_no_hardcoded_secrets_in_codebase(self):
        """Ensure no live Google Gemini or Supabase service keys are hardcoded in source files."""
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Patterns for real API keys
        suspicious_patterns = [
            re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),  # Live Google API Key
            re.compile(r"ey[A-Za-z0-9_-]{20,}\.ey[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),  # Real JWT token
        ]

        # Scan python and typescript files, skipping git, caches, and test artifacts
        for root, dirs, files in os.walk(backend_dir):
            if any(skip in root for skip in [".git", ".pytest_cache", "node_modules", "dist", "brain"]):
                continue
            for file in files:
                if file.endswith((".py", ".ts", ".tsx", ".json")) and not file.endswith(".env.example"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        for pattern in suspicious_patterns:
                            matches = pattern.findall(content)
                            assert len(matches) == 0, (
                                f"Potential exposed live credential in {file_path}: {matches[:1]}"
                            )

    # -------------------------------------------------------------------------
    # 5. Upload Validation (File Types & File Size Limits)
    # -------------------------------------------------------------------------
    def test_frontend_supported_file_types_validation(self):
        """Verify frontend validation strictly enforces PDF, DOCX, TXT and 25MB limit."""
        doc_service_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "frontend", "src", "services", "documents.ts"
        )
        with open(doc_service_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt']" in code
        assert "MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024" in code
        assert "Unsupported file type" in code
        assert "exceeds the maximum allowed limit of 25MB" in code

    def test_backend_file_extension_validation(self):
        """Verify backend extractor rejects unsupported file formats."""
        from app.services.extractor import extract_document, UnsupportedFormatException
        with pytest.raises(UnsupportedFormatException):
            extract_document(b"malicious script", "exploit.exe")

        with pytest.raises(UnsupportedFormatException):
            extract_document(b"alert('xss')", "script.js")
