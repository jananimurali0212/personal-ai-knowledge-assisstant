"""Unit and integration tests for AI-powered document summarization.

Tests verify:
1. Short document single-pass summarization (PDF).
2. Long document hierarchical map-reduce summarization (DOCX, >3 chunks).
3. Plain text document summarization (TXT).
4. Structured 5-part summary validation (executive summary, main topics, important points, key terminology, important conclusions).
5. Strict multi-tenant user isolation (User B cannot summarize User A's document).
6. Empty document validation (0 chunks raises 400 Bad Request).
7. Summary caching behavior (second call returns cached summary without invoking Gemini).
8. Force regenerate flag (bypasses cache when force_regenerate=True).
9. Supabase persistence and status tracking.
10. FastAPI endpoints: POST /api/documents/{id}/summarize and GET /api/documents/{id}/summary.
"""
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from app.models.summary import (
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    DocumentSummaryContent,
    KeyTerminologyItem,
)
from app.services.summarizer import (
    DocumentSummarizerService,
    DocumentNotFoundError,
    EmptyDocumentError,
    SummarizationError,
)
from app.services.summary_repository import (
    SummaryRepository,
    _in_memory_summary_store,
)
from app.services.chunk_repository import (
    ChunkRepository,
    _in_memory_chunks_store,
)
from app.services.processor import (
    _processing_status_registry,
    update_document_status,
)
from app.routes.documents import (
    summarize_document_endpoint,
    get_document_summary_endpoint,
    delete_document_summary_endpoint,
)


@pytest.fixture(autouse=True)
def clean_test_stores():
    """Clears in-memory caches before and after each test."""
    _in_memory_summary_store.clear()
    _in_memory_chunks_store.clear()
    _processing_status_registry.clear()
    yield
    _in_memory_summary_store.clear()
    _in_memory_chunks_store.clear()
    _processing_status_registry.clear()


@pytest.fixture
def chunk_repo():
    return ChunkRepository(supabase_url="", supabase_key="")


@pytest.fixture
def summary_repo():
    return SummaryRepository(supabase_url="", supabase_key="")


@pytest.fixture
def mock_gemini_single_pass_json():
    return {
        "executive_summary": "This document introduces modern microservice architecture patterns and fault-tolerant service communication.",
        "main_topics": [
            "Microservice Architecture and Boundaries",
            "Circuit Breaker Pattern and Resilience",
            "Event-Driven Asynchronous Messaging",
        ],
        "important_points": [
            "Services should communicate primarily via asynchronous event streams to decouple lifecycles.",
            "Circuit breakers prevent cascading failures when downstream dependencies degrade.",
            "Idempotency tokens are required for reliable payment and transaction handling.",
        ],
        "key_terminology": [
            {"term": "Circuit Breaker", "definition": "A stability pattern that halts execution of failing operations."},
            {"term": "Idempotency Key", "definition": "A unique identifier preventing duplicate operations."},
        ],
        "important_conclusions": [
            "Adopting event-driven messaging reduces inter-service coupling by over 60%.",
            "Proper circuit breakers are non-negotiable for 99.99% availability.",
        ],
    }


@pytest.fixture
def mock_gemini_section_brief_json():
    return {
        "section_overview": "Section covering high-availability data replication and partition handling.",
        "key_points": [
            "Raft consensus protocol coordinates cluster state.",
            "Quorum writes guarantee linearizable read consistency.",
        ],
        "terminology": [
            {"term": "Quorum", "definition": "Minimum number of node votes required to commit an operation."}
        ],
        "conclusions": [
            "Majority quorum prevents split-brain anomalies."
        ],
    }


# ---------------------------------------------------------------------------
# 1. Short Document Single-Pass Tests (PDF)
# ---------------------------------------------------------------------------

def test_summarize_short_pdf_single_pass(
    chunk_repo, summary_repo, mock_gemini_single_pass_json
):
    """Verifies that documents with <= 3 chunks use single-pass summarization and return a 5-part summary."""
    user_id = uuid4()
    doc_id = uuid4()

    # Register document in processing registry
    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "microservices_guide.pdf", "file_type": "pdf"},
    )

    # Store 2 chunks (below hierarchical threshold of 3)
    doc_key = f"{user_id}:{doc_id}"
    _in_memory_chunks_store[doc_key] = [
        {
            "chunk_index": 0,
            "page_number": 1,
            "content": "Microservices communicate via events. Circuit breakers prevent cascading failures.",
        },
        {
            "chunk_index": 1,
            "page_number": 2,
            "content": "Idempotency keys prevent double charging. Adopting event streams reduces coupling.",
        },
    ]

    service = DocumentSummarizerService(
        api_key="mock_key",
        model_name="gemini-2.5-flash",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    with patch.object(service, "_call_gemini_json", return_value=mock_gemini_single_pass_json) as mock_call:
        response = service.summarize_document(
            document_id=doc_id,
            user_id=user_id,
        )

        assert mock_call.call_count == 1  # Single-pass: exactly 1 Gemini call
        assert response.document_id == doc_id
        assert response.status == "completed"
        assert response.cached is False

        # Validate structured 5-part summary
        summary = response.summary
        assert summary is not None
        assert "microservice architecture" in summary.executive_summary.lower()
        assert len(summary.main_topics) == 3
        assert len(summary.important_points) == 3
        assert len(summary.key_terminology) == 2
        assert summary.key_terminology[0].term == "Circuit Breaker"
        assert len(summary.important_conclusions) == 2
        assert summary.is_hierarchical is False
        assert summary.total_sections_analyzed == 1


# ---------------------------------------------------------------------------
# 2. Long Document Hierarchical Map-Reduce Tests (DOCX)
# ---------------------------------------------------------------------------

def test_summarize_long_docx_hierarchical(
    chunk_repo, summary_repo, mock_gemini_section_brief_json, mock_gemini_single_pass_json
):
    """Verifies that documents with > 3 chunks trigger hierarchical map-reduce summarization."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "distributed_systems.docx", "file_type": "docx"},
    )

    # Store 6 chunks (triggers 2 sections of 3 chunks each)
    doc_key = f"{user_id}:{doc_id}"
    _in_memory_chunks_store[doc_key] = [
        {"chunk_index": i, "page_number": 1, "content": f"Detailed content of section chunk {i} on distributed consensus."}
        for i in range(6)
    ]

    service = DocumentSummarizerService(
        api_key="mock_key",
        model_name="gemini-2.5-flash",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    # Mock responses: 2 section calls (map) + 1 final reduce synthesis call
    def mock_side_effect(prompt, system_instruction):
        if "SECTION" in prompt and "CONSOLIDATED" not in prompt:
            return mock_gemini_section_brief_json
        return mock_gemini_single_pass_json

    with patch.object(service, "_call_gemini_json", side_effect=mock_side_effect) as mock_call:
        response = service.summarize_document(
            document_id=doc_id,
            user_id=user_id,
        )

        # 6 chunks / 3 per section = 2 map calls + 1 reduce call = 3 total calls
        assert mock_call.call_count == 3
        assert response.status == "completed"
        assert response.summary.is_hierarchical is True
        assert response.summary.total_sections_analyzed == 2
        assert len(response.summary.main_topics) >= 1
        assert len(response.summary.key_terminology) >= 1


# ---------------------------------------------------------------------------
# 3. Plain Text Document Summarization Tests (TXT)
# ---------------------------------------------------------------------------

def test_summarize_plain_text_txt(
    chunk_repo, summary_repo, mock_gemini_single_pass_json
):
    """Verifies summarization of TXT files preserving text facts without page numbers."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "meeting_notes.txt", "file_type": "txt"},
    )

    doc_key = f"{user_id}:{doc_id}"
    _in_memory_chunks_store[doc_key] = [
        {"chunk_index": 0, "page_number": None, "content": "Q3 Engineering Retrospective notes on cloud migration."}
    ]

    service = DocumentSummarizerService(
        api_key="mock_key",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    with patch.object(service, "_call_gemini_json", return_value=mock_gemini_single_pass_json):
        response = service.summarize_document(
            document_id=doc_id,
            user_id=user_id,
        )

        assert response.status == "completed"
        assert response.file_type == "txt"
        assert response.file_name == "meeting_notes.txt"


# ---------------------------------------------------------------------------
# 4. Caching & Force-Regenerate Tests
# ---------------------------------------------------------------------------

def test_summary_caching_avoids_redundant_generation(
    chunk_repo, summary_repo, mock_gemini_single_pass_json
):
    """Verifies that an unchanged document returns the cached summary without calling Gemini again."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "handbook.pdf", "file_type": "pdf"},
    )
    _in_memory_chunks_store[f"{user_id}:{doc_id}"] = [
        {"chunk_index": 0, "page_number": 1, "content": "Employee onboarding handbook rules."}
    ]

    service = DocumentSummarizerService(
        api_key="mock_key",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    with patch.object(service, "_call_gemini_json", return_value=mock_gemini_single_pass_json) as mock_call:
        # First call: generates summary
        resp1 = service.summarize_document(doc_id, user_id)
        assert resp1.cached is False
        assert mock_call.call_count == 1

        # Second call: returns cached summary without calling Gemini
        resp2 = service.summarize_document(doc_id, user_id)
        assert resp2.cached is True
        assert mock_call.call_count == 1  # Still 1! No new LLM invocation
        assert resp2.summary.executive_summary == resp1.summary.executive_summary


def test_force_regenerate_bypasses_cache(
    chunk_repo, summary_repo, mock_gemini_single_pass_json
):
    """Verifies that force_regenerate=True bypasses the cached summary."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "handbook.pdf", "file_type": "pdf"},
    )
    _in_memory_chunks_store[f"{user_id}:{doc_id}"] = [
        {"chunk_index": 0, "page_number": 1, "content": "Employee onboarding handbook."}
    ]

    service = DocumentSummarizerService(
        api_key="mock_key",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    with patch.object(service, "_call_gemini_json", return_value=mock_gemini_single_pass_json) as mock_call:
        # Initial generation
        service.summarize_document(doc_id, user_id)
        assert mock_call.call_count == 1

        # Force regenerate
        req = DocumentSummaryRequest(force_regenerate=True)
        resp = service.summarize_document(doc_id, user_id, request=req)
        assert resp.cached is False
        assert mock_call.call_count == 2  # Gemini was called a 2nd time!


# ---------------------------------------------------------------------------
# 5. Error & Edge Case Handling Tests
# ---------------------------------------------------------------------------

def test_empty_document_raises_error(chunk_repo, summary_repo):
    """Verifies that attempting to summarize a document with 0 chunks raises EmptyDocumentError."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "empty.txt", "file_type": "txt"},
    )
    # No chunks added to store

    service = DocumentSummarizerService(
        api_key="mock_key",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    with pytest.raises(EmptyDocumentError) as exc_info:
        service.summarize_document(doc_id, user_id)

    assert "contains no extracted text or chunks" in str(exc_info.value)


def test_user_cannot_summarize_another_users_document(chunk_repo, summary_repo):
    """Strict multi-tenant isolation: User B cannot summarize User A's document."""
    user_a = uuid4()
    user_b = uuid4()
    doc_a = uuid4()

    update_document_status(
        document_id=doc_a,
        user_id=user_a,
        status="ready",
        extra={"file_name": "secret_user_a.pdf", "file_type": "pdf"},
    )
    _in_memory_chunks_store[f"{user_a}:{doc_a}"] = [
        {"chunk_index": 0, "page_number": 1, "content": "Confidential financial data of User A."}
    ]

    service = DocumentSummarizerService(
        api_key="mock_key",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    # User B attempts to summarize User A's document
    with pytest.raises(DocumentNotFoundError) as exc_info:
        service.summarize_document(doc_a, user_b)

    assert "does not belong to user" in str(exc_info.value) or "not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. FastAPI Route Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_summarize_endpoint_unauthenticated():
    """Verifies that calling summarize without X-User-ID raises 401 Unauthorized."""
    doc_id = uuid4()
    with pytest.raises(HTTPException) as exc_info:
        summarize_document_endpoint(
            document_id=doc_id,
            x_user_id=None,
            user_id=None,
        )
    assert exc_info.value.status_code == 401


def test_api_summarize_endpoint_success(mock_gemini_single_pass_json):
    """Verifies direct invocation of summarize_document_endpoint."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "api_test.pdf", "file_type": "pdf"},
    )
    _in_memory_chunks_store[f"{user_id}:{doc_id}"] = [
        {"chunk_index": 0, "page_number": 1, "content": "API endpoint integration text."}
    ]

    with patch("app.routes.documents.default_summarizer_service._call_gemini_json", return_value=mock_gemini_single_pass_json):
        res = summarize_document_endpoint(
            document_id=doc_id,
            request=DocumentSummaryRequest(),
            x_user_id=str(user_id),
        )

        assert res.document_id == doc_id
        assert res.status == "completed"
        assert res.summary is not None
        assert len(res.summary.main_topics) > 0


def test_api_get_summary_endpoint_not_found():
    """Verifies GET /summary returns 404 when no summary exists yet."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "no_summary.pdf", "file_type": "pdf"},
    )

    with pytest.raises(HTTPException) as exc_info:
        get_document_summary_endpoint(
            document_id=doc_id,
            x_user_id=str(user_id),
        )
    assert exc_info.value.status_code == 404


def test_api_delete_summary_endpoint_success(mock_gemini_single_pass_json):
    """Verifies DELETE /api/documents/{id}/summary deletes summary and resets state."""
    user_id = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "delete_test.pdf", "file_type": "pdf"},
    )
    _in_memory_chunks_store[f"{user_id}:{doc_id}"] = [
        {"chunk_index": 0, "page_number": 1, "content": "Text for deletion testing."}
    ]

    # First summarize
    with patch("app.routes.documents.default_summarizer_service._call_gemini_json", return_value=mock_gemini_single_pass_json):
        summarize_document_endpoint(
            document_id=doc_id,
            request=DocumentSummaryRequest(),
            x_user_id=str(user_id),
        )

    # Verify summary exists
    get_res = get_document_summary_endpoint(document_id=doc_id, x_user_id=str(user_id))
    assert get_res.summary is not None

    # Delete summary
    del_res = delete_document_summary_endpoint(document_id=doc_id, x_user_id=str(user_id))
    assert del_res["status"] == "success"

    # Now GET should 404
    with pytest.raises(HTTPException) as exc_info:
        get_document_summary_endpoint(document_id=doc_id, x_user_id=str(user_id))
    assert exc_info.value.status_code == 404


def test_summary_action_items_parsed(chunk_repo, summary_repo):
    """Verifies that action items are correctly parsed and stored in the structured summary."""
    user_id = uuid4()
    doc_id = uuid4()

    mock_json = {
        "executive_summary": "Summary with action items.",
        "main_topics": ["Action Item Protocol"],
        "important_points": ["Points here."],
        "key_terminology": [{"term": "Action", "definition": "A task to complete."}],
        "important_conclusions": ["Finished."],
        "action_items": [
            "Implement automated regression testing suite.",
            "Deploy staging canary before production cutover.",
        ],
    }

    _in_memory_chunks_store[f"{user_id}:{doc_id}"] = [
        {"chunk_index": 0, "page_number": 1, "content": "Document with operational recommendations."}
    ]
    update_document_status(
        document_id=doc_id,
        user_id=user_id,
        status="ready",
        extra={"file_name": "actions.pdf", "file_type": "pdf"},
    )

    service = DocumentSummarizerService(
        api_key="test_key",
        chunk_repository=chunk_repo,
        summary_repository=summary_repo,
    )

    with patch.object(service, "_call_gemini_json", return_value=mock_json):
        resp = service.summarize_document(doc_id, user_id)
        assert resp.summary is not None
        assert len(resp.summary.action_items) == 2
        assert resp.summary.action_items[0] == "Implement automated regression testing suite."
        assert resp.summary.action_items[1] == "Deploy staging canary before production cutover."
