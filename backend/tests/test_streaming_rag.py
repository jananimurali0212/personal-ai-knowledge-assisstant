"""Unit and integration tests for real-time Server-Sent Events (SSE) AI response streaming.
Tests cover:
1. Generation service streaming (Gemini generate_content_stream)
2. SSE format helper (event: <type>\\ndata: <json>\\n\\n)
3. RAG Orchestrator streaming pipeline (ask_stream)
4. Event sequence verification (start -> token -> sources -> complete)
5. Insufficient context streaming behavior
6. Cancellation handling when client disconnects
7. FastAPI route POST /api/chat/stream
"""
import json
import asyncio
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from starlette.requests import Request

from app.models.chat import ChatAskRequest
from app.models.search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
)
from app.services.rag import (
    RAGOrchestratorService,
    format_sse,
    INSUFFICIENT_CONTEXT_MESSAGE,
)
from app.services.generation import (
    GeminiGenerationService,
    GenerateAnswerResponse,
    GeminiRateLimitError,
)
from app.services.conversation_repository import (
    ConversationRepository,
    _in_memory_conversations_store,
    _in_memory_messages_store,
)
from app.routes.chat import chat_stream_endpoint


# Helper class to mock Gemini streaming chunks
class MockChunk:
    def __init__(self, text: str):
        self.text = text


@pytest.fixture(autouse=True)
def reset_in_memory_stores():
    _in_memory_conversations_store.clear()
    _in_memory_messages_store.clear()
    yield
    _in_memory_conversations_store.clear()
    _in_memory_messages_store.clear()


@pytest.fixture
def sample_context():
    return (
        "Java inheritance is a mechanism in which one class acquires the properties and "
        "behaviors of a parent class using the 'extends' keyword. It establishes an IS-A relationship."
    )


@pytest.fixture
def sample_sources():
    return [
        SearchResultItem(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_name="Java_Guide.pdf",
            chunk_index=0,
            content="Java inheritance uses extends.",
            similarity=0.92,
            page_number=1,
        )
    ]


@pytest.fixture
def in_memory_repo():
    return ConversationRepository(supabase_url="", supabase_key="")


# ---------------------------------------------------------------------------
# 1. SSE Formatting Helper Tests
# ---------------------------------------------------------------------------

def test_format_sse_basic():
    """Verifies that format_sse outputs the standard SSE wire format."""
    sse_output = format_sse("token", {"token": "Hello"})
    expected = 'event: token\ndata: {"token": "Hello"}\n\n'
    assert sse_output == expected


def test_format_sse_complex_payload():
    """Verifies format_sse serializes dictionaries and lists correctly."""
    data = {"sources": [{"id": 1, "name": "doc.pdf"}], "count": 1}
    sse_output = format_sse("sources", data)
    assert sse_output.startswith("event: sources\ndata: ")
    assert sse_output.endswith("\n\n")
    parsed_json = json.loads(sse_output.replace("event: sources\ndata: ", "").strip())
    assert parsed_json["count"] == 1
    assert parsed_json["sources"][0]["name"] == "doc.pdf"


# ---------------------------------------------------------------------------
# 2. Generation Service Streaming Tests
# ---------------------------------------------------------------------------

def test_generate_answer_stream_tokens():
    """Tests that generate_answer_stream yields tokens progressively from Gemini stream."""
    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = [
        MockChunk("Inheritance in Java "),
        MockChunk("allows a subclass "),
        MockChunk("to reuse code [Source 1]."),
    ]

    svc = GeminiGenerationService(api_key="test-key", model_name="gemini-2.5-flash")
    svc._client = mock_client

    tokens = list(svc.generate_answer_stream(
        question="What is Java inheritance?",
        context="Java inheritance allows a subclass to reuse code.",
    ))

    assert tokens == [
        "Inheritance in Java ",
        "allows a subclass ",
        "to reuse code [Source 1].",
    ]


def test_generate_answer_stream_insufficient_context():
    """Tests that empty or missing context directly yields the insufficient context notice."""
    svc = GeminiGenerationService(api_key="test-key", model_name="gemini-2.5-flash")
    tokens = list(svc.generate_answer_stream(
        question="What is Java inheritance?",
        context="   ",  # whitespace only
    ))
    assert len(tokens) == 1
    assert "sufficient information" in tokens[0].lower()


def test_generate_answer_stream_api_error_handling():
    """Tests that API rate limit exceptions during streaming raise GeminiRateLimitError."""
    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = Exception("429 Resource Exhausted")

    svc = GeminiGenerationService(api_key="test-key", model_name="gemini-2.5-flash")
    svc._client = mock_client

    with pytest.raises(GeminiRateLimitError):
        list(svc.generate_answer_stream(
            question="What is Java inheritance?",
            context="Java inheritance is a core OOP concept.",
        ))


# ---------------------------------------------------------------------------
# 3. RAG Orchestrator ask_stream Tests
# ---------------------------------------------------------------------------

def test_ask_stream_event_sequence(sample_sources, in_memory_repo):
    """Verifies that ask_stream yields the strict sequence: start -> token(s) -> sources -> complete."""
    mock_search = MagicMock()
    mock_search.search.return_value = SemanticSearchResponse(
        query="What is Java inheritance?",
        total_results=len(sample_sources),
        match_threshold=0.3,
        embedding_model="text-embedding-004",
        results=sample_sources,
    )

    mock_gen = MagicMock()
    def fake_stream(*args, **kwargs):
        yield "Java "
        yield "inheritance "
        yield "allows "
        yield "subclasses [Source 1]."
    mock_gen.generate_answer_stream = fake_stream
    mock_gen.validate_and_sanitize_citations.return_value = (
        "Java inheritance allows subclasses [Source 1].",
        ["Source 1"]
    )

    rag = RAGOrchestratorService(
        search_service=mock_search,
        generation_service=mock_gen,
        conversation_repository=in_memory_repo,
    )

    user_id = uuid4()
    req = ChatAskRequest(question="What is Java inheritance?")

    events = list(rag.ask_stream(req, user_id=user_id))

    # Parse received SSE events
    parsed_events = []
    for raw in events:
        lines = [l for l in raw.strip().split("\n") if l]
        event_name = lines[0].replace("event: ", "").strip()
        data = json.loads(lines[1].replace("data: ", "").strip())
        parsed_events.append((event_name, data))

    event_types = [e[0] for e in parsed_events]
    assert event_types[0] == "start"
    assert "token" in event_types
    assert "sources" in event_types
    assert event_types[-1] == "complete"

    # Verify complete event metadata
    complete_event = parsed_events[-1][1]
    assert complete_event["grounded"] is True
    assert complete_event["is_insufficient_context"] is False

    # Verify sources event payload
    sources_event = [e[1] for e in parsed_events if e[0] == "sources"][0]
    assert len(sources_event["sources"]) == 1
    assert sources_event["document_names"] == ["Java_Guide.pdf"]
    assert sources_event["page_numbers"] == [1]


def test_ask_stream_insufficient_context_handling(in_memory_repo):
    """Verifies that when no chunks meet threshold, the stream yields insufficient context."""
    mock_search = MagicMock()
    mock_search.search.return_value = SemanticSearchResponse(
        query="Completely unrelated topic?",
        total_results=0,
        match_threshold=0.3,
        embedding_model="text-embedding-004",
        results=[],
    )

    mock_gen = MagicMock()

    rag = RAGOrchestratorService(
        search_service=mock_search,
        generation_service=mock_gen,
        conversation_repository=in_memory_repo,
    )

    user_id = uuid4()
    req = ChatAskRequest(question="Completely unrelated topic?")

    parsed_events = []
    for sse_chunk in rag.ask_stream(req, user_id=user_id):
        lines = [l for l in sse_chunk.strip().split("\n") if l]
        event_name = lines[0].replace("event: ", "").strip()
        data = json.loads(lines[1].replace("data: ", "").strip())
        parsed_events.append((event_name, data))

    event_types = [e[0] for e in parsed_events]
    assert "start" in event_types
    assert "token" in event_types
    assert "complete" in event_types

    # Tokens should convey insufficient context notice
    tokens = [e[1]["token"] for e in parsed_events if e[0] == "token"]
    full_text = "".join(tokens)
    assert INSUFFICIENT_CONTEXT_MESSAGE in full_text

    # Complete event should flag insufficient context
    complete_data = [e[1] for e in parsed_events if e[0] == "complete"][0]
    assert complete_data["grounded"] is False
    assert complete_data["is_insufficient_context"] is True


def test_ask_stream_client_cancellation(in_memory_repo):
    """Verifies that when client disconnects (raw_request.is_disconnected() == True), generation stops."""
    mock_search = MagicMock()
    mock_search.search.return_value = SemanticSearchResponse(
        query="What is Java inheritance?",
        total_results=1,
        match_threshold=0.3,
        embedding_model="text-embedding-004",
        results=[
            SearchResultItem(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_name="Java.pdf",
                chunk_index=0,
                content="Java guide",
                similarity=0.9,
            )
        ],
    )

    tokens_generated = 0
    def infinite_stream(*args, **kwargs):
        nonlocal tokens_generated
        for i in range(100):
            tokens_generated += 1
            yield f"token_{i} "

    mock_gen = MagicMock()
    mock_gen.generate_answer_stream = infinite_stream

    rag = RAGOrchestratorService(
        search_service=mock_search,
        generation_service=mock_gen,
        conversation_repository=in_memory_repo,
    )

    # Mock raw request that disconnects after 3 calls
    disconnect_counter = 0
    def fake_is_disconnected():
        nonlocal disconnect_counter
        disconnect_counter += 1
        return disconnect_counter >= 3

    mock_req = MagicMock(spec=Request)
    mock_req.is_disconnected = fake_is_disconnected

    user_id = uuid4()
    req = ChatAskRequest(question="What is Java inheritance?")

    yielded_chunks = list(rag.ask_stream(req, user_id=user_id, raw_request=mock_req))

    # Should have stopped well before all 100 tokens
    assert tokens_generated < 20


# ---------------------------------------------------------------------------
# 4. FastAPI Route Endpoint Tests
# ---------------------------------------------------------------------------

def test_chat_stream_endpoint_unauthenticated():
    """Verifies that chat_stream endpoint raises 401 when no user ID is provided."""
    req = ChatAskRequest(question="Valid question?")
    mock_raw = MagicMock(spec=Request)

    with pytest.raises(HTTPException) as exc_info:
        chat_stream_endpoint(
            raw_request=mock_raw,
            request=req,
            current_user=None,
        )
    assert exc_info.value.status_code == 401


def test_chat_stream_endpoint_empty_question():
    """Verifies that ChatAskRequest validation rejects blank questions."""
    with pytest.raises(ValidationError):
        ChatAskRequest(question="   ")


def test_chat_stream_endpoint_returns_streaming_response(sample_sources):
    """Verifies that the endpoint returns a StreamingResponse with text/event-stream media type."""
    mock_rag = MagicMock()
    def mock_stream_gen(*args, **kwargs):
        yield 'event: start\ndata: {"conversation_id": "c1", "status": "started"}\n\n'
        yield 'event: token\ndata: {"token": "Hi"}\n\n'
        yield 'event: complete\ndata: {"grounded": true, "is_insufficient_context": false}\n\n'

    mock_rag.ask_stream = mock_stream_gen

    req = ChatAskRequest(question="What is Java inheritance?")
    mock_raw = MagicMock(spec=Request)
    user_id = uuid4()

    resp = chat_stream_endpoint(
        raw_request=mock_raw,
        request=req,
        current_user=user_id,
        rag_service=mock_rag,
    )

    assert resp.media_type == "text/event-stream"
    assert resp.headers["Cache-Control"] == "no-cache"
    assert resp.headers["Connection"] == "keep-alive"
