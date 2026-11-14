"""Comprehensive testing suite for RAG Orchestration Pipeline and /api/chat/ask endpoint.
Verifies all 5 core requirements:
1. A relevant question produces a grounded answer.
2. An unrelated question does not produce a fabricated answer.
3. Multiple relevant chunks can be combined.
4. Selected document filters work.
5. Another user's documents cannot be retrieved.
"""
import math
from uuid import uuid4
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from app.models.chat import (
    ChatAskRequest,
    ChatAskResponse,
)
from app.models.search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
)
from app.services.rag import (
    RAGOrchestratorService,
    default_rag_service,
    INSUFFICIENT_CONTEXT_MESSAGE,
)
from app.services.search import (
    SemanticSearchService,
    default_semantic_search_service,
)
from app.services.generation import (
    GeminiGenerationService,
    default_generation_service,
    GenerateAnswerResponse,
)
from app.services.chunk_repository import ChunkRepository, _in_memory_chunks_store
from app.routes.chat import ask_chat_endpoint
from tests.test_semantic_search import (
    JAVA_CHUNK_1,
    JAVA_CHUNK_2,
    JAVA_CHUNK_3,
    make_semantic_vector,
)


@pytest.fixture
def rag_pipeline_setup():
    """Sets up an isolated RAG environment with indexed multi-page document chunks."""
    user_id = uuid4()
    doc_1_id = uuid4()
    doc_2_id = uuid4()
    doc_1_name = "Java_Inheritance_and_Polymorphism_Guide.pdf"
    doc_2_name = "Design_Patterns_Quick_Reference.docx"

    # Simulated embeddings for concepts
    v1 = make_semantic_vector(["inheritance", "oop", "parent", "child", "class", "is-a"])
    v2 = make_semantic_vector(["child", "parent", "extends", "super", "override", "animal", "dog"])
    v3 = make_semantic_vector(["purpose", "extending", "polymorphism", "modularity", "base", "class"])
    v4 = make_semantic_vector(["design", "patterns", "singleton", "factory"])

    # Document 1 chunks (3 pages)
    doc_1_key = f"{user_id}:{doc_1_id}"
    _in_memory_chunks_store[doc_1_key] = [
        {
            "id": str(uuid4()),
            "document_id": str(doc_1_id),
            "document_name": doc_1_name,
            "user_id": str(user_id),
            "chunk_index": 0,
            "page_number": 1,
            "content": JAVA_CHUNK_1,
            "embedding": v1,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_1_id),
            "document_name": doc_1_name,
            "user_id": str(user_id),
            "chunk_index": 1,
            "page_number": 2,
            "content": JAVA_CHUNK_2,
            "embedding": v2,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_1_id),
            "document_name": doc_1_name,
            "user_id": str(user_id),
            "chunk_index": 2,
            "page_number": 3,
            "content": JAVA_CHUNK_3,
            "embedding": v3,
        },
    ]

    # Document 2 chunks (Design patterns)
    doc_2_key = f"{user_id}:{doc_2_id}"
    _in_memory_chunks_store[doc_2_key] = [
        {
            "id": str(uuid4()),
            "document_id": str(doc_2_id),
            "document_name": doc_2_name,
            "user_id": str(user_id),
            "chunk_index": 0,
            "page_number": 1,
            "content": "The Singleton pattern ensures a class has only one instance and provides a global point of access to it.",
            "embedding": v4,
        }
    ]

    # Mock query embedding hook
    mock_embedding_service = MagicMock()
    mock_embedding_service.model = "text-embedding-004"
    mock_embedding_service.dimension = 768

    def mock_embed_text(text, task_type="RETRIEVAL_QUERY"):
        words = text.lower().replace("?", "").replace(".", "").split()
        return make_semantic_vector(words)

    mock_embedding_service.embed_text.side_effect = mock_embed_text

    # Search service configured with offline in-memory repository
    repo = ChunkRepository(supabase_url="https://mock.supabase.co", supabase_key="")
    search_service = SemanticSearchService(
        embedding_service=mock_embedding_service,
        chunk_repository=repo,
    )

    # Generation service
    gen_service = GeminiGenerationService(
        api_key="mock_key",
        model_name="gemini-2.5-flash",
        temperature=0.2,
    )

    rag_service = RAGOrchestratorService(
        search_service=search_service,
        generation_service=gen_service,
    )

    return {
        "rag_service": rag_service,
        "search_service": search_service,
        "gen_service": gen_service,
        "user_id": user_id,
        "doc_1_id": doc_1_id,
        "doc_2_id": doc_2_id,
        "doc_1_name": doc_1_name,
        "doc_2_name": doc_2_name,
    }


# ==============================================================================
# Verification 1: A relevant question produces a grounded answer
# ==============================================================================

def test_1_relevant_question_produces_grounded_answer(rag_pipeline_setup):
    setup = rag_pipeline_setup
    rag_svc = setup["rag_service"]
    user_id = setup["user_id"]

    mock_llm_response = MagicMock()
    mock_llm_response.text = (
        "Inheritance in Java is a core principle of OOP where a subclass acquires attributes and methods "
        "from a superclass. It establishes an 'is-a' relationship and improves code reuse (Java_Inheritance_and_Polymorphism_Guide.pdf, Page 1)."
    )
    mock_llm_response.usage_metadata.total_token_count = 120
    mock_llm_response.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_llm_response
        rag_svc.generation_service._client = mock_c

        request = ChatAskRequest(
            question="What is inheritance in Java and what relationship does it establish?",
            match_threshold=0.3,
            limit=3,
        )
        response = rag_svc.ask(request=request, user_id=user_id)

        assert response.retrieval_metadata.grounded is True
        assert response.retrieval_metadata.is_insufficient_context is False
        assert "is-a" in response.answer
        assert len(response.sources) >= 1
        assert setup["doc_1_name"] in response.document_names
        assert 1 in response.page_numbers
        assert response.retrieval_metadata.total_chunks_retrieved >= 1


# ==============================================================================
# Verification 2: An unrelated question does not produce a fabricated answer
# ==============================================================================

def test_2_unrelated_question_does_not_produce_fabricated_answer(rag_pipeline_setup):
    setup = rag_pipeline_setup
    rag_svc = setup["rag_service"]
    user_id = setup["user_id"]

    # When query is unrelated, vector search finds 0 matches above threshold
    request = ChatAskRequest(
        question="How do I bake chocolate chip cookies from scratch with brown sugar?",
        match_threshold=0.5,
        limit=5,
    )
    response = rag_svc.ask(request=request, user_id=user_id)

    assert response.retrieval_metadata.is_insufficient_context is True
    assert response.retrieval_metadata.grounded is False
    assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(response.sources) == 0
    assert len(response.document_names) == 0
    assert len(response.page_numbers) == 0
    assert response.retrieval_metadata.total_chunks_retrieved == 0


# ==============================================================================
# Verification 3: Multiple relevant chunks can be combined
# ==============================================================================

def test_3_multiple_relevant_chunks_combined(rag_pipeline_setup):
    setup = rag_pipeline_setup
    rag_svc = setup["rag_service"]
    user_id = setup["user_id"]

    # Question touches both definition (Chunk 0 / Page 1) and implementation extends/super (Chunk 1 / Page 2)
    mock_llm_response = MagicMock()
    mock_llm_response.text = (
        "In Java, inheritance allows a child class to inherit fields and methods from a parent class "
        "establishing an 'is-a' relationship (Page 1). The child class implements this using the 'extends' "
        "keyword, accesses superclass members via 'super', and can override methods with @Override (Page 2)."
    )
    mock_llm_response.usage_metadata.total_token_count = 190
    mock_llm_response.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_llm_response
        rag_svc.generation_service._client = mock_c

        request = ChatAskRequest(
            question="Explain how child classes extend parent classes in Java OOP inheritance.",
            match_threshold=0.25,
            limit=5,
        )
        response = rag_svc.ask(request=request, user_id=user_id)

        assert response.retrieval_metadata.grounded is True
        assert response.retrieval_metadata.total_chunks_retrieved >= 2
        # Multiple pages combined
        assert 1 in response.page_numbers
        assert 2 in response.page_numbers
        assert len(response.sources) >= 2


# ==============================================================================
# Verification 4: Selected document filters work
# ==============================================================================

def test_4_selected_document_filters_work(rag_pipeline_setup):
    setup = rag_pipeline_setup
    rag_svc = setup["rag_service"]
    user_id = setup["user_id"]
    doc_1_id = setup["doc_1_id"]
    doc_2_id = setup["doc_2_id"]

    mock_llm_response = MagicMock()
    mock_llm_response.text = "Grounded answer from Document 1 only."
    mock_llm_response.usage_metadata.total_token_count = 80
    mock_llm_response.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_llm_response
        rag_svc.generation_service._client = mock_c

        # Case A: Filter strictly to doc_1_id
        req_doc1 = ChatAskRequest(
            question="What is inheritance in Java?",
            document_ids=[doc_1_id],
            match_threshold=0.2,
        )
        res_doc1 = rag_svc.ask(request=req_doc1, user_id=user_id)
        assert res_doc1.retrieval_metadata.total_chunks_retrieved >= 1
        for s in res_doc1.sources:
            assert s.document_id == doc_1_id

        # Case B: Filter to doc_2_id (which only has Design Patterns, no Java Inheritance)
        req_doc2 = ChatAskRequest(
            question="What is inheritance in Java?",
            document_ids=[doc_2_id],
            match_threshold=0.6,
        )
        res_doc2 = rag_svc.ask(request=req_doc2, user_id=user_id)
        # Should return insufficient context because doc_2 doesn't contain inheritance
        assert res_doc2.retrieval_metadata.is_insufficient_context is True
        assert res_doc2.answer == INSUFFICIENT_CONTEXT_MESSAGE


# ==============================================================================
# Verification 5: Another user's documents cannot be retrieved
# ==============================================================================

def test_5_another_users_documents_cannot_be_retrieved(rag_pipeline_setup):
    setup = rag_pipeline_setup
    rag_svc = setup["rag_service"]
    owner_user_id = setup["user_id"]
    attacker_user_id = uuid4()

    # Attacker tries to query the owner's documents
    attacker_request = ChatAskRequest(
        question="What is inheritance in Java and what relationship does it establish?",
        match_threshold=0.1,  # even with very low threshold
        limit=10,
    )

    attacker_response = rag_svc.ask(request=attacker_request, user_id=attacker_user_id)

    # Must find 0 chunks and return insufficient context - absolutely no data leaked!
    assert attacker_response.retrieval_metadata.total_chunks_retrieved == 0
    assert attacker_response.retrieval_metadata.is_insufficient_context is True
    assert len(attacker_response.sources) == 0
    assert len(attacker_response.document_names) == 0
    assert attacker_response.answer == INSUFFICIENT_CONTEXT_MESSAGE


# ==============================================================================
# Verification 6: Deduplication avoids sending duplicate chunks to Gemini
# ==============================================================================

def test_chunk_deduplication():
    mock_search = MagicMock()
    mock_gen = MagicMock()
    rag_svc = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen)

    doc_id = uuid4()
    # Duplicate item with same document_id and chunk_index
    duplicate_results = [
        SearchResultItem(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="Test.pdf",
            page_number=1,
            chunk_index=0,
            content="Duplicate chunk content",
            similarity=0.9,
        ),
        SearchResultItem(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="Test.pdf",
            page_number=1,
            chunk_index=0,
            content="Duplicate chunk content",
            similarity=0.88,
        ),
    ]

    mock_search.search.return_value = SemanticSearchResponse(
        query="test query",
        total_results=2,
        match_threshold=0.3,
        embedding_model="text-embedding-004",
        results=duplicate_results,
    )

    mock_gen.generate_answer.return_value = GenerateAnswerResponse(
        question="test query",
        answer="Grounded answer",
        sources=[],
        model="gemini-2.5-flash",
        grounded=True,
        is_insufficient_context=False,
    )

    req = ChatAskRequest(question="test query")
    res = rag_svc.ask(req, user_id=uuid4())

    # Only 1 unique chunk sent to generation service
    assert mock_gen.generate_answer.called
    context_sent = mock_gen.generate_answer.call_args.kwargs["context"]
    assert len(context_sent) == 1
    assert len(res.sources) == 1


# ==============================================================================
# Verification 7: API Endpoint POST /api/chat/ask
# ==============================================================================

def test_chat_ask_endpoint_direct(rag_pipeline_setup):
    setup = rag_pipeline_setup
    user_id = setup["user_id"]

    mock_llm_response = MagicMock()
    mock_llm_response.text = "Inheritance allows subclasses to acquire methods and fields."
    mock_llm_response.usage_metadata.total_token_count = 95
    mock_llm_response.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_llm_response

        with patch.object(default_generation_service, "_client", mock_c):
            with patch.object(default_generation_service, "api_key", "test_key"):
                with patch.object(default_semantic_search_service, "embedding_service", setup["search_service"].embedding_service):
                    with patch.object(default_semantic_search_service.chunk_repository, "is_configured", return_value=False):
                        req = ChatAskRequest(
                            question="What is inheritance in Java?",
                            conversation_id=uuid4(),
                            match_threshold=0.2,
                            limit=3,
                        )

                        res = ask_chat_endpoint(request=req, current_user=user_id)
                        assert res.question == "What is inheritance in Java?"
                        assert "Inheritance allows subclasses" in res.answer
                        assert res.retrieval_metadata.grounded is True
                        assert len(res.sources) >= 1
                        assert setup["doc_1_name"] in res.document_names
