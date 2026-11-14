"""Unit and integration tests for Cross-Encoder Re-Ranking Service and RAG integration."""
from uuid import uuid4
from unittest.mock import MagicMock, patch
import pytest

from app.models.search import (
    SearchResultItem,
    ReRankRequest,
    ReRankResponse,
    SemanticSearchResponse,
)
from app.models.chat import ChatAskRequest
from app.services.reranker import (
    ReRankingService,
    LightweightCrossAttentionBackend,
)
from app.services.rag import RAGOrchestratorService
from app.services.conversation_repository import ConversationRepository
from app.routes.search import rerank_candidates


@pytest.fixture
def sample_candidates():
    doc_id = uuid4()
    return [
        # Candidate 0: Mentions general topics (rank 1 initially in similarity)
        SearchResultItem(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="general_overview.pdf",
            chunk_index=0,
            content="Object-oriented programming languages provide class inheritance, interfaces, and modular programming structures.",
            similarity=0.88,
            rrf_score=0.032,
            page_number=1,
        ),
        # Candidate 1: Contains the exact technical answer (rank 2 initially)
        SearchResultItem(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="jvm_internals.pdf",
            chunk_index=1,
            content="In Java polymorphic method dispatch, the JVM uses the invokevirtual bytecode instruction and resolves the vtable index at runtime.",
            similarity=0.76,
            rrf_score=0.018,
            page_number=14,
        ),
        # Candidate 2: Completely irrelevant chunk (rank 3 initially)
        SearchResultItem(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="unrelated_recipes.txt",
            chunk_index=2,
            content="Bake the cookies at 350 degrees Fahrenheit for 12 minutes until golden brown.",
            similarity=0.45,
            rrf_score=0.010,
            page_number=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Cross-Encoder Relevance Scoring Tests
# ---------------------------------------------------------------------------

def test_cross_encoder_scoring_elevates_relevant_passage(sample_candidates):
    """
    Verifies that the cross-encoder scores the specific answer chunk higher
    than the general chunk, elevating it from rank 2 to rank 1.
    """
    service = ReRankingService()
    query = "How does Java resolve polymorphic method dispatch with invokevirtual?"

    res = service.rerank(
        query=query,
        candidates=sample_candidates,
        top_k=3,
    )

    assert res.total_returned == 3
    # Candidate 1 (vtable / invokevirtual) should now be #1
    top_item = res.results[0]
    assert "invokevirtual" in top_item.content
    assert top_item.rerank_rank == 1
    assert top_item.initial_rank == 2
    assert top_item.rerank_score is not None

    # Irrelevant recipe chunk should have lowest score
    assert res.results[-1].document_name == "unrelated_recipes.txt"


def test_cross_encoder_preserves_all_metadata(sample_candidates):
    """Verifies that chunk_id, document_id, page_number, and citations are preserved."""
    service = ReRankingService()
    query = "Java polymorphism"

    res = service.rerank(query=query, candidates=sample_candidates, top_k=2)

    for item in res.results:
        assert item.chunk_id is not None
        assert item.document_id is not None
        assert item.document_name is not None
        assert item.page_number is not None
        assert item.content is not None
        assert item.similarity is not None
        assert item.rerank_score is not None
        assert item.rerank_rank is not None
        assert item.initial_rank is not None


def test_score_threshold_filters_out_irrelevant_chunks(sample_candidates):
    """Verifies that score_threshold filters out chunks with relevance below the cutoff."""
    service = ReRankingService()
    query = "invokevirtual polymorphic vtable dispatch"

    # Set threshold high enough to discard the recipe chunk
    res = service.rerank(
        query=query,
        candidates=sample_candidates,
        top_k=5,
        score_threshold=0.0,
    )

    # Both candidate 0 (general OOP) and candidate 2 (recipe) are filtered out by threshold 0.0
    doc_names = [item.document_name for item in res.results]
    assert "unrelated_recipes.txt" not in doc_names
    assert len(res.results) == 1
    assert "invokevirtual" in res.results[0].content


def test_batching_execution():
    """Verifies batching logic with multiple candidates."""
    backend = LightweightCrossAttentionBackend()
    query = "distributed consensus algorithm"
    passages = [
        f"Passage {i} discussing distributed consensus, Paxos and Raft protocols in multi-node clusters."
        for i in range(25)
    ]

    scores = backend.score_pairs(query=query, passages=passages, batch_size=8)
    assert len(scores) == 25
    assert all(isinstance(s, float) for s in scores)


def test_reranker_handles_empty_candidates():
    """Verifies reranker behaves gracefully when candidate list is empty."""
    service = ReRankingService()
    res = service.rerank(query="empty test", candidates=[], top_k=5)
    assert res.total_candidates == 0
    assert res.total_returned == 0
    assert res.results == []


def test_reranker_fallback_on_model_error(sample_candidates):
    """Verifies that if inference raises an error, it falls back to first-stage order without crashing."""
    service = ReRankingService()
    service.backend = MagicMock()
    service.backend.score_pairs.side_effect = RuntimeError("GPU out of memory")

    res = service.rerank(query="test", candidates=sample_candidates, top_k=2)
    # Should fall back cleanly
    assert res.total_returned == 2
    assert res.results[0].document_name == "general_overview.pdf"  # initial rank 1 preserved


def test_reranking_disabled_flag(sample_candidates):
    """Verifies that when enabled=False, original order is preserved."""
    service = ReRankingService(enabled=False)
    res = service.rerank(query="invokevirtual", candidates=sample_candidates, top_k=3)

    assert res.total_returned == 3
    assert res.results[0].document_name == "general_overview.pdf"  # did not reorder


# ---------------------------------------------------------------------------
# 2. RAG Pipeline Integration Tests
# ---------------------------------------------------------------------------

def test_rag_pipeline_integrates_reranking():
    """Verifies that RAG pipeline retrieves a larger candidate pool and applies re-ranking."""
    mock_search = MagicMock()
    doc_id = uuid4()
    candidate_1 = SearchResultItem(
        chunk_id=uuid4(),
        document_id=doc_id,
        document_name="general.pdf",
        chunk_index=0,
        content="General programming concepts.",
        similarity=0.95,
        page_number=1,
    )
    candidate_2 = SearchResultItem(
        chunk_id=uuid4(),
        document_id=doc_id,
        document_name="specific.pdf",
        chunk_index=1,
        content="Specific answer: Java uses invokevirtual opcode for polymorphic method dispatch.",
        similarity=0.75,
        page_number=5,
    )

    mock_search.search.return_value = SemanticSearchResponse(
        query="invokevirtual polymorphic method dispatch",
        total_results=2,
        match_threshold=0.3,
        embedding_model="text-embedding-004",
        results=[candidate_1, candidate_2],
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = MagicMock(
        answer="The JVM uses invokevirtual for polymorphic dispatch [Source 1].",
        sources=[],
    )
    mock_gen.validate_and_sanitize_citations.return_value = (
        "The JVM uses invokevirtual for polymorphic dispatch [Source 1].",
        ["Source 1"],
    )

    reranker = ReRankingService()
    in_memory_repo = ConversationRepository(supabase_url="", supabase_key="")

    rag = RAGOrchestratorService(
        search_service=mock_search,
        generation_service=mock_gen,
        conversation_repository=in_memory_repo,
        reranker_service=reranker,
        use_reranker=True,
    )

    user_id = uuid4()
    req = ChatAskRequest(question="What opcode does Java use for polymorphic method dispatch?", limit=2)
    response = rag.ask(req, user_id=user_id)

    assert response.answer is not None
    # Verify that the generated prompt received the re-ranked top chunk (specific.pdf)
    call_args = mock_gen.generate_answer.call_args[1]
    context_chunks = call_args["context"]
    assert len(context_chunks) > 0
    # Top chunk should be specific.pdf because of cross-encoder re-ranking!
    assert "invokevirtual" in context_chunks[0].content


# ---------------------------------------------------------------------------
# 3. API Route Tests
# ---------------------------------------------------------------------------

def test_api_rerank_endpoint_direct(sample_candidates):
    """Verifies POST /api/search/rerank route endpoint."""
    user_id = uuid4()
    req = ReRankRequest(
        query="invokevirtual polymorphic method dispatch",
        candidates=sample_candidates,
        top_k=2,
    )

    response = rerank_candidates(request=req, current_user=user_id)
    assert response.total_candidates == 3
    assert response.total_returned == 2
    assert response.results[0].rerank_rank == 1
    assert "invokevirtual" in response.results[0].content
