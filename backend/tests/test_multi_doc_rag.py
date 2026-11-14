"""Unit and integration tests for Multi-Document RAG and Day 3 Evaluation Test Cases."""
import pytest
import time
from uuid import uuid4, UUID
from unittest.mock import MagicMock
from fastapi import HTTPException

from app.models.chat import ChatAskRequest, ChatAskResponse
from app.models.search import SearchResultItem, SemanticSearchResponse
from app.models.generation import GenerateAnswerResponse, ContextChunk
from app.services.rag import RAGOrchestratorService, RAGOrchestrationError, INSUFFICIENT_CONTEXT_MESSAGE
from app.services.generation import (
    GeminiGenerationService,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiAPIKeyMissingError,
)
from app.services.conversation_repository import (
    ConversationRepository,
    _in_memory_conversations_store,
    _in_memory_messages_store,
)
from app.routes.chat import ask_chat_endpoint


@pytest.fixture(autouse=True)
def clean_stores():
    _in_memory_conversations_store.clear()
    _in_memory_messages_store.clear()
    yield
    _in_memory_conversations_store.clear()
    _in_memory_messages_store.clear()


class TestDay3MultiDocumentRAG:
    """Covers all 8 evaluation test cases specified for Day 3."""

    # --------------------------------------------------------------------------
    # TEST CASE 1: Question directly answered by one document
    # --------------------------------------------------------------------------
    def test_case_1_single_document_direct_answer(self):
        user_id = uuid4()
        doc_a_id = uuid4()

        mock_search = MagicMock()
        mock_search.search.return_value = SemanticSearchResponse(
            query="What is inheritance in Java?",
            total_results=1,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=doc_a_id,
                    document_name="Java_Inheritance.pdf",
                    chunk_index=0,
                    content="Inheritance in Java allows classes to inherit fields and methods from a superclass.",
                    page_number=1,
                    similarity=0.91,
                )
            ],
        )

        mock_gen = MagicMock()
        mock_gen.generate_answer.return_value = GenerateAnswerResponse(
            question="What is inheritance in Java?",
            answer="Inheritance in Java allows a subclass to acquire attributes and methods from an existing superclass [Source 1].",
            sources=[],
            model="gemini-2.5-flash",
            grounded=True,
            is_insufficient_context=False,
            tokens_used=40,
        )

        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen, conversation_repository=repo)

        start = time.perf_counter()
        req = ChatAskRequest(question="What is inheritance in Java?")
        res = rag.ask(req, user_id)
        latency_ms = (time.perf_counter() - start) * 1000

        assert res.retrieval_metadata.grounded is True
        assert res.retrieval_metadata.is_insufficient_context is False
        assert len(res.sources) == 1
        assert res.document_names == ["Java_Inheritance.pdf"]
        assert res.page_numbers == [1]
        assert "[Source 1]" in res.answer
        assert latency_ms > 0

    # --------------------------------------------------------------------------
    # TEST CASE 2: Question requiring two documents (Multi-Document Synthesis)
    # --------------------------------------------------------------------------
    def test_case_2_multi_document_synthesis(self):
        user_id = uuid4()
        doc_a_id = uuid4()
        doc_b_id = uuid4()

        # Doc A = Inheritance, Doc B = Polymorphism
        mock_search = MagicMock()
        mock_search.search.return_value = SemanticSearchResponse(
            query="How are inheritance and polymorphism related in Java?",
            total_results=2,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=doc_a_id,
                    document_name="Java_Inheritance.pdf",
                    chunk_index=0,
                    content="Inheritance establishes an is-a relationship and allows code reusability across class hierarchies.",
                    page_number=2,
                    similarity=0.89,
                ),
                SearchResultItem(
                    document_id=doc_b_id,
                    document_name="Java_Polymorphism.pdf",
                    chunk_index=0,
                    content="Polymorphism in Java builds upon inheritance, enabling a subclass to override superclass methods to achieve runtime dynamic dispatch.",
                    page_number=4,
                    similarity=0.87,
                ),
            ],
        )

        mock_gen = MagicMock()
        mock_gen.generate_answer.return_value = GenerateAnswerResponse(
            question="How are inheritance and polymorphism related in Java?",
            answer=(
                "In Java, inheritance and polymorphism are closely linked OOP pillars:\n\n"
                "1. **Inheritance** establishes an 'is-a' hierarchy and promotes code reusability [Source 1].\n"
                "2. **Polymorphism** builds on this inheritance hierarchy by allowing child classes to override superclass methods for dynamic method dispatch at runtime [Source 2]."
            ),
            sources=[],
            model="gemini-2.5-flash",
            grounded=True,
            is_insufficient_context=False,
            tokens_used=75,
        )

        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen, conversation_repository=repo)

        req = ChatAskRequest(question="How are inheritance and polymorphism related in Java?")
        res = rag.ask(req, user_id)

        # Verify multi-document synthesis
        assert len(res.sources) == 2
        assert len(res.document_names) == 2
        assert "Java_Inheritance.pdf" in res.document_names
        assert "Java_Polymorphism.pdf" in res.document_names
        assert res.page_numbers == [2, 4]
        assert "[Source 1]" in res.answer
        assert "[Source 2]" in res.answer
        assert res.retrieval_metadata.grounded is True

    # --------------------------------------------------------------------------
    # TEST CASE 3: Question requiring a follow-up
    # --------------------------------------------------------------------------
    def test_case_3_contextual_follow_up(self):
        user_id = uuid4()
        doc_id = uuid4()

        mock_search = MagicMock()
        mock_search.search.return_value = SemanticSearchResponse(
            query="Polymorphism in Java",
            total_results=1,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=doc_id,
                    document_name="Java_Polymorphism.pdf",
                    chunk_index=0,
                    content="Polymorphism allows one interface to control access to a general class of actions.",
                    page_number=1,
                    similarity=0.85,
                )
            ],
        )

        mock_gen = MagicMock()
        mock_gen.generate_answer.return_value = GenerateAnswerResponse(
            question="Explain that in simpler terms.",
            answer="In simpler terms, polymorphism lets one object act like another when needed [Source 1].",
            sources=[],
            model="gemini-2.5-flash",
            grounded=True,
            is_insufficient_context=False,
            tokens_used=35,
        )

        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen, conversation_repository=repo)

        # Initial turn
        conv = repo.create_conversation(user_id=user_id, title="Polymorphism Discussion")
        repo.add_message(conv.id, user_id, role="user", content="What is polymorphism in Java?")
        repo.add_message(conv.id, user_id, role="assistant", content="Polymorphism enables dynamic method binding.")

        # Follow-up turn
        req = ChatAskRequest(question="Explain that in simpler terms.", conversation_id=conv.id)
        res = rag.ask(req, user_id)

        assert res.conversation_id == conv.id
        # Verify history passed to generator
        assert mock_gen.generate_answer.call_args[1].get("history") is not None
        assert len(mock_gen.generate_answer.call_args[1]["history"]) >= 2
        assert res.retrieval_metadata.grounded is True

    # --------------------------------------------------------------------------
    # TEST CASE 4: Question not covered by uploaded documents
    # --------------------------------------------------------------------------
    def test_case_4_insufficient_context_not_covered(self):
        user_id = uuid4()

        # No chunks match query above threshold
        mock_search = MagicMock()
        mock_search.search.return_value = SemanticSearchResponse(
            query="How do I bake chocolate chip cookies?",
            total_results=0,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[],
        )

        mock_gen = MagicMock()
        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen, conversation_repository=repo)

        req = ChatAskRequest(question="How do I bake chocolate chip cookies?")
        res = rag.ask(req, user_id)

        assert res.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert res.retrieval_metadata.is_insufficient_context is True
        assert res.retrieval_metadata.grounded is False
        assert len(res.sources) == 0
        assert mock_gen.generate_answer.called is False  # LLM not called when 0 chunks found

    # --------------------------------------------------------------------------
    # TEST CASE 5: Empty question
    # --------------------------------------------------------------------------
    def test_case_5_empty_question_validation(self):
        # Empty string
        with pytest.raises(ValueError):
            ChatAskRequest(question="")

        # Whitespace-only string
        with pytest.raises(ValueError):
            ChatAskRequest(question="     ")

        # Single character
        with pytest.raises(ValueError):
            ChatAskRequest(question="?")

    # --------------------------------------------------------------------------
    # TEST CASE 6: Invalid document selection
    # --------------------------------------------------------------------------
    def test_case_6_invalid_document_selection(self):
        user_id = uuid4()
        non_existent_doc_id = uuid4()

        mock_search = MagicMock()
        # Filter on non_existent_doc_id produces 0 results
        mock_search.search.return_value = SemanticSearchResponse(
            query="What is inheritance?",
            total_results=0,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[],
        )

        mock_gen = MagicMock()
        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen, conversation_repository=repo)

        req = ChatAskRequest(
            question="What is inheritance?",
            document_ids=[non_existent_doc_id],
        )
        res = rag.ask(req, user_id)

        # Handles safely without crashing or leaking other docs
        assert res.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert res.retrieval_metadata.is_insufficient_context is True
        assert len(res.sources) == 0

    # --------------------------------------------------------------------------
    # TEST CASE 7: Unauthorized document access
    # --------------------------------------------------------------------------
    def test_case_7_unauthorized_document_access(self):
        owner_id = uuid4()
        attacker_id = uuid4()
        owner_doc_id = uuid4()

        mock_search = MagicMock()
        # Search service rejects or returns empty for attacker targeting owner's doc
        def mock_search_impl(request, user_id):
            if user_id != owner_id:
                return SemanticSearchResponse(
                    query=request.query,
                    total_results=0,
                    match_threshold=request.match_threshold,
                    embedding_model="text-embedding-004",
                    results=[],
                )
            return SemanticSearchResponse(
                query=request.query,
                total_results=1,
                match_threshold=request.match_threshold,
                embedding_model="text-embedding-004",
                results=[
                    SearchResultItem(
                        document_id=owner_doc_id,
                        document_name="Owner_Secret.pdf",
                        chunk_index=0,
                        content="Owner secret contents.",
                        similarity=0.95,
                    )
                ],
            )
        mock_search.search.side_effect = mock_search_impl

        mock_gen = MagicMock()
        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen, conversation_repository=repo)

        # Attacker tries to query owner's document
        attacker_req = ChatAskRequest(
            question="What is the owner secret?",
            document_ids=[owner_doc_id],
        )
        res = rag.ask(attacker_req, attacker_id)

        assert res.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert len(res.sources) == 0
        assert "Owner_Secret.pdf" not in res.document_names

    # --------------------------------------------------------------------------
    # TEST CASE 8: Gemini API failure handling
    # --------------------------------------------------------------------------
    def test_case_8_gemini_api_failure_handling(self):
        user_id = uuid4()
        doc_id = uuid4()

        mock_search = MagicMock()
        mock_search.search.return_value = SemanticSearchResponse(
            query="Explain Java inheritance",
            total_results=1,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=doc_id,
                    document_name="java.pdf",
                    chunk_index=0,
                    content="Content here",
                    similarity=0.8,
                )
            ],
        )

        # 8.1 Rate limit error (HTTP 429)
        mock_gen_rate_limit = MagicMock()
        mock_gen_rate_limit.generate_answer.side_effect = GeminiRateLimitError("Rate limit exceeded")
        rag_rl = RAGOrchestratorService(search_service=mock_search, generation_service=mock_gen_rate_limit)

        with pytest.raises(GeminiRateLimitError):
            rag_rl.ask(ChatAskRequest(question="Explain Java inheritance"), user_id)

        # Verify endpoint maps to HTTP 429
        endpoint_req = ChatAskRequest(question="Explain Java inheritance")
        from unittest.mock import patch
        with patch("app.routes.chat.default_rag_service.ask", side_effect=GeminiRateLimitError("Rate limit exceeded")):
            with pytest.raises(HTTPException) as exc_rl:
                ask_chat_endpoint(request=endpoint_req, current_user=user_id)
            assert exc_rl.value.status_code == 429

        # Verify endpoint maps to HTTP 504
        with patch("app.routes.chat.default_rag_service.ask", side_effect=GeminiTimeoutError("Timeout occurred")):
            with pytest.raises(HTTPException) as exc_to:
                ask_chat_endpoint(request=endpoint_req, current_user=user_id)
            assert exc_to.value.status_code == 504

        # Verify endpoint maps to HTTP 503
        with patch("app.routes.chat.default_rag_service.ask", side_effect=GeminiAPIKeyMissingError("Missing key")):
            with pytest.raises(HTTPException) as exc_key:
                ask_chat_endpoint(request=endpoint_req, current_user=user_id)
            assert exc_key.value.status_code == 503
