"""Comprehensive tests for persistent conversation history and contextual follow-up questions."""
import pytest
from uuid import uuid4, UUID
from unittest.mock import MagicMock
from datetime import datetime, timezone

from app.models.conversation import (
    generate_conversation_title,
    ConversationCreate,
    ConversationUpdate,
    ConversationRecord,
    ChatMessageRecord,
)
from app.services.conversation_repository import (
    ConversationRepository,
    _in_memory_conversations_store,
    _in_memory_messages_store,
)
from app.routes.conversations import (
    list_user_conversations,
    create_conversation,
    get_conversation_details,
    update_conversation_title,
    delete_user_conversation,
)
from app.models.chat import ChatAskRequest
from app.models.search import SearchResultItem, SemanticSearchResponse
from app.models.generation import GenerateAnswerResponse
from app.services.rag import RAGOrchestratorService
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def clear_in_memory_stores():
    """Clear in-memory stores before each test."""
    _in_memory_conversations_store.clear()
    _in_memory_messages_store.clear()
    yield
    _in_memory_conversations_store.clear()
    _in_memory_messages_store.clear()


class TestConversationModelsAndUtils:
    def test_generate_conversation_title_short(self):
        title = generate_conversation_title("What is inheritance in Java?")
        assert title == "What is inheritance in Java?"

    def test_generate_conversation_title_long(self):
        long_q = "Can you explain in extensive and comprehensive detail how inheritance works in Java with constructors and polymorphism?"
        title = generate_conversation_title(long_q, max_length=45)
        assert len(title) <= 50
        assert title.endswith("...")

    def test_generate_conversation_title_empty(self):
        assert generate_conversation_title("") == "New Conversation"
        assert generate_conversation_title("   ") == "New Conversation"


class TestConversationRepository:
    def test_create_and_get_conversation(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()
        conv = repo.create_conversation(user_id=user_id, title="Java Study Session")

        assert conv.user_id == user_id
        assert conv.title == "Java Study Session"
        assert conv.message_count == 0

        # Retrieve
        fetched = repo.get_conversation(conv.id, user_id)
        assert fetched is not None
        assert fetched.id == conv.id
        assert fetched.title == "Java Study Session"

    def test_user_isolation_get_conversation(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_a = uuid4()
        user_b = uuid4()

        conv_a = repo.create_conversation(user_id=user_a, title="User A Private Chat")
        
        # User B cannot access User A's conversation
        assert repo.get_conversation(conv_a.id, user_b) is None

    def test_list_conversations_ordered(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()
        other_user = uuid4()

        conv1 = repo.create_conversation(user_id=user_id, title="First Chat")
        conv2 = repo.create_conversation(user_id=user_id, title="Second Chat")
        repo.create_conversation(user_id=other_user, title="Other User Chat")

        user_convs = repo.list_conversations(user_id)
        assert len(user_convs) == 2
        titles = [c.title for c in user_convs]
        assert "First Chat" in titles
        assert "Second Chat" in titles

    def test_update_conversation_title(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()
        conv = repo.create_conversation(user_id=user_id, title="Old Title")

        updated = repo.update_conversation_title(conv.id, user_id, "New Renamed Title")
        assert updated is not None
        assert updated.title == "New Renamed Title"

        # Another user cannot update title
        other_user = uuid4()
        assert repo.update_conversation_title(conv.id, other_user, "Hacked Title") is None

    def test_delete_conversation(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()
        conv = repo.create_conversation(user_id=user_id, title="To Be Deleted")

        # Add message
        repo.add_message(conv.id, user_id, role="user", content="Hello")
        assert repo.get_messages_count(conv.id, user_id) == 1

        # Delete by owner
        assert repo.delete_conversation(conv.id, user_id) is True
        assert repo.get_conversation(conv.id, user_id) is None
        assert repo.get_messages(conv.id, user_id) == []

    def test_messages_persistence_and_ordering(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()
        conv = repo.create_conversation(user_id=user_id)

        msg1 = repo.add_message(conv.id, user_id, role="user", content="Question 1")
        msg2 = repo.add_message(conv.id, user_id, role="assistant", content="Answer 1", sources=[{"source_id": "Source 1"}])
        msg3 = repo.add_message(conv.id, user_id, role="user", content="Follow-up 2")

        messages = repo.get_messages(conv.id, user_id)
        assert len(messages) == 3
        assert messages[0].content == "Question 1"
        assert messages[1].content == "Answer 1"
        assert len(messages[1].sources) == 1
        assert messages[2].content == "Follow-up 2"

        # Test bounded recent history
        history = repo.get_recent_history(conv.id, user_id, limit=2)
        assert len(history) == 2
        assert history[0]["content"] == "Answer 1"
        assert history[1]["content"] == "Follow-up 2"


class TestConversationRoutes:
    def test_list_and_create_endpoints(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()

        # Create
        created = create_conversation(
            payload=ConversationCreate(title="Physics Notes"),
            current_user=user_id,
            repo=repo,
        )
        assert created.title == "Physics Notes"

        # List
        conv_list = list_user_conversations(current_user=user_id, repo=repo)
        assert len(conv_list) == 1
        assert conv_list[0].id == created.id

    def test_get_conversation_details_endpoint(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()

        conv = repo.create_conversation(user_id=user_id, title="History Chat")
        repo.add_message(conv.id, user_id, role="user", content="Who was Napoleon?")
        repo.add_message(conv.id, user_id, role="assistant", content="Napoleon was a French military general.")

        detail = get_conversation_details(
            conversation_id=conv.id,
            current_user=user_id,
            repo=repo,
        )
        assert detail.conversation.id == conv.id
        assert len(detail.messages) == 2

    def test_get_conversation_details_unauthorized_404(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_a = uuid4()
        user_b = uuid4()

        conv = repo.create_conversation(user_id=user_a, title="Private A")
        with pytest.raises(HTTPException) as exc:
            get_conversation_details(
                conversation_id=conv.id,
                current_user=user_b,
                repo=repo,
            )
        assert exc.value.status_code == 404

    def test_update_and_delete_endpoints(self):
        repo = ConversationRepository(supabase_url="", supabase_key="")
        user_id = uuid4()

        conv = repo.create_conversation(user_id=user_id, title="Old")
        updated = update_conversation_title(
            conversation_id=conv.id,
            payload=ConversationUpdate(title="Renamed"),
            current_user=user_id,
            repo=repo,
        )
        assert updated.title == "Renamed"

        res = delete_user_conversation(
            conversation_id=conv.id,
            current_user=user_id,
            repo=repo,
        )
        assert res["deleted"] is True


class TestRAGContextualFollowUp:
    def test_rag_creates_conversation_and_saves_history(self):
        user_id = uuid4()
        doc_id = uuid4()

        mock_search = MagicMock()
        mock_search.search.return_value = SemanticSearchResponse(
            query="What is inheritance in Java?",
            total_results=1,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=doc_id,
                    document_name="java_guide.pdf",
                    chunk_index=0,
                    content="Inheritance in Java allows a subclass to inherit attributes and methods [Source 1].",
                    page_number=1,
                    similarity=0.88,
                )
            ]
        )

        mock_gen = MagicMock()
        mock_gen.generate_answer.return_value = GenerateAnswerResponse(
            question="What is inheritance in Java?",
            answer="Inheritance in Java allows subclasses to reuse superclass attributes [Source 1].",
            sources=[],
            model="gemini-2.5-flash",
            grounded=True,
            is_insufficient_context=False,
            tokens_used=42,
        )

        repo = ConversationRepository(supabase_url="", supabase_key="")
        rag = RAGOrchestratorService(
            search_service=mock_search,
            generation_service=mock_gen,
            conversation_repository=repo,
        )

        # 1. Ask initial question (no conversation_id)
        req1 = ChatAskRequest(question="What is inheritance in Java?")
        res1 = rag.ask(req1, user_id)

        assert res1.conversation_id is not None
        conv_id = res1.conversation_id

        # Verify conversation was saved with auto-title
        conv = repo.get_conversation(conv_id, user_id)
        assert conv is not None
        assert "inheritance in Java" in conv.title

        # Verify messages saved
        messages = repo.get_messages(conv_id, user_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is inheritance in Java?"
        assert messages[1].role == "assistant"
        assert "subclasses to reuse" in messages[1].content

        # 2. Ask contextual follow-up question in the same conversation
        mock_search.search.return_value = SemanticSearchResponse(
            query="Inheritance in Java Explain that in simpler terms.",
            total_results=1,
            match_threshold=0.3,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=doc_id,
                    document_name="java_guide.pdf",
                    chunk_index=0,
                    content="Inheritance in Java allows a subclass to inherit attributes and methods [Source 1].",
                    page_number=1,
                    similarity=0.85,
                )
            ]
        )
        mock_gen.generate_answer.return_value = GenerateAnswerResponse(
            question="Explain that in simpler terms.",
            answer="In simpler terms, it means a child class gets everything its parent class has [Source 1].",
            sources=[],
            model="gemini-2.5-flash",
            grounded=True,
            is_insufficient_context=False,
            tokens_used=35,
        )

        req2 = ChatAskRequest(question="Explain that in simpler terms.", conversation_id=conv_id)
        res2 = rag.ask(req2, user_id)

        # Verify history passed to generation service
        assert mock_gen.generate_answer.call_args[1].get("history") is not None
        history_passed = mock_gen.generate_answer.call_args[1]["history"]
        assert len(history_passed) >= 2
        assert history_passed[0]["content"] == "What is inheritance in Java?"

        # Verify all 4 messages now in conversation history
        messages_after = repo.get_messages(conv_id, user_id)
        assert len(messages_after) == 4
        assert messages_after[2].role == "user"
        assert messages_after[2].content == "Explain that in simpler terms."
        assert messages_after[3].role == "assistant"
        assert "In simpler terms" in messages_after[3].content

        # 3. Ask second follow-up ("Give me an example.")
        mock_gen.generate_answer.return_value = GenerateAnswerResponse(
            question="Give me an example.",
            answer="For example, a `Dog` class can inherit from an `Animal` class [Source 1].",
            sources=[],
            model="gemini-2.5-flash",
            grounded=True,
            is_insufficient_context=False,
            tokens_used=30,
        )
        req3 = ChatAskRequest(question="Give me an example.", conversation_id=conv_id)
        res3 = rag.ask(req3, user_id)

        messages_final = repo.get_messages(conv_id, user_id)
        assert len(messages_final) == 6
        assert messages_final[4].content == "Give me an example."
        assert "Dog" in messages_final[5].content
