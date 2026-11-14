"""End-to-end demonstration and verification of persistent conversation history and contextual follow-up questions."""
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import logging
from uuid import uuid4, UUID
from typing import List, Dict, Any

from app.models.chat import ChatAskRequest
from app.models.search import SearchResultItem, SemanticSearchResponse
from app.models.generation import GenerateAnswerResponse
from app.services.rag import RAGOrchestratorService
from app.services.conversation_repository import (
    ConversationRepository,
    default_conversation_repository,
    _in_memory_conversations_store,
    _in_memory_messages_store,
)
from app.models.conversation import generate_conversation_title

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo_conversations")


def run_demonstration():
    print("=" * 90)
    print("PERSONAL AI KNOWLEDGE ASSISTANT: PERSISTENT CONVERSATIONS & FOLLOW-UP DEMO")
    print("=" * 90)

    # 1. Setup isolated user repository
    user_id = uuid4()
    attacker_id = uuid4()
    doc_id = uuid4()

    print(f"\n[PHASE 1] INITIALIZING TEST USER SESSION")
    print(f"Authenticated User ID : {user_id}")
    print(f"Attacker User ID       : {attacker_id}")
    print(f"Knowledge Document ID  : {doc_id} (Java_Inheritance_Guide.pdf)")

    # Mock search and generation services for deterministic verification
    mock_search = MockSearchService(doc_id)
    mock_gen = MockGenerationService()
    repo = ConversationRepository(supabase_url="", supabase_key="")

    rag = RAGOrchestratorService(
        search_service=mock_search,
        generation_service=mock_gen,
        conversation_repository=repo,
    )

    # --------------------------------------------------------------------------
    # STEP 1: INITIAL QUESTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[STEP 1] INITIAL USER QUESTION -> AUTO TITLE & CONVERSATION CREATION")
    print("=" * 90)
    q1 = "What is inheritance in Java and what relationship does it establish?"
    print(f"User Question: \"{q1}\"")

    req1 = ChatAskRequest(question=q1)
    res1 = rag.ask(req1, user_id)

    conv_id = res1.conversation_id
    print(f"\nCreated Conversation ID: {conv_id}")
    print(f"AI Assistant Response  : \"{res1.answer}\"")
    print(f"Sources Cited          : {[s.source_id for s in res1.sources]}")

    # Verify conversation title
    conv_record = repo.get_conversation(conv_id, user_id)
    assert conv_record is not None, "Conversation must exist in database"
    print(f"Auto-Generated Title   : \"{conv_record.title}\"")
    assert "inheritance in Java" in conv_record.title

    # Verify message count
    msgs_step1 = repo.get_messages(conv_id, user_id)
    assert len(msgs_step1) == 2, f"Expected 2 messages (1 user, 1 assistant), got {len(msgs_step1)}"
    print(f"Persisted Messages     : {len(msgs_step1)} (User + Assistant)")
    print(">>> STEP 1 PASSED: Conversation created, auto-titled, and messages saved.")

    # --------------------------------------------------------------------------
    # STEP 2: CONTEXTUAL FOLLOW-UP 1 ("Explain that in simpler terms.")
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[STEP 2] CONTEXTUAL FOLLOW-UP 1 -> 'Explain that in simpler terms.'")
    print("=" * 90)
    q2 = "Explain that in simpler terms."
    print(f"User Follow-up: \"{q2}\"")
    print(f"Sending with Conversation ID: {conv_id}")

    req2 = ChatAskRequest(question=q2, conversation_id=conv_id)
    res2 = rag.ask(req2, user_id)

    print(f"\nAI Assistant Response  : \"{res2.answer}\"")
    print(f"Sources Cited          : {[s.source_id for s in res2.sources]}")

    # Verify that conversation history was passed to generation
    assert mock_gen.last_history is not None
    assert len(mock_gen.last_history) >= 2
    print(f"Preceding History Turns: {len(mock_gen.last_history)} messages provided to Gemini")
    print(f"Prior Context Topic    : \"{mock_gen.last_history[0]['content']}\"")

    msgs_step2 = repo.get_messages(conv_id, user_id)
    assert len(msgs_step2) == 4, f"Expected 4 messages, got {len(msgs_step2)}"
    print(f"Persisted Messages     : {len(msgs_step2)} (2 User, 2 Assistant)")
    print(">>> STEP 2 PASSED: Contextual follow-up resolved with preceding history.")

    # --------------------------------------------------------------------------
    # STEP 3: CONTEXTUAL FOLLOW-UP 2 ("Give me an example.")
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[STEP 3] CONTEXTUAL FOLLOW-UP 2 -> 'Give me an example.'")
    print("=" * 90)
    q3 = "Give me an example."
    print(f"User Follow-up: \"{q3}\"")
    print(f"Sending with Conversation ID: {conv_id}")

    req3 = ChatAskRequest(question=q3, conversation_id=conv_id)
    res3 = rag.ask(req3, user_id)

    print(f"\nAI Assistant Response  : \"{res3.answer}\"")
    print(f"Sources Cited          : {[s.source_id for s in res3.sources]}")

    msgs_step3 = repo.get_messages(conv_id, user_id)
    assert len(msgs_step3) == 6, f"Expected 6 messages, got {len(msgs_step3)}"
    print(f"Persisted Messages     : {len(msgs_step3)} (3 User, 3 Assistant)")
    print(">>> STEP 3 PASSED: Second follow-up processed in conversation thread.")

    # --------------------------------------------------------------------------
    # STEP 4: SIMULATE PAGE REFRESH & REOPEN CONVERSATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[STEP 4] SIMULATE PAGE REFRESH -> REOPEN CONVERSATION")
    print("=" * 90)
    print("Simulating browser refresh: loading user conversation list & details from database...")

    # Fetch conversation list for sidebar
    user_conversations = repo.list_conversations(user_id)
    assert len(user_conversations) >= 1
    found = [c for c in user_conversations if c.id == conv_id][0]
    print(f"Found in Sidebar List  : \"{found.title}\" (Updated: {found.updated_at.isoformat()})")

    # Fetch all messages for reopened conversation
    reopened_messages = repo.get_messages(conv_id, user_id)
    print(f"Reopened Message Count : {len(reopened_messages)}")
    assert len(reopened_messages) == 6

    print("\nReconstructed Conversation Thread:")
    for idx, m in enumerate(reopened_messages, 1):
        role_label = "USER" if m.role == "user" else "ASSISTANT"
        sources_info = f" ({len(m.sources)} sources cited)" if m.sources else ""
        print(f"  [{idx}] {role_label}: {m.content}{sources_info}")

    print(">>> STEP 4 PASSED: Conversation history completely reloaded with all citations.")

    # --------------------------------------------------------------------------
    # STEP 5: MULTI-TENANT ISOLATION & RLS VERIFICATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[STEP 5] ROW LEVEL SECURITY & USER OWNERSHIP VERIFICATION")
    print("=" * 90)
    print(f"Attacker ({attacker_id}) attempting to read User A's ({user_id}) conversation...")
    attacker_view = repo.get_conversation(conv_id, attacker_id)
    assert attacker_view is None, "Attacker must NOT be able to view another user's conversation"
    print("Attacker Conversation Access : BLOCKED (None returned)")

    attacker_messages = repo.get_messages(conv_id, attacker_id)
    assert len(attacker_messages) == 0, "Attacker must NOT see any messages"
    print("Attacker Messages Access     : BLOCKED (0 messages returned)")

    attacker_delete = repo.delete_conversation(conv_id, attacker_id)
    assert attacker_delete is False, "Attacker must NOT be able to delete another user's conversation"
    print("Attacker Delete Attempt      : BLOCKED (False returned)")
    print(">>> STEP 5 PASSED: Multi-tenant ownership verified. Attacker completely isolated.")

    # --------------------------------------------------------------------------
    # STEP 6: CONVERSATION DELETION BY OWNER
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[STEP 6] CONVERSATION DELETION BY OWNER")
    print("=" * 90)
    delete_result = repo.delete_conversation(conv_id, user_id)
    assert delete_result is True, "Owner should be able to delete their conversation"
    print(f"Owner deleted conversation {conv_id} : SUCCESS")

    assert repo.get_conversation(conv_id, user_id) is None
    assert len(repo.get_messages(conv_id, user_id)) == 0
    print("Verified: Conversation and messages permanently deleted.")
    print(">>> STEP 6 PASSED: Clean deletion verified.")

    print("\n" + "=" * 90)
    print("ALL CONVERSATION HISTORY & FOLLOW-UP VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 90)


class MockSearchService:
    def __init__(self, doc_id: UUID):
        self.doc_id = doc_id
        self.embedding_service = MagicMock()
        self.embedding_service.model = "text-embedding-004"

    def search(self, request, user_id):
        return SemanticSearchResponse(
            query=request.query,
            total_results=1,
            match_threshold=request.match_threshold,
            embedding_model="text-embedding-004",
            results=[
                SearchResultItem(
                    document_id=self.doc_id,
                    document_name="Java_Inheritance_Guide.pdf",
                    chunk_index=0,
                    content="Inheritance in Java allows a subclass to acquire attributes and methods from a superclass. It establishes an is-a relationship [Source 1].",
                    page_number=1,
                    similarity=0.88,
                )
            ]
        )


class MockGenerationService:
    def __init__(self):
        self.model = "gemini-2.5-flash"
        self.last_history = None

    def generate_answer(self, question, context, history=None, **kwargs):
        self.last_history = history
        q_lower = question.lower()
        if "simpler" in q_lower:
            answer = "In simpler terms, inheritance means a child class gets everything its parent class already has, like code reusability [Source 1]."
        elif "example" in q_lower:
            answer = "For example, `class Dog extends Animal` inherits all properties of `Animal` [Source 1]."
        else:
            answer = "Inheritance in Java establishes an **'is-a' relationship** that allows classes to inherit fields and methods from a base class [Source 1]."

        return GenerateAnswerResponse(
            question=question,
            answer=answer,
            sources=[],
            model=self.model,
            grounded=True,
            is_insufficient_context=False,
            tokens_used=40,
        )


if __name__ == "__main__":
    from unittest.mock import MagicMock
    run_demonstration()
