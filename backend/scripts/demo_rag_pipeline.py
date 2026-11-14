"""End-to-End RAG Pipeline Demonstration Script with Citation Grounding.
Verifies:
1. A relevant question produces a grounded answer with page/document citations.
2. Unrelated questions return an insufficient-context response (zero hallucination).
3. Multiple relevant chunks from different pages/documents are combined.
4. Selected document filters restrict search to specified documents.
5. Another user's documents cannot be accessed or retrieved (multi-tenant security).
6. Citations correspond to actual retrieved sources and fabricated references are stripped.
"""
import sys
import os
import json
from uuid import uuid4
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.chat import ChatAskRequest
from app.services.rag import (
    RAGOrchestratorService,
    INSUFFICIENT_CONTEXT_MESSAGE,
)
from app.services.search import SemanticSearchService
from app.services.generation import GeminiGenerationService
from app.services.chunk_repository import ChunkRepository, _in_memory_chunks_store
from tests.test_semantic_search import (
    JAVA_CHUNK_1,
    JAVA_CHUNK_2,
    JAVA_CHUNK_3,
    make_semantic_vector,
)


def run_rag_demonstration():
    print("=" * 90)
    print("PERSONAL AI KNOWLEDGE ASSISTANT: COMPLETE RAG PIPELINE DEMONSTRATION")
    print("=" * 90)

    # User 1: Authorized Document Owner
    owner_user_id = uuid4()
    doc_1_id = uuid4()
    doc_2_id = uuid4()
    doc_1_name = "Java_Inheritance_and_Polymorphism_Guide.pdf"
    doc_2_name = "System_Architecture_Overview.docx"

    # User 2: Attacker / Unauthorized Tenant
    attacker_user_id = uuid4()

    chunk_1_id = uuid4()
    chunk_2_id = uuid4()
    chunk_3_id = uuid4()
    chunk_4_id = uuid4()

    print(f"\n[PHASE 1] POPULATING USER KNOWLEDGE REPOSITORY")
    print(f"Owner User ID : {owner_user_id}")
    print(f"Document 1 ID : {doc_1_id} ({doc_1_name}) - PDF with verified page numbering")
    print(f"Document 2 ID : {doc_2_id} ({doc_2_name}) - DOCX unpaginated document")
    print("-" * 90)

    # Document 1: 3 Chunks across 3 pages
    v1 = make_semantic_vector(["inheritance", "oop", "parent", "child", "class", "is-a"])
    v2 = make_semantic_vector(["child", "parent", "extends", "super", "override", "animal", "dog"])
    v3 = make_semantic_vector(["purpose", "extending", "polymorphism", "modularity", "base", "class"])
    
    # Document 2: 1 Chunk (DOCX)
    v4 = make_semantic_vector(["system", "architecture", "microservices", "database", "supabase"])

    doc_1_key = f"{owner_user_id}:{doc_1_id}"
    _in_memory_chunks_store[doc_1_key] = [
        {
            "id": str(chunk_1_id),
            "document_id": str(doc_1_id),
            "document_name": doc_1_name,
            "user_id": str(owner_user_id),
            "chunk_index": 0,
            "page_number": 1,
            "content": JAVA_CHUNK_1,
            "embedding": v1,
        },
        {
            "id": str(chunk_2_id),
            "document_id": str(doc_1_id),
            "document_name": doc_1_name,
            "user_id": str(owner_user_id),
            "chunk_index": 1,
            "page_number": 2,
            "content": JAVA_CHUNK_2,
            "embedding": v2,
        },
        {
            "id": str(chunk_3_id),
            "document_id": str(doc_1_id),
            "document_name": doc_1_name,
            "user_id": str(owner_user_id),
            "chunk_index": 2,
            "page_number": 3,
            "content": JAVA_CHUNK_3,
            "embedding": v3,
        },
    ]

    doc_2_key = f"{owner_user_id}:{doc_2_id}"
    _in_memory_chunks_store[doc_2_key] = [
        {
            "id": str(chunk_4_id),
            "document_id": str(doc_2_id),
            "document_name": doc_2_name,
            "user_id": str(owner_user_id),
            "chunk_index": 0,
            "page_number": 1,
            "content": "The system architecture consists of a FastAPI backend service communicating with Supabase PostgreSQL and Gemini LLM.",
            "embedding": v4,
        }
    ]

    print(f"Indexed 4 chunks across 2 documents for User {owner_user_id}.")

    # Configure embedding hook for retrieval
    mock_embedding_service = MagicMock()
    mock_embedding_service.model = "text-embedding-004"
    mock_embedding_service.dimension = 768

    def mock_embed_text(text, task_type="RETRIEVAL_QUERY"):
        words = text.lower().replace("?", "").replace(".", "").split()
        return make_semantic_vector(words)

    mock_embedding_service.embed_text.side_effect = mock_embed_text

    # Setup RAG Orchestrator Service
    repo = ChunkRepository(supabase_url="https://mock.supabase.co", supabase_key="")
    search_service = SemanticSearchService(
        embedding_service=mock_embedding_service,
        chunk_repository=repo,
    )
    gen_service = GeminiGenerationService(
        api_key="mock_key",
        model_name="gemini-2.5-flash",
        temperature=0.2,
    )
    rag_service = RAGOrchestratorService(
        search_service=search_service,
        generation_service=gen_service,
    )

    # --------------------------------------------------------------------------
    # Verification 1: Relevant question produces a grounded answer with citations
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[TEST 1] RELEVANT QUESTION -> GROUNDED ANSWER WITH CITATIONS")
    print("=" * 90)
    q1 = "What is inheritance in Java and what relationship does it establish?"
    print(f"Query: \"{q1}\"")

    mock_resp_1 = MagicMock()
    mock_resp_1.text = (
        "Inheritance in Java is a fundamental principle of Object-Oriented Programming (OOP) "
        "that allows a subclass to acquire attributes and methods from an existing superclass [Source 1].\n\n"
        "- **Relationship Established**: It establishes an **'is-a' relationship** [Source 1].\n"
        "- **Key Benefit**: It improves code reusability across software components [Source 1]."
    )
    mock_resp_1.usage_metadata.total_token_count = 140
    mock_resp_1.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_resp_1
        rag_service.generation_service._client = mock_c

        req_1 = ChatAskRequest(question=q1, match_threshold=0.3, limit=3)
        res_1 = rag_service.ask(request=req_1, user_id=owner_user_id)

    print("\nGenerated Grounded Answer:")
    print(res_1.answer)
    print("\nStructured Citations List:")
    for idx, s in enumerate(res_1.sources, 1):
        print(f"  [{idx}] Citation Tag : {s.source_id}")
        print(f"      Document Name: {s.document_name}")
        print(f"      Document ID  : {s.document_id}")
        print(f"      Chunk ID     : {s.chunk_id}")
        print(f"      Page Number  : {s.page_number} (has_page_number={s.has_page_number})")
        print(f"      Similarity   : {s.similarity * 100:.1f}%")
        print(f"      Excerpt      : \"{s.snippet}\"")

    assert res_1.retrieval_metadata.grounded is True
    assert len(res_1.sources) >= 1
    assert res_1.sources[0].has_page_number is True
    assert res_1.sources[0].page_number == 1
    print("\n>>> TEST 1 PASSED: Grounded answer with structured citation metadata verified.")

    # --------------------------------------------------------------------------
    # Verification 2: Unrelated question -> Insufficient Context (Zero Hallucination)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[TEST 2] UNRELATED QUESTION -> INSUFFICIENT CONTEXT (ZERO HALLUCINATION)")
    print("=" * 90)
    q2 = "How do I bake homemade chocolate chip cookies with melted butter?"
    print(f"Query: \"{q2}\"")

    req_2 = ChatAskRequest(question=q2, match_threshold=0.5, limit=5)
    res_2 = rag_service.ask(request=req_2, user_id=owner_user_id)

    print("\nAnswer Response:")
    print(res_2.answer)
    print(f"Grounded Status          : {res_2.retrieval_metadata.grounded}")
    print(f"Insufficient Context Flag: {res_2.retrieval_metadata.is_insufficient_context}")
    print(f"Total Sources Returned   : {len(res_2.sources)}")
    assert res_2.retrieval_metadata.is_insufficient_context is True
    assert res_2.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(res_2.sources) == 0
    print(">>> TEST 2 PASSED: Zero hallucination, returned insufficient context.")

    # --------------------------------------------------------------------------
    # Verification 3: Multiple relevant chunks combined across pages
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[TEST 3] MULTIPLE RELEVANT CHUNKS COMBINED")
    print("=" * 90)
    q3 = "Explain how a child class uses a parent class in Java and the purpose of extending classes."
    print(f"Query: \"{q3}\"")

    mock_resp_3 = MagicMock()
    mock_resp_3.text = (
        "In Java OOP, inheritance operates through the following mechanisms:\n\n"
        "1. **Inheritance Mechanism**: A child class uses the `extends` keyword to access public and protected "
        "members of the parent class, and calls superclass constructors via `super` [Source 2].\n"
        "2. **Purpose of Extending**: Extending a base class enables polymorphism, modularity, and hierarchical classification [Source 3]."
    )
    mock_resp_3.usage_metadata.total_token_count = 210
    mock_resp_3.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_resp_3
        rag_service.generation_service._client = mock_c

        req_3 = ChatAskRequest(question=q3, match_threshold=0.2, limit=5)
        res_3 = rag_service.ask(request=req_3, user_id=owner_user_id)

    print("\nAnswer Response:")
    print(res_3.answer)
    print(f"Combined Pages           : {res_3.page_numbers}")
    print(f"Total Sources Cited      : {len(res_3.sources)}")
    assert len(res_3.sources) >= 2
    assert 2 in res_3.page_numbers and 3 in res_3.page_numbers
    print(">>> TEST 3 PASSED: Multiple chunks across pages combined successfully.")

    # --------------------------------------------------------------------------
    # Verification 4: Selected document filters work (DOCX without page numbers)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[TEST 4] SELECTED DOCUMENT FILTER (DOCX FORMAT - UNPAGINATED)")
    print("=" * 90)
    q4 = "What is the system architecture overview?"
    print(f"Query: \"{q4}\" with document filter: [{doc_2_id}] ({doc_2_name})")

    mock_resp_4 = MagicMock()
    mock_resp_4.text = (
        "The system architecture consists of a FastAPI backend service communicating with "
        "Supabase PostgreSQL and Gemini LLM [Source 1]."
    )
    mock_resp_4.usage_metadata.total_token_count = 90
    mock_resp_4.candidates = []

    with patch("google.genai.Client") as mock_client_cls:
        mock_c = MagicMock()
        mock_client_cls.return_value = mock_c
        mock_c.models.generate_content.return_value = mock_resp_4
        rag_service.generation_service._client = mock_c

        req_4 = ChatAskRequest(
            question=q4,
            document_ids=[doc_2_id],
            match_threshold=0.2,
        )
        res_4 = rag_service.ask(request=req_4, user_id=owner_user_id)

    print("\nAnswer Response:")
    print(res_4.answer)
    print(f"Documents Cited          : {res_4.document_names}")
    print(f"Source 1 Page Number     : {res_4.sources[0].page_number} (has_page_number={res_4.sources[0].has_page_number})")
    assert res_4.document_names == [doc_2_name]
    assert res_4.sources[0].has_page_number is False
    assert res_4.sources[0].page_number is None  # Never invents a page for DOCX!
    print(">>> TEST 4 PASSED: DOCX unpaginated document cited without fabricating page numbers.")

    # --------------------------------------------------------------------------
    # Verification 5: Multi-tenant security isolation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[TEST 5] MULTI-TENANT ISOLATION: UNAUTHORIZED USER ACCESS CHECK")
    print("=" * 90)
    print(f"Attacker User ID: {attacker_user_id}")
    print(f"Targeting Owner : {owner_user_id}")
    
    attacker_req = ChatAskRequest(
        question="What is inheritance in Java and what relationship does it establish?",
        match_threshold=0.05,
        limit=10,
    )
    attacker_res = rag_service.ask(request=attacker_req, user_id=attacker_user_id)

    print("\nAttacker Query Result:")
    print(f"Answer Returned          : \"{attacker_res.answer}\"")
    print(f"Chunks Retrieved         : {attacker_res.retrieval_metadata.total_chunks_retrieved}")
    print(f"Sources Leaked           : {len(attacker_res.sources)}")
    assert attacker_res.retrieval_metadata.total_chunks_retrieved == 0
    assert len(attacker_res.sources) == 0
    print(">>> TEST 5 PASSED: Multi-tenant boundary secure. Zero data leaked to other users.")

    # --------------------------------------------------------------------------
    # Verification 6: Stripping fabricated citation tags
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("[TEST 6] FABRICATED CITATION SANITIZATION")
    print("=" * 90)
    # Simulate LLM trying to cite [Source 99] which was not in the retrieved context
    raw_llm_text = (
        "Java inheritance establishes an is-a relationship [Source 1]. "
        "Invented claim with non-existent source [Source 99]."
    )
    cleaned_text, cited = gen_service.validate_and_sanitize_citations(raw_llm_text, {"Source 1"})
    print(f"Raw Answer with Fabricated Tag: \"{raw_llm_text}\"")
    print(f"Sanitized Answer              : \"{cleaned_text}\"")
    print(f"Valid Cited Sources           : {cited}")
    assert "[Source 1]" in cleaned_text
    assert "[Source 99]" not in cleaned_text
    print(">>> TEST 6 PASSED: Fabricated citations successfully detected and removed.")

    print("\n" + "=" * 90)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 90)


if __name__ == "__main__":
    run_rag_demonstration()
