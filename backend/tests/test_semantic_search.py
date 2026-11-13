"""Tests for semantic search endpoint and vector similarity retrieval."""
import math
from uuid import uuid4
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException

from app.models.search import SemanticSearchRequest
from app.services.search import (
    SemanticSearchService,
    compute_cosine_similarity,
)
from app.services.chunk_repository import ChunkRepository, _in_memory_chunks_store
from app.services.embeddings import GeminiEmbeddingService
from app.routes.search import semantic_search
from app.auth import get_authenticated_user

# ==============================================================================
# Java Inheritance Test Data
# ==============================================================================

JAVA_CHUNK_1 = (
    "Inheritance in Java is a fundamental principle of Object-Oriented Programming (OOP) "
    "that enables a new class to inherit attributes and methods from an existing class. "
    "The class that inherits is known as the subclass or child class, and the class whose "
    "properties are inherited is known as the superclass or parent class. Inheritance establishes "
    "an 'is-a' relationship and significantly improves code reusability."
)

JAVA_CHUNK_2 = (
    "In Java, a child class uses the 'extends' keyword to inherit from a parent class. "
    "The child class can access all public and protected members of the parent class, "
    "override methods to provide specialized implementations using @Override, and "
    "call superclass constructors or methods using the 'super' keyword. For example, "
    "class Dog extends Animal allows Dog to use Animal's eat() method while adding its own bark() method."
)

JAVA_CHUNK_3 = (
    "The purpose of extending a class in Java is polymorphism, code reuse, and hierarchical classification. "
    "Extending a base class allows developers to design specialized domain models without rewriting "
    "foundational logic. Java enforces single class inheritance to prevent ambiguity like the diamond problem, "
    "while allowing multiple interface inheritance."
)

def make_semantic_vector(keywords_present: list[str]) -> list[float]:
    """
    Creates a deterministic 768-dimensional embedding representing semantic concept presence.
    Simulates high semantic alignment for related keywords.
    """
    vec = [0.01] * 768
    # Concept dimension mappings
    concept_map = {
        "inheritance": (10, 50, 1.0),
        "oop": (10, 50, 0.8),
        "child": (100, 150, 1.0),
        "parent": (100, 150, 1.0),
        "extends": (100, 150, 0.9),
        "super": (100, 150, 0.7),
        "purpose": (200, 250, 1.0),
        "extending": (200, 250, 0.95),
        "polymorphism": (200, 250, 0.8),
        "cookie": (600, 650, 1.0),
        "recipe": (600, 650, 1.0),
    }
    for word in keywords_present:
        w_lower = word.lower()
        for k, (start, end, weight) in concept_map.items():
            if k in w_lower:
                for idx in range(start, end):
                    vec[idx] += weight

    # Normalize vector to unit length
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]

# ==============================================================================
# 1. Cosine Similarity Function Tests
# ==============================================================================

def test_cosine_similarity_computation():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert compute_cosine_similarity(v1, v2) == 1.0
    assert compute_cosine_similarity(v1, v3) == 0.0

# ==============================================================================
# 2. Input Validation Tests
# ==============================================================================

def test_search_request_validates_empty_query():
    with pytest.raises(ValueError, match="at least 2 non-whitespace"):
        SemanticSearchRequest(query="   ")

def test_search_request_validates_excessively_long_query():
    long_query = "a" * 2005
    with pytest.raises(ValueError, match="2000"):
        SemanticSearchRequest(query=long_query)

# ==============================================================================
# 3. Java Inheritance Semantic Search Tests
# ==============================================================================

@pytest.fixture
def java_search_setup():
    user_id = uuid4()
    doc_id = uuid4()

    # Pre-index Java chunks with simulated semantic embeddings
    v1 = make_semantic_vector(["inheritance", "oop", "parent", "child"])
    v2 = make_semantic_vector(["child", "parent", "extends", "super"])
    v3 = make_semantic_vector(["purpose", "extending", "polymorphism"])

    doc_key = f"{user_id}:{doc_id}"
    _in_memory_chunks_store[doc_key] = [
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "document_name": "Java_Inheritance_Guide.pdf",
            "user_id": str(user_id),
            "chunk_index": 0,
            "page_number": 1,
            "content": JAVA_CHUNK_1,
            "embedding": v1,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "document_name": "Java_Inheritance_Guide.pdf",
            "user_id": str(user_id),
            "chunk_index": 1,
            "page_number": 2,
            "content": JAVA_CHUNK_2,
            "embedding": v2,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "document_name": "Java_Inheritance_Guide.pdf",
            "user_id": str(user_id),
            "chunk_index": 2,
            "page_number": 3,
            "content": JAVA_CHUNK_3,
            "embedding": v3,
        },
    ]

    mock_embedding_service = MagicMock(spec=GeminiEmbeddingService)
    mock_embedding_service.model = "text-embedding-004"
    mock_embedding_service.dimension = 768

    def mock_embed_text(text, task_type="RETRIEVAL_QUERY"):
        # Map query to vector based on semantic terms
        words = text.lower().split()
        return make_semantic_vector(words)

    mock_embedding_service.embed_text.side_effect = mock_embed_text

    repo = ChunkRepository(supabase_url="https://mock.supabase.co", supabase_key="")
    service = SemanticSearchService(
        embedding_service=mock_embedding_service,
        chunk_repository=repo,
    )

    return service, user_id, doc_id

def test_query_1_what_is_inheritance_in_java(java_search_setup):
    service, user_id, doc_id = java_search_setup
    request = SemanticSearchRequest(
        query="What is inheritance in Java?",
        match_threshold=0.3,
        limit=5,
    )
    res = service.search(request, user_id=user_id)

    assert res.total_results >= 1
    top_result = res.results[0]
    # Page 1 contains the direct definition of inheritance
    assert top_result.page_number == 1
    assert "fundamental principle of Object-Oriented Programming" in top_result.content
    assert top_result.similarity > 0.5

def test_query_2_explain_how_child_uses_parent(java_search_setup):
    service, user_id, doc_id = java_search_setup
    request = SemanticSearchRequest(
        query="Explain how a child class uses a parent class.",
        match_threshold=0.3,
        limit=5,
    )
    res = service.search(request, user_id=user_id)

    assert res.total_results >= 1
    top_result = res.results[0]
    # Page 2 explains how child extends parent using 'extends' and 'super'
    assert top_result.page_number == 2
    assert "extends" in top_result.content
    assert top_result.similarity > 0.5

def test_query_3_what_is_purpose_of_extending_class(java_search_setup):
    service, user_id, doc_id = java_search_setup
    request = SemanticSearchRequest(
        query="What is the purpose of extending a class?",
        match_threshold=0.3,
        limit=5,
    )
    res = service.search(request, user_id=user_id)

    assert res.total_results >= 1
    top_result = res.results[0]
    # Page 3 explains the purpose of extending a class (polymorphism, modularity)
    assert top_result.page_number == 3
    assert "purpose of extending a class" in top_result.content
    assert top_result.similarity > 0.5

def test_unrelated_query_is_rejected_or_filtered_out(java_search_setup):
    service, user_id, doc_id = java_search_setup
    unrelated_request = SemanticSearchRequest(
        query="How do I bake chocolate chip cookies from scratch?",
        match_threshold=0.6,  # Threshold filters out weak semantic correlation
        limit=5,
    )
    res = service.search(unrelated_request, user_id=user_id)

    # Irrelevant chunks are not returned as relevant
    assert res.total_results == 0

# ==============================================================================
# 4. Security & Multi-Tenant User Isolation Tests
# ==============================================================================

def test_user_cannot_retrieve_another_users_document_chunks(java_search_setup):
    service, owner_user_id, doc_id = java_search_setup
    attacker_user_id = uuid4()

    request = SemanticSearchRequest(
        query="What is inheritance in Java?",
        match_threshold=0.1,
        limit=5,
    )

    # Attacker queries -> returns 0 results because of strict user_id filtering
    attacker_res = service.search(request, user_id=attacker_user_id)
    assert attacker_res.total_results == 0

def test_search_document_id_filtering(java_search_setup):
    service, user_id, doc_id = java_search_setup
    other_doc_id = uuid4()

    # Search with filter targeting non-existent document ID
    request = SemanticSearchRequest(
        query="What is inheritance in Java?",
        document_ids=[other_doc_id],
        match_threshold=0.1,
    )
    res = service.search(request, user_id=user_id)
    assert res.total_results == 0

    # Search targeting the actual doc_id
    valid_request = SemanticSearchRequest(
        query="What is inheritance in Java?",
        document_ids=[doc_id],
        match_threshold=0.1,
    )
    valid_res = service.search(valid_request, user_id=user_id)
    assert valid_res.total_results > 0
    assert valid_res.results[0].document_id == doc_id

# ==============================================================================
# 5. FastAPI Endpoint & Auth Dependency Tests
# ==============================================================================

def test_auth_dependency_rejects_missing_headers():
    with pytest.raises(HTTPException) as exc_401:
        get_authenticated_user(authorization=None, x_user_id=None)
    assert exc_401.value.status_code == 401

def test_auth_dependency_accepts_valid_uuid_header():
    uid = uuid4()
    extracted = get_authenticated_user(authorization=None, x_user_id=str(uid))
    assert extracted == uid

def test_api_semantic_search_endpoint_direct():
    user_id = uuid4()
    request = SemanticSearchRequest(query="Inheritance concept in OOP", limit=3)
    
    with patch.object(
        service_search := __import__("app.services.search", fromlist=["default_semantic_search_service"]).default_semantic_search_service.embedding_service,
        "embed_text",
        return_value=[0.01] * 768
    ):
        response = semantic_search(request=request, current_user=user_id)
        assert response.query == "Inheritance concept in OOP"
        assert response.embedding_model == "text-embedding-004"
