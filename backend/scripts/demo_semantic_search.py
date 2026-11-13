"""Demonstration script testing Java Inheritance Semantic Vector Search."""
import sys
import os
from uuid import uuid4

# Ensure backend root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.search import SemanticSearchRequest
from app.services.search import default_semantic_search_service
from app.services.chunk_repository import _in_memory_chunks_store
from tests.test_semantic_search import (
    JAVA_CHUNK_1,
    JAVA_CHUNK_2,
    JAVA_CHUNK_3,
    make_semantic_vector,
)

def run_semantic_search_demo():
    print("=" * 85)
    print("PERSONAL AI KNOWLEDGE ASSISTANT: SEMANTIC VECTOR SEARCH DEMONSTRATION")
    print("=" * 85)

    user_id = uuid4()
    doc_id = uuid4()
    document_name = "Java_Inheritance_and_Polymorphism_Guide.pdf"

    # Step 1: Ingest Java inheritance document chunks with 768-dimensional embeddings
    print(f"\n[STEP 1] INDEXING DOCUMENT: '{document_name}' (ID: {doc_id})")
    print(f"Target User ID: {user_id}")
    print("-" * 85)

    v1 = make_semantic_vector(["inheritance", "oop", "parent", "child", "class", "is-a"])
    v2 = make_semantic_vector(["child", "parent", "extends", "super", "override", "animal", "dog"])
    v3 = make_semantic_vector(["purpose", "extending", "polymorphism", "modularity", "base", "class"])

    doc_key = f"{user_id}:{doc_id}"
    _in_memory_chunks_store[doc_key] = [
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "document_name": document_name,
            "user_id": str(user_id),
            "chunk_index": 0,
            "page_number": 1,
            "content": JAVA_CHUNK_1,
            "embedding": v1,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "document_name": document_name,
            "user_id": str(user_id),
            "chunk_index": 1,
            "page_number": 2,
            "content": JAVA_CHUNK_2,
            "embedding": v2,
        },
        {
            "id": str(uuid4()),
            "document_id": str(doc_id),
            "document_name": document_name,
            "user_id": str(user_id),
            "chunk_index": 2,
            "page_number": 3,
            "content": JAVA_CHUNK_3,
            "embedding": v3,
        },
    ]

    print(f"Indexed 3 chunks (768-dim vectors) across 3 pages.")
    print("  Chunk 0 (Page 1): Definition of Java Inheritance & 'is-a' relationship")
    print("  Chunk 1 (Page 2): Mechanism of 'extends', @Override, and 'super'")
    print("  Chunk 2 (Page 3): Purpose of extending classes (Polymorphism & Modularity)")

    # Step 2: Setup embedding service hook for queries
    def embed_query_hook(text, task_type="RETRIEVAL_QUERY"):
        words = text.lower().replace("?", "").replace(".", "").split()
        return make_semantic_vector(words)

    default_semantic_search_service.embedding_service.embed_text = embed_query_hook

    # Step 3: Test Required Queries
    test_queries = [
        ("Query 1", "What is inheritance in Java?", 0.3),
        ("Query 2", "Explain how a child class uses a parent class.", 0.3),
        ("Query 3", "What is the purpose of extending a class?", 0.3),
        ("Query 4 (Unrelated)", "How do I bake chocolate chip cookies from scratch?", 0.5),
    ]

    print("\n" + "=" * 85)
    print("[STEP 2] RUNNING SEMANTIC SEARCH QUERIES")
    print("=" * 85)

    for label, query, threshold in test_queries:
        print(f"\n>>> {label}: \"{query}\" (threshold={threshold})")
        request = SemanticSearchRequest(
            query=query,
            match_threshold=threshold,
            limit=5,
        )

        response = default_semantic_search_service.search(request, user_id=user_id)
        print(f"Total Matches Found: {response.total_results}")

        if response.total_results == 0:
            print("  [x] No relevant chunks returned (correctly filtered out by threshold).")
        else:
            for rank, r in enumerate(response.results, 1):
                print(f"  Result #{rank}:")
                print(f"    - Document Name : {r.document_name}")
                print(f"    - Document ID   : {r.document_id}")
                print(f"    - Page Number   : Page {r.page_number}")
                print(f"    - Chunk Index   : #{r.chunk_index}")
                print(f"    - Similarity    : {r.similarity * 100:.2f}% (Score: {r.similarity})")
                print(f"    - Content Snippet: \"{r.content[:130]}...\"")

    # Step 4: Multi-Tenant Access Control Verification
    print("\n" + "=" * 85)
    print("[STEP 3] MULTI-TENANT ISOLATION SECURITY CHECK")
    print("=" * 85)
    attacker_user_id = uuid4()
    attack_request = SemanticSearchRequest(query="What is inheritance in Java?", match_threshold=0.1)
    attack_res = default_semantic_search_service.search(attack_request, user_id=attacker_user_id)
    print(f"Attacker User ID: {attacker_user_id}")
    print(f"Attacker Retrieved Chunks Count: {attack_res.total_results}")
    print(f"Isolation Verification: {'SECURE - Zero Data Leaked' if attack_res.total_results == 0 else 'VULNERABILITY DETECTED'}")

    print("\n" + "=" * 85)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    run_semantic_search_demo()
