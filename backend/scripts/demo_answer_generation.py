"""Demonstration script testing Gemini-Powered Grounded Answer Generation.
Demonstrates:
1. Chunk indexing & semantic retrieval of Java Inheritance documentation.
2. Grounded answer generation using GeminiGenerationService.
3. Source attribution with document names, page numbers, and snippets.
4. Insufficient context detection for ungrounded/out-of-scope queries.
"""
import sys
import os
from uuid import uuid4
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.search import SemanticSearchRequest
from app.models.generation import ContextChunk
from app.services.search import default_semantic_search_service
from app.services.generation import (
    GeminiGenerationService,
    SYSTEM_INSTRUCTION,
)
from app.services.chunk_repository import _in_memory_chunks_store
from tests.test_semantic_search import (
    JAVA_CHUNK_1,
    JAVA_CHUNK_2,
    JAVA_CHUNK_3,
    make_semantic_vector,
)


def run_answer_generation_demo():
    print("=" * 85)
    print("PERSONAL AI KNOWLEDGE ASSISTANT: GEMINI-POWERED ANSWER GENERATION")
    print("=" * 85)

    user_id = uuid4()
    doc_id = uuid4()
    document_name = "Java_Inheritance_and_Polymorphism_Guide.pdf"

    # Step 1: Populate document chunks in the repository
    print(f"\n[STEP 1] LOADING DOCUMENT REPOSITORY: '{document_name}'")
    print(f"User ID: {user_id}")
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

    print(f"Indexed 3 document chunks across 3 pages into knowledge repository.")

    # Configure semantic search query embedding hook
    def embed_query_hook(text, task_type="RETRIEVAL_QUERY"):
        words = text.lower().replace("?", "").replace(".", "").split()
        return make_semantic_vector(words)

    default_semantic_search_service.embedding_service.embed_text = embed_query_hook

    # Step 2: Initialize Gemini Generation Service
    generation_service = GeminiGenerationService(
        api_key="mock_backend_gemini_key",
        model_name="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=1024,
    )

    print(f"\n[STEP 2] CONFIGURED GEMINI GENERATION SERVICE")
    print(f"Model: {generation_service.model}")
    print(f"Temperature: {generation_service.temperature}")
    print(f"Max Tokens: {generation_service.max_output_tokens}")
    print(f"Timeout: {generation_service.timeout}s")

    # Step 3: Define Scenarios
    test_scenarios = [
        {
            "scenario": "Scenario 1: Factual definition grounded in Document Page 1",
            "question": "What is inheritance in Java and what relationship does it establish?",
            "threshold": 0.3,
            "mock_llm_response": (
                "Based on the provided document context, **inheritance** in Java is a fundamental principle "
                "of Object-Oriented Programming (OOP) that allows a subclass (or child class) to inherit "
                "attributes and methods from a superclass (or parent class).\n\n"
                "- **Relationship Established**: It establishes an **'is-a' relationship** between classes.\n"
                "- **Key Benefit**: It significantly improves code reusability.\n\n"
                "*Source Citation*: According to **Java_Inheritance_and_Polymorphism_Guide.pdf** (Page 1)."
            ),
        },
        {
            "scenario": "Scenario 2: Technical implementation details grounded in Document Page 2",
            "question": "How does a child class use a parent class in Java?",
            "threshold": 0.3,
            "mock_llm_response": (
                "In Java, a child class utilizes a parent class through the following mechanisms:\n\n"
                "1. **`extends` Keyword**: The child class uses the `extends` keyword to declare inheritance from the parent class.\n"
                "2. **Member Access**: It can access all public and protected members of the parent class.\n"
                "3. **Method Overriding**: It can provide specialized implementations of inherited methods using the `@Override` annotation.\n"
                "4. **`super` Keyword**: It can call parent class constructors or methods using `super`.\n\n"
                "*Source Citation*: Grounded in **Java_Inheritance_and_Polymorphism_Guide.pdf** (Page 2)."
            ),
        },
        {
            "scenario": "Scenario 3: Out-of-domain query with insufficient context",
            "question": "How do I bake chocolate chip cookies from scratch?",
            "threshold": 0.5,
            "mock_llm_response": (
                "The provided document context does not contain sufficient information to answer this question. "
                "The available documents focus exclusively on Java object-oriented programming and inheritance, "
                "with no baking recipes or culinary instructions."
            ),
        },
    ]

    print("\n" + "=" * 85)
    print("[STEP 3] RUNNING END-TO-END RETRIEVAL & ANSWER GENERATION")
    print("=" * 85)

    for item in test_scenarios:
        scenario = item["scenario"]
        question = item["question"]
        threshold = item["threshold"]
        simulated_text = item["mock_llm_response"]

        print(f"\n>>> {scenario}")
        print(f"Question: \"{question}\"")
        print("-" * 85)

        # 1. Retrieve Chunks via Semantic Vector Search
        search_req = SemanticSearchRequest(
            query=question,
            match_threshold=threshold,
            limit=3,
        )
        search_res = default_semantic_search_service.search(search_req, user_id=user_id)
        print(f"Retrieved Chunks: {search_res.total_results}")

        # Convert to ContextChunk objects
        context_chunks = [
            ContextChunk(
                content=r.content,
                document_name=r.document_name,
                document_id=r.document_id,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                similarity=r.similarity,
            )
            for r in search_res.results
        ]

        for idx, chunk in enumerate(context_chunks, 1):
            sim_str = f"{chunk.similarity * 100:.1f}%" if chunk.similarity else "N/A"
            print(f"  [Chunk {idx}] Document: {chunk.document_name} | Page: {chunk.page_number} | Relevance: {sim_str}")
            print(f"            Snippet: \"{chunk.content[:90]}...\"")

        # 2. Mock Gemini API response to verify prompt structure and service logic
        mock_response = MagicMock()
        mock_response.text = simulated_text
        mock_response.usage_metadata.total_token_count = 180
        mock_cand = MagicMock()
        mock_cand.finish_reason = "STOP"
        mock_response.candidates = [mock_cand]

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            generation_service._client = mock_client

            gen_result = generation_service.generate_answer(
                question=question,
                context=context_chunks,
            )

        print("\n--- GENERATED ANSWER ---")
        print(gen_result.answer)
        print("------------------------")
        print(f"Grounded in Context      : {gen_result.grounded}")
        print(f"Insufficient Context Flag: {gen_result.is_insufficient_context}")
        print(f"Model Utilized           : {gen_result.model}")
        print(f"Tokens Used              : {gen_result.tokens_used}")
        print(f"Citations Returned ({len(gen_result.sources)}):")
        for s in gen_result.sources:
            sim_pct = f"{s.similarity * 100:.1f}%" if s.similarity else "N/A"
            print(f"  * {s.document_name} (Page {s.page_number}, Chunk #{s.chunk_index}, Relevance: {sim_pct})")
            print(f"    Snippet: \"{s.snippet}\"")

    print("\n" + "=" * 85)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_answer_generation_demo()
