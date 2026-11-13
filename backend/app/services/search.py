"""Semantic Vector Search Service using Gemini embeddings and Supabase pgvector."""
import math
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import httpx

from .embeddings import (
    GeminiEmbeddingService,
    default_embedding_service,
    EmbeddingError,
)
from .chunk_repository import ChunkRepository, default_chunk_repository, _in_memory_chunks_store
from ..models.search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SemanticSearchError(Exception):
    """Base exception for semantic search failures."""
    pass

def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two float vectors (range -1.0 to 1.0)."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = dot_product / (norm_a * norm_b)
    # Clamp to valid 0.0 - 1.0 range
    return max(0.0, min(1.0, float(similarity)))

class SemanticSearchService:
    """
    Coordinates semantic vector retrieval:
    1. Embeds the user's natural language query using Gemini (task_type=RETRIEVAL_QUERY).
    2. Queries Supabase match_document_chunks RPC with pgvector cosine distance (<=>).
    3. Strictly enforces user-level tenant isolation (users can NEVER retrieve another user's chunks).
    4. Filters by document IDs and match threshold.
    """

    def __init__(
        self,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        chunk_repository: Optional[ChunkRepository] = None,
    ):
        self.embedding_service = embedding_service or default_embedding_service
        self.chunk_repository = chunk_repository or default_chunk_repository

    def search(
        self,
        request: SemanticSearchRequest,
        user_id: UUID,
    ) -> SemanticSearchResponse:
        """
        Executes semantic search for the authenticated user.
        - Enforces user_id ownership.
        - Embeds query with Gemini.
        - Searches vector database with cosine similarity.
        - Returns ranked results meeting threshold.
        """
        query_text = request.query.strip()
        logger.info(
            f"Executing semantic search for user={user_id}: '{query_text[:60]}...' "
            f"(threshold={request.match_threshold}, limit={request.limit})"
        )

        # Step 1: Generate 768-dimensional query vector using Gemini
        try:
            query_vector = self.embedding_service.embed_text(
                text=query_text,
                task_type="RETRIEVAL_QUERY"
            )
        except EmbeddingError as ee:
            logger.error(f"Failed to generate query embedding: {ee}")
            raise SemanticSearchError(f"Embedding generation failed: {str(ee)}") from ee
        except Exception as e:
            logger.error(f"Unexpected error embedding query: {e}")
            raise SemanticSearchError(f"Failed to generate embedding: {str(e)}") from e

        # Step 2: Query vector store
        results: List[SearchResultItem] = []

        if self.chunk_repository.is_configured():
            # Invoke Supabase match_document_chunks RPC function
            results = self._search_supabase_rpc(
                query_vector=query_vector,
                user_id=user_id,
                document_ids=request.document_ids,
                match_threshold=request.match_threshold,
                limit=request.limit,
            )
        else:
            # Fallback for local testing / offline development
            results = self._search_in_memory(
                query_vector=query_vector,
                user_id=user_id,
                document_ids=request.document_ids,
                match_threshold=request.match_threshold,
                limit=request.limit,
            )

        return SemanticSearchResponse(
            query=query_text,
            total_results=len(results),
            match_threshold=request.match_threshold,
            embedding_model=self.embedding_service.model,
            results=results,
        )

    def _search_supabase_rpc(
        self,
        query_vector: List[float],
        user_id: UUID,
        document_ids: Optional[List[UUID]],
        match_threshold: float,
        limit: int,
    ) -> List[SearchResultItem]:
        """Invokes Supabase match_document_chunks stored procedure via PostgREST."""
        supabase_url = self.chunk_repository.supabase_url
        supabase_key = self.chunk_repository.supabase_key

        rpc_endpoint = f"{supabase_url}/rest/v1/rpc/match_document_chunks"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }

        rpc_payload: Dict[str, Any] = {
            "query_embedding": query_vector,
            "match_threshold": match_threshold,
            "match_count": limit,
            "filter_user_id": str(user_id),
            "filter_document_ids": [str(d) for d in document_ids] if document_ids else None,
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(rpc_endpoint, json=rpc_payload, headers=headers)
                if response.status_code != 200:
                    logger.error(f"Supabase RPC error ({response.status_code}): {response.text}")
                    # Fallback to in-memory if Supabase table is empty or error occurs
                    return self._search_in_memory(query_vector, user_id, document_ids, match_threshold, limit)

                data = response.json()
                return [
                    SearchResultItem(
                        chunk_id=UUID(item["id"]) if item.get("id") else None,
                        document_id=UUID(item["document_id"]),
                        document_name=item.get("document_name") or "Document",
                        page_number=int(item.get("page_number", 1)),
                        chunk_index=int(item.get("chunk_index", 0)),
                        content=item["content"],
                        similarity=round(float(item["similarity"]), 4),
                    )
                    for item in data
                ]

        except Exception as e:
            logger.warning(f"Supabase RPC call failed ({e}). Falling back to repository cache: {e}")
            return self._search_in_memory(query_vector, user_id, document_ids, match_threshold, limit)

    def _search_in_memory(
        self,
        query_vector: List[float],
        user_id: UUID,
        document_ids: Optional[List[UUID]],
        match_threshold: float,
        limit: int,
    ) -> List[SearchResultItem]:
        """In-memory vector similarity calculation for offline tests and development."""
        doc_id_strs = [str(d) for d in document_ids] if document_ids else None
        user_id_str = str(user_id)

        candidates: List[tuple[float, Dict[str, Any]]] = []

        for key, chunk_list in _in_memory_chunks_store.items():
            # Key format is '{user_id}:{document_id}'
            owner_id = key.split(":")[0] if ":" in key else ""
            if owner_id != user_id_str:
                # STRICT multi-tenant enforcement: skip any chunks not owned by user_id
                continue

            for chunk in chunk_list:
                c_doc_id = str(chunk.get("document_id"))
                if doc_id_strs and c_doc_id not in doc_id_strs:
                    continue

                chunk_vector = chunk.get("embedding")
                if not chunk_vector:
                    continue

                sim = compute_cosine_similarity(query_vector, chunk_vector)
                if sim >= match_threshold:
                    candidates.append((sim, chunk))

        # Sort descending by similarity score
        candidates.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResultItem(
                chunk_id=UUID(c["id"]) if c.get("id") else None,
                document_id=UUID(c["document_id"]),
                document_name=c.get("document_name") or "Java_Inheritance_Notes.pdf",
                page_number=int(c.get("page_number", 1)),
                chunk_index=int(c.get("chunk_index", 0)),
                content=c["content"],
                similarity=round(sim, 4),
            )
            for sim, c in candidates[:limit]
        ]

default_semantic_search_service = SemanticSearchService()

