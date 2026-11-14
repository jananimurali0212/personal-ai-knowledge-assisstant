"""Semantic and Hybrid search API routes."""
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_authenticated_user
from ..models.search import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    ReRankRequest,
    ReRankResponse,
)
from ..services.search import (
    default_semantic_search_service,
    SemanticSearchError,
)
from ..services.hybrid_search import (
    default_hybrid_search_service,
    HybridSearchError,
)
from ..services.reranker import (
    default_reranker_service,
    ReRankingError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Vector & Hybrid Search"])


@router.post("", response_model=SemanticSearchResponse)
def semantic_search(
    request: SemanticSearchRequest,
    current_user: UUID = Depends(get_authenticated_user),
):
    """
    Semantic Vector Search Endpoint:
    - Accepts a natural language query (2-2000 characters).
    - Derives authenticated user identity from Bearer JWT or verified session header.
    - Generates a 768-dimensional query vector using Google Gemini (task_type=RETRIEVAL_QUERY).
    - Performs vector similarity search via pgvector (HNSW cosine distance).
    - Returns top 5 relevant document chunks by default (filtered by match_threshold).
    - Supports restricting search to selected document IDs.
    - Strictly enforces user-level tenant isolation in the database query.
    """
    logger.info(
        f"Search request received from user={current_user}: query='{request.query[:50]}...' "
        f"(limit={request.limit}, threshold={request.match_threshold})"
    )

    try:
        response = default_semantic_search_service.search(
            request=request,
            user_id=current_user,
        )
        return response

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except SemanticSearchError as sse:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(sse)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during semantic vector search."
        )


@router.post("/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    request: HybridSearchRequest,
    current_user: UUID = Depends(get_authenticated_user),
):
    """
    Hybrid Search Endpoint (Semantic Vector + PostgreSQL Full-Text Keyword Search):
    - Combines dense vector similarity with lexical tsvector full-text search.
    - Fuses rankings using Reciprocal Rank Fusion (RRF):
        Score(d) = (vector_weight / (rrf_k + vector_rank)) + (keyword_weight / (rrf_k + keyword_rank))
    - Prioritizes exact programming error codes, variable names, and rare technical keywords.
    - Preserves semantic conceptual recall for natural-language questions.
    - Supports document selection filters and multi-tenant RLS user isolation.
    """
    logger.info(
        f"Hybrid search request received from user={current_user}: query='{request.query[:50]}...' "
        f"mode={request.search_mode}, k={request.rrf_k}, limit={request.limit}"
    )

    try:
        response = default_hybrid_search_service.search_hybrid(
            request=request,
            user_id=current_user,
        )
        return response

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except (HybridSearchError, SemanticSearchError) as hse:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(hse)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected hybrid search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during hybrid retrieval."
        )


@router.post("/rerank", response_model=ReRankResponse)
def rerank_candidates(
    request: ReRankRequest,
    current_user: UUID = Depends(get_authenticated_user),
):
    """
    Cross-Encoder Re-Ranking Endpoint:
    - Accepts query and first-stage candidate chunks (retrieved via hybrid or semantic search).
    - Applies a cross-encoder model to score bidirectional query-chunk textual relevance.
    - Preserves all chunk IDs, document IDs, page numbers, and source metadata.
    - Slices to top_k and filters out irrelevant chunks below score threshold.
    """
    logger.info(
        f"Re-ranking request from user={current_user}: query='{request.query[:50]}...' "
        f"candidates={len(request.candidates)}, top_k={request.top_k}"
    )

    try:
        response = default_reranker_service.rerank(
            query=request.query,
            candidates=request.candidates,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
        return response
    except Exception as e:
        logger.exception(f"Unexpected error during cross-encoder re-ranking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-ranking failed: {str(e)}",
        )
