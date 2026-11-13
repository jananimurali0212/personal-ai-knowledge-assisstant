"""Semantic search API routes."""
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_authenticated_user
from ..models.search import (
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from ..services.search import (
    default_semantic_search_service,
    SemanticSearchError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Semantic Vector Search"])

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

