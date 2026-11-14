"""API routes for Gemini-powered answer generation and RAG question answering."""
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_authenticated_user
from ..models.generation import (
    GenerateAnswerRequest,
    GenerateAnswerResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    ContextChunk,
)
from ..models.search import SemanticSearchRequest
from ..services.generation import (
    default_generation_service,
    GenerationError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiAPIKeyMissingError,
)
from ..services.search import (
    default_semantic_search_service,
    SemanticSearchError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generation", tags=["Gemini Answer Generation"])


@router.post("/answer", response_model=GenerateAnswerResponse)
def generate_answer_endpoint(
    request: GenerateAnswerRequest,
    current_user: UUID = Depends(get_authenticated_user),
):
    """
    Dedicated Gemini Answer Generation Endpoint.
    - Accepts question, retrieved document context, and optional source metadata.
    - Keeps answer generation decoupled from document retrieval.
    - Constructs structured prompt with strict grounding instructions.
    - Enforces zero-hallucination rules and accurate source citations.
    - Gracefully handles rate limits, timeouts, and API errors.
    """
    logger.info(
        f"Answer generation request from user={current_user}: question='{request.question[:60]}...'"
    )

    try:
        response = default_generation_service.generate_answer(
            question=request.question,
            context=request.context,
            source_metadata=request.source_metadata,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )
        return response

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except GeminiRateLimitError as rle:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(rle),
        )
    except GeminiTimeoutError as te:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(te),
        )
    except GeminiAPIKeyMissingError as kme:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(kme),
        )
    except GenerationError as ge:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(ge)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during answer generation.",
        )


@router.post("/rag", response_model=RAGQueryResponse)
def rag_query_endpoint(
    request: RAGQueryRequest,
    current_user: UUID = Depends(get_authenticated_user),
):
    """
    End-to-End RAG Q&A Endpoint.
    1. Retrieves relevant chunks from user's private documents via semantic search (pgvector).
    2. Passes retrieved chunks and source metadata into Gemini Generation Service.
    3. Returns grounded answer with citations and chunk count.
    """
    logger.info(
        f"RAG query request from user={current_user}: question='{request.question[:60]}...' "
        f"(threshold={request.match_threshold}, limit={request.limit})"
    )

    # Step 1: Document Retrieval via Semantic Search Service
    try:
        search_req = SemanticSearchRequest(
            query=request.question,
            document_ids=request.document_ids,
            match_threshold=request.match_threshold,
            limit=request.limit,
        )
        search_res = default_semantic_search_service.search(search_req, user_id=current_user)
    except Exception as e:
        logger.error(f"Failed to retrieve context chunks during RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {str(e)}",
        )

    # Step 2: Convert search results to ContextChunk objects
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

    # Step 3: Generate Grounded Answer using Gemini Generation Service
    try:
        gen_res = default_generation_service.generate_answer(
            question=request.question,
            context=context_chunks,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )

        return RAGQueryResponse(
            question=request.question,
            answer=gen_res.answer,
            sources=gen_res.sources,
            retrieved_chunks_count=len(context_chunks),
            model=gen_res.model,
            grounded=gen_res.grounded,
            is_insufficient_context=gen_res.is_insufficient_context,
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except GeminiRateLimitError as rle:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(rle))
    except GeminiTimeoutError as te:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(te))
    except GeminiAPIKeyMissingError as kme:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(kme))
    except GenerationError as ge:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ge))
    except Exception as e:
        logger.exception(f"Unexpected RAG error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during RAG answering.",
        )
