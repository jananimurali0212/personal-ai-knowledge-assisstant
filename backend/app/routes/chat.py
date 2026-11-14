"""Chat API routes for RAG-powered question answering."""
import logging
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..auth import get_authenticated_user
from ..models.chat import (
    ChatAskRequest,
    ChatAskResponse,
)
from ..services.rag import (
    default_rag_service,
    RAGOrchestratorService,
    RAGOrchestrationError,
)
from ..services.generation import (
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiAPIKeyMissingError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["RAG Chat"])


@router.post("/ask", response_model=ChatAskResponse)
def ask_chat_endpoint(
    request: ChatAskRequest,
    current_user: UUID = Depends(get_authenticated_user),
):
    """
    RAG-Powered Question Answering Endpoint (POST /api/chat/ask).
    - Accepts user's natural-language question, optional selected document IDs, and conversation ID.
    - Strictly derives user identity from verified JWT or verified session header.
    - Generates query embedding and retrieves relevant chunks via pgvector similarity search.
    - Enforces user-level document isolation and filters by similarity threshold.
    - Formats retrieved chunks and prompts Gemini with strict grounding guidelines.
    - Returns the grounded answer, verified source citations, document names, page numbers,
      relevant chunk IDs, and telemetry metadata.
    """
    logger.info(
        f"Incoming chat ask request from authenticated user={current_user}: "
        f"question='{request.question[:60]}...' "
        f"(doc_filter={request.document_ids}, threshold={request.match_threshold})"
    )

    try:
        response = default_rag_service.ask(
            request=request,
            user_id=current_user,
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
    except RAGOrchestrationError as roe:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG processing failed: {str(roe)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error in /chat/ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the answer.",
        )


@router.post("/stream")
def chat_stream_endpoint(
    request: ChatAskRequest,
    raw_request: Request,
    current_user: UUID = Depends(get_authenticated_user),
    rag_service: RAGOrchestratorService = Depends(lambda: default_rag_service),
):
    """
    RAG-Powered Question Answering Streaming Endpoint (POST /api/chat/stream).
    - Accepts user question, optional selected document IDs, and conversation ID in request body.
    - Authenticates user via Bearer JWT token or X-User-ID header.
    - Yields real-time Server-Sent Events (SSE):
        • event: start -> handshake with conversation ID
        • event: token -> individual progressive answer tokens
        • event: sources -> structured source citations and document names
        • event: complete -> completion status, grounding verification
        • event: error -> diagnostic error detail if failure occurs
    - Cancels generation gracefully if client disconnects.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to stream chat responses.",
        )

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty or whitespace.",
        )

    logger.info(
        f"Incoming chat stream request from user={current_user}: "
        f"question='{request.question[:60]}...' (doc_filter={request.document_ids})"
    )

    generator = rag_service.ask_stream(
        request=request,
        user_id=current_user,
        raw_request=raw_request,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
