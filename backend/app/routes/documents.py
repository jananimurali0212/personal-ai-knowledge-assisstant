"""Document processing API routes."""
from typing import Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, status
import logging

from ..services.processor import (
    default_processor,
    get_document_status,
    DocumentProcessingResponse,
)
from ..models.summary import (
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    DocumentSummaryContent,
)
from ..services.summarizer import (
    default_summarizer_service,
    DocumentNotFoundError,
    EmptyDocumentError,
    SummarizationError,
)
from ..services.summary_repository import default_summary_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents Processing"])

def get_current_user_id(
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID (passed from Supabase session)")
) -> UUID:
    """Helper to extract and validate authenticated user identity from request headers."""
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Missing X-User-ID header."
        )
    try:
        return UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format. Must be a valid UUID."
        )

@router.post("/process-file", response_model=DocumentProcessingResponse)
def process_document_file(
    file: UploadFile = File(..., description="Document file to extract and process (PDF, DOCX, TXT)"),
    document_id: Optional[str] = Form(None, description="Optional existing document UUID"),
    user_id_override: Optional[str] = Form(None, description="User UUID form parameter fallback"),
    x_user_id: Optional[str] = Header(None, description="User UUID from auth header"),
):
    """
    Direct document processing endpoint:
    - Accepts uploaded PDF, DOCX, or TXT file.
    - Validates user identity ensuring account isolation.
    - Extracts text page-by-page preserving page numbers.
    - Normalizes text and detects scanned PDFs requiring OCR.
    - Generates chunks (approx 800 tokens, 150 overlap).
    - Returns extraction summary, sample text, and chunk counts.
    """
    # Authenticate user from header or form field
    raw_user_id = x_user_id or user_id_override
    if not raw_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Please provide X-User-ID header or user_id."
        )
    try:
        user_uuid = UUID(raw_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user_id format. Must be a valid UUID."
        )

    # Resolve document UUID
    doc_uuid = UUID(document_id) if document_id else uuid4()

    # Read file content safely
    try:
        file_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes)."
        )

    # Determine extension
    filename = file.filename or "uploaded_file"
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    logger.info(f"Processing document '{filename}' for user '{user_uuid}' (doc_id={doc_uuid})")

    # Run processing pipeline
    result = default_processor.process_file_content(
        file_bytes=file_bytes,
        file_type=ext,
        file_name=filename,
        document_id=doc_uuid,
        user_id=user_uuid
    )

    return result

@router.get("/{document_id}/status")
def get_status(
    document_id: UUID,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None
):
    """
    Retrieves the processing status of a document.
    Enforces user isolation: users can only check documents belonging to their own account.
    """
    raw_uid = x_user_id or user_id
    if not raw_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide X-User-ID header."
        )
    try:
        user_uuid = UUID(raw_uid)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format."
        )

    status_record = get_document_status(document_id=document_id, user_id=user_uuid)
    if not status_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized to view this document."
        )

    return status_record


@router.post("/{document_id}/summarize", response_model=DocumentSummaryResponse)
def summarize_document_endpoint(
    document_id: UUID,
    request: Optional[DocumentSummaryRequest] = None,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """
    Generates or retrieves a cached AI-powered structured summary of a document.
    - Verifies document ownership for the authenticated user.
    - Employs hierarchical map-reduce summarization for multi-section/long documents.
    - Returns executive summary, main topics, important points, key terminology, and conclusions.
    - Caches results to prevent redundant LLM invocations unless force_regenerate is requested.
    """
    raw_uid = x_user_id or user_id
    if not raw_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide X-User-ID header."
        )
    try:
        user_uuid = UUID(raw_uid)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format. Must be a valid UUID."
        )

    req = request or DocumentSummaryRequest()

    try:
        response = default_summarizer_service.summarize_document(
            document_id=document_id,
            user_id=user_uuid,
            request=req,
        )
        return response
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except EmptyDocumentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except SummarizationError as e:
        logger.error(f"Summarization failed for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate document summary: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in summarization endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during summarization.",
        )


@router.get("/{document_id}/summary", response_model=DocumentSummaryResponse)
def get_document_summary_endpoint(
    document_id: UUID,
    x_user_id: Optional[str] = Header(None, description="Authenticated User UUID"),
    user_id: Optional[str] = None,
):
    """
    Retrieves the existing saved summary for a document without regenerating.
    """
    raw_uid = x_user_id or user_id
    if not raw_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide X-User-ID header."
        )
    try:
        user_uuid = UUID(raw_uid)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format."
        )

    # Fetch document metadata
    try:
        doc_meta = default_summarizer_service._get_document_metadata(document_id, user_uuid)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized.",
        )

    summary_record = default_summary_repository.get_summary(document_id, user_uuid)
    if not summary_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No summary has been generated for this document yet.",
        )

    content = default_summarizer_service._format_summary_content(
        data=summary_record,
        total_sections=summary_record.get("total_sections_analyzed", 1),
        is_hierarchical=summary_record.get("hierarchical", False),
    )

    return DocumentSummaryResponse(
        document_id=document_id,
        user_id=user_uuid,
        file_name=doc_meta.get("file_name", "document"),
        file_type=doc_meta.get("file_type", "txt"),
        status="completed",
        summary=content,
        cached=True,
        created_at=summary_record.get("created_at"),
    )
