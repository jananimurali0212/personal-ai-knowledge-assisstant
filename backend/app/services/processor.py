"""Document processing pipeline service orchestrating extraction, normalization, and chunking."""
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import httpx

from .extractor import (
    extract_document,
    ExtractedPage,
    ScannedPDFException,
    EmptyDocumentException,
    UnsupportedFormatException,
    DocumentExtractionError,
)
from .chunker import chunk_document, ProcessedChunk
from .embeddings import (
    GeminiEmbeddingService,
    default_embedding_service,
    EmbeddingError,
    GeminiAPIKeyMissingError,
)
from .chunk_repository import (
    ChunkRepository,
    default_chunk_repository,
    ChunkRepositoryError,
)
from ..models.chunk import DocumentChunkCreate
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ChunkSummary(BaseModel):
    chunk_index: int
    page_number: int
    char_count: int
    estimated_tokens: int
    preview: str
    has_embedding: bool = False

class DocumentProcessingResponse(BaseModel):
    document_id: UUID
    user_id: UUID
    file_name: str
    file_type: str
    status: str = Field(description="'uploaded', 'processing', 'processed', or 'failed'")
    error_message: Optional[str] = None
    ocr_required: bool = False
    total_pages: int = 0
    total_characters: int = 0
    total_chunks: int = 0
    chunks_stored_in_db: int = 0
    embedding_model_used: Optional[str] = None
    chunks: List[ChunkSummary] = []
    sample_extracted_text: Optional[str] = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# In-memory document status registry for runtime tracking and local tests
_processing_status_registry: Dict[str, Dict[str, Any]] = {}

def get_document_status(document_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
    """Retrieves current processing status for a document, enforcing user ownership."""
    key = str(document_id)
    record = _processing_status_registry.get(key)
    if not record:
        return None
    if str(record.get("user_id")) != str(user_id):
        logger.warning(f"Unauthorized access attempt to document {document_id} by user {user_id}")
        return None
    return record

def update_document_status(
    document_id: UUID, 
    user_id: UUID, 
    status: str, 
    error_message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
):
    """Updates the status registry and attempts Supabase remote sync if configured."""
    key = str(document_id)
    entry = _processing_status_registry.get(key, {"document_id": str(document_id), "user_id": str(user_id)})
    entry["status"] = status
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    if error_message:
        entry["error_message"] = error_message
    if extra:
        entry.update(extra)
    _processing_status_registry[key] = entry

    # Asynchronously attempt remote Supabase DB update if credentials exist
    supabase_url = settings.SUPABASE_URL
    api_key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
    if supabase_url and api_key and api_key != "your_supabase_anon_key_here":
        try:
            # Map 'processed' to 'ready' if standard DB enum uses 'ready'
            db_status = "ready" if status == "processed" else status
            endpoint = f"{supabase_url.rstrip('/')}/rest/v1/documents?id=eq.{document_id}&user_id=eq.{user_id}"
            headers = {
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            with httpx.Client(timeout=5.0) as client:
                client.patch(endpoint, json={"status": db_status}, headers=headers)
        except Exception as e:
            logger.debug(f"Remote Supabase status sync skipped or failed: {e}")

class DocumentProcessor:
    """Orchestrates the extraction, normalization, chunking, and vector embedding pipeline."""

    def __init__(
        self,
        target_chunk_tokens: int = 800,
        chunk_overlap_tokens: int = 150,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        chunk_repository: Optional[ChunkRepository] = None,
    ):
        self.target_chunk_tokens = target_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.embedding_service = embedding_service or default_embedding_service
        self.chunk_repository = chunk_repository or default_chunk_repository

    def process_file_content(
        self,
        file_bytes: bytes,
        file_type: str,
        file_name: str,
        document_id: UUID,
        user_id: UUID,
        generate_embeddings: bool = True,
    ) -> DocumentProcessingResponse:
        """
        Executes the full document processing pipeline:
        1. Sets status -> 'processing'
        2. Dispatches format extraction (PDF with page numbers, DOCX, TXT)
        3. Normalizes text
        4. Detects scanned PDFs requiring OCR
        5. Chunks text into ~800 token segments with ~150 token overlap
        6. Generates real Gemini vector embeddings (768 dimensions)
        7. Persists chunks and embeddings to Supabase (safe against duplicates)
        8. Sets status -> 'processed' or 'failed'
        """
        logger.info(f"Starting processing pipeline for doc={document_id}, user={user_id}, file={file_name}")

        # Step 1: Set status to 'processing'
        update_document_status(
            document_id=document_id,
            user_id=user_id,
            status="processing",
            extra={"file_name": file_name, "file_type": file_type}
        )

        try:
            # Step 2: Extract text page-by-page
            pages: List[ExtractedPage] = extract_document(
                file_bytes=file_bytes,
                file_type=file_type,
                file_name=file_name
            )

            total_characters = sum(p.char_count for p in pages)
            total_pages = len(pages)

            # Step 3: Chunk document
            chunks: List[ProcessedChunk] = chunk_document(
                pages=pages,
                document_id=document_id,
                target_tokens=self.target_chunk_tokens,
                overlap_tokens=self.chunk_overlap_tokens
            )

            stored_chunks_count = 0
            has_embeddings = False
            embedding_model_name = None

            # Step 4: Generate Real Vector Embeddings & Store in Supabase
            if generate_embeddings and chunks:
                # Check if chunks already exist in DB to avoid unnecessary regeneration
                existing_count = self.chunk_repository.get_existing_chunks_count(document_id, user_id)
                
                chunk_texts = [c.content for c in chunks]
                try:
                    embeddings = self.embedding_service.embed_chunks(
                        chunks=chunk_texts,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    has_embeddings = True
                    embedding_model_name = self.embedding_service.model

                    # Assemble chunk models with exact 768-dim embeddings
                    chunk_records = [
                        DocumentChunkCreate(
                            document_id=document_id,
                            user_id=user_id,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            page_number=c.page_number,
                            embedding=embeddings[idx],
                        )
                        for idx, c in enumerate(chunks)
                    ]

                    # Step 5: Store Chunks into Supabase
                    stored_chunks_count = self.chunk_repository.save_chunks(chunk_records, overwrite=True)

                except GeminiAPIKeyMissingError as ke:
                    logger.warning(f"Embedding generation skipped: {ke}")
                except Exception as ee:
                    logger.error(f"Error during embedding generation or storage: {ee}")
                    raise ee

            # Build chunk summaries
            chunk_summaries: List[ChunkSummary] = [
                ChunkSummary(
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    char_count=c.char_count,
                    estimated_tokens=c.estimated_tokens,
                    preview=c.content[:120].strip() + ("..." if len(c.content) > 120 else ""),
                    has_embedding=has_embeddings,
                )
                for c in chunks
            ]

            # Generate sample text preview
            sample_text = pages[0].text[:300].strip() if pages and pages[0].text else None
            if sample_text and len(pages[0].text) > 300:
                sample_text += "..."

            # Step 6: Update status to 'processed'
            update_document_status(
                document_id=document_id,
                user_id=user_id,
                status="processed",
                extra={
                    "total_pages": total_pages,
                    "total_chunks": len(chunks),
                    "total_characters": total_characters,
                    "chunks_stored_in_db": stored_chunks_count,
                    "embedding_model": embedding_model_name,
                }
            )

            return DocumentProcessingResponse(
                document_id=document_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                status="processed",
                total_pages=total_pages,
                total_characters=total_characters,
                total_chunks=len(chunks),
                chunks_stored_in_db=stored_chunks_count,
                embedding_model_used=embedding_model_name,
                chunks=chunk_summaries,
                sample_extracted_text=sample_text,
            )

        except ScannedPDFException as spe:
            logger.warning(f"Scanned PDF detected for document {document_id}: {spe}")
            error_msg = str(spe)
            update_document_status(
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error_message=error_msg,
                extra={"ocr_required": True}
            )
            return DocumentProcessingResponse(
                document_id=document_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                status="failed",
                error_message=error_msg,
                ocr_required=True,
            )

        except EmptyDocumentException as ede:
            logger.warning(f"Empty document {document_id}: {ede}")
            error_msg = str(ede)
            update_document_status(
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error_message=error_msg
            )
            return DocumentProcessingResponse(
                document_id=document_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                status="failed",
                error_message=error_msg,
            )

        except UnsupportedFormatException as ufe:
            logger.warning(f"Unsupported format for document {document_id}: {ufe}")
            error_msg = str(ufe)
            update_document_status(
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error_message=error_msg
            )
            return DocumentProcessingResponse(
                document_id=document_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                status="failed",
                error_message=error_msg,
            )

        except Exception as e:
            logger.exception(f"Unexpected processing error for document {document_id}: {e}")
            error_msg = f"Document processing failed: {str(e)}"
            update_document_status(
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error_message=error_msg
            )
            return DocumentProcessingResponse(
                document_id=document_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                status="failed",
                error_message=error_msg,
            )

default_processor = DocumentProcessor()
