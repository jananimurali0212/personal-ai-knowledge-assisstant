"""Services package for document processing and AI orchestration."""
from .normalizer import normalize_text
from .extractor import (
    extract_document,
    extract_pdf,
    extract_docx,
    extract_txt,
    ExtractedPage,
    ScannedPDFException,
    EmptyDocumentException,
    UnsupportedFormatException,
    DocumentExtractionError,
)
from .chunker import chunk_document, ProcessedChunk, estimate_tokens
from .embeddings import (
    GeminiEmbeddingService,
    default_embedding_service,
    EmbeddingError,
    GeminiAPIKeyMissingError,
    EmbeddingDimensionMismatchError,
)
from .chunk_repository import (
    ChunkRepository,
    default_chunk_repository,
    ChunkRepositoryError,
)
from .processor import (
    DocumentProcessor,
    DocumentProcessingResponse,
    ChunkSummary,
    default_processor,
    get_document_status,
    update_document_status,
)

__all__ = [
    "normalize_text",
    "extract_document",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
    "ExtractedPage",
    "ScannedPDFException",
    "EmptyDocumentException",
    "UnsupportedFormatException",
    "DocumentExtractionError",
    "chunk_document",
    "ProcessedChunk",
    "estimate_tokens",
    "GeminiEmbeddingService",
    "default_embedding_service",
    "EmbeddingError",
    "GeminiAPIKeyMissingError",
    "EmbeddingDimensionMismatchError",
    "ChunkRepository",
    "default_chunk_repository",
    "ChunkRepositoryError",
    "DocumentProcessor",
    "DocumentProcessingResponse",
    "ChunkSummary",
    "default_processor",
    "get_document_status",
    "update_document_status",
]
