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
from .generation import (
    GeminiGenerationService,
    default_generation_service,
    GenerationError,
    GeminiAPIKeyMissingError as GenerationAPIKeyMissingError,
    GeminiRateLimitError,
    GeminiTimeoutError,
)
from .rag import (
    RAGOrchestratorService,
    default_rag_service,
    RAGOrchestrationError,
)
from .conversation_repository import (
    ConversationRepository,
    default_conversation_repository,
    ConversationRepositoryError,
)
from .summarizer import (
    DocumentSummarizerService,
    default_summarizer_service,
    SummarizationError,
    DocumentNotFoundError,
    EmptyDocumentError,
)
from .summary_repository import (
    SummaryRepository,
    default_summary_repository,
    SummaryRepositoryError,
)
from .reranker import (
    ReRankingService,
    default_reranker_service,
    ReRankingError,
)
from .quiz_repository import (
    QuizRepository,
    default_quiz_repository,
    QuizRepositoryError,
)
from .quiz import (
    QuizGenerationService,
    default_quiz_service,
    QuizServiceError,
    QuizNotFoundError,
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
    "GeminiGenerationService",
    "default_generation_service",
    "GenerationError",
    "GenerationAPIKeyMissingError",
    "GeminiRateLimitError",
    "GeminiTimeoutError",
    "RAGOrchestratorService",
    "default_rag_service",
    "RAGOrchestrationError",
    "ConversationRepository",
    "default_conversation_repository",
    "ConversationRepositoryError",
    "DocumentSummarizerService",
    "default_summarizer_service",
    "SummarizationError",
    "DocumentNotFoundError",
    "EmptyDocumentError",
    "SummaryRepository",
    "default_summary_repository",
    "SummaryRepositoryError",
    "ReRankingService",
    "default_reranker_service",
    "ReRankingError",
    "QuizRepository",
    "default_quiz_repository",
    "QuizRepositoryError",
    "QuizGenerationService",
    "default_quiz_service",
    "QuizServiceError",
    "QuizNotFoundError",
]
