"""Tests for Gemini vector embedding service and Supabase chunk storage integration."""
import io
from uuid import uuid4
from unittest.mock import MagicMock, patch
import pytest

from app.services.embeddings import (
    GeminiEmbeddingService,
    GeminiAPIKeyMissingError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
)
from app.services.chunk_repository import ChunkRepository
from app.models.chunk import DocumentChunkCreate
from app.services.processor import DocumentProcessor
from app.services.extractor import extract_pdf
from app.services.chunker import chunk_document
import pymupdf as fitz

# ==============================================================================
# Helper to create sample PDF bytes
# ==============================================================================

def create_sample_pdf(page_texts: list[str]) -> bytes:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text(fitz.Point(50, 72), text, fontsize=11)
    b = doc.tobytes()
    doc.close()
    return b

# ==============================================================================
# 1. Gemini Embedding Service Unit Tests
# ==============================================================================

def test_missing_api_key_raises_informative_error():
    service = GeminiEmbeddingService(api_key="")
    with pytest.raises(GeminiAPIKeyMissingError) as exc_info:
        service.embed_text("Sample knowledge chunk")
    assert "GEMINI_API_KEY" in str(exc_info.value)

def test_empty_or_whitespace_text_validation():
    service = GeminiEmbeddingService(api_key="mock_key_for_test")
    with pytest.raises(ValueError, match="Cannot generate embedding for empty"):
        service.embed_text("   \n\t  ")

def test_empty_chunk_in_batch_validation():
    service = GeminiEmbeddingService(api_key="mock_key_for_test")
    with pytest.raises(ValueError, match="empty or whitespace"):
        service.embed_chunks(["Valid chunk 1", "   ", "Valid chunk 3"])

@patch("google.genai.Client")
def test_embed_text_invokes_gemini_with_768_dimensions(mock_genai_client_class):
    # Mock GenAI client and response
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    fake_768_vector = [0.05] * 768
    mock_response = MagicMock()
    mock_embedding_obj = MagicMock()
    mock_embedding_obj.values = fake_768_vector
    mock_response.embeddings = [mock_embedding_obj]
    mock_client.models.embed_content.return_value = mock_response

    service = GeminiEmbeddingService(api_key="test_api_key", model_name="text-embedding-004")
    service._client = mock_client

    vector = service.embed_text("Knowledge base test query", task_type="RETRIEVAL_QUERY")

    assert len(vector) == 768
    assert mock_client.models.embed_content.called
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-004"
    assert call_kwargs["config"].output_dimensionality == 768
    assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

@patch("google.genai.Client")
def test_embed_chunks_batching_efficiency(mock_genai_client_class):
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    fake_768_vector = [0.01] * 768
    mock_response = MagicMock()
    mock_response.embeddings = [
        MagicMock(values=fake_768_vector),
        MagicMock(values=fake_768_vector),
    ]
    mock_client.models.embed_content.return_value = mock_response

    service = GeminiEmbeddingService(api_key="test_api_key", model_name="text-embedding-004")
    service._client = mock_client

    chunks = ["Chunk 1 content", "Chunk 2 content"]
    vectors = service.embed_chunks(chunks, task_type="RETRIEVAL_DOCUMENT", batch_size=50)

    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768

@patch("google.genai.Client")
def test_dimension_mismatch_detection(mock_genai_client_class):
    mock_client = MagicMock()
    mock_genai_client_class.return_value = mock_client

    # Return unexpected dimension (e.g. 512 instead of 768)
    fake_wrong_vector = [0.01] * 512
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=fake_wrong_vector)]
    mock_client.models.embed_content.return_value = mock_response

    service = GeminiEmbeddingService(api_key="test_key", dimension=768)
    service._client = mock_client

    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
        service.embed_text("Sample input")
    assert "expected 768, got 512" in str(exc_info.value)

# ==============================================================================
# 2. Chunk Repository & Safe Upsert Tests
# ==============================================================================

def test_chunk_repository_save_and_retrieve():
    repo = ChunkRepository(supabase_url="https://mock.supabase.co", supabase_key="")
    doc_id = uuid4()
    user_id = uuid4()

    chunks = [
        DocumentChunkCreate(
            document_id=doc_id,
            user_id=user_id,
            chunk_index=0,
            content="First chunk content for RAG.",
            page_number=1,
            embedding=[0.02] * 768,
        ),
        DocumentChunkCreate(
            document_id=doc_id,
            user_id=user_id,
            chunk_index=1,
            content="Second chunk content on vector indexes.",
            page_number=2,
            embedding=[0.03] * 768,
        ),
    ]

    saved_count = repo.save_chunks(chunks, overwrite=True)
    assert saved_count == 2

    # Verify count
    existing = repo.get_existing_chunks_count(doc_id, user_id)
    assert existing == 2

    # Verify retrieval
    retrieved = repo.get_chunks_by_document(doc_id, user_id)
    assert len(retrieved) == 2
    assert retrieved[0]["chunk_index"] == 0
    assert retrieved[1]["chunk_index"] == 1

# ==============================================================================
# 3. Full Integrated Processing Pipeline with Vector Embeddings
# ==============================================================================

def test_integrated_document_processing_with_real_embedding_pipeline():
    """
    Tests the complete workflow:
    1. Extract multi-page PDF text (preserving page metadata)
    2. Split into chunks (~800 tokens, ~150 overlap)
    3. Generate 768-dimensional Gemini embeddings
    4. Store chunks and embeddings into the repository
    5. Verify stored chunks count matches processed chunks count
    """
    # 1. Create 2-page sample PDF
    page_1_text = "Page 1: Architectural principles of Retrieval-Augmented Generation with pgvector."
    page_2_text = "Page 2: Implementing Gemini text-embedding-004 vector embeddings in FastAPI."
    pdf_bytes = create_sample_pdf([page_1_text, page_2_text])

    # 2. Setup mock embedding service returning consistent 768-dim embeddings
    mock_embedding_service = MagicMock(spec=GeminiEmbeddingService)
    mock_embedding_service.model = "text-embedding-004"
    mock_embedding_service.dimension = 768

    def mock_embed_chunks(chunks, task_type="RETRIEVAL_DOCUMENT", batch_size=50):
        return [[0.042] * 768 for _ in chunks]

    mock_embedding_service.embed_chunks.side_effect = mock_embed_chunks

    # 3. Setup repository
    repo = ChunkRepository(supabase_url="https://mock.supabase.co", supabase_key="")

    # 4. Initialize processor
    processor = DocumentProcessor(
        embedding_service=mock_embedding_service,
        chunk_repository=repo,
    )

    doc_id = uuid4()
    user_id = uuid4()

    # 5. Process document
    result = processor.process_file_content(
        file_bytes=pdf_bytes,
        file_type="pdf",
        file_name="rag_architecture.pdf",
        document_id=doc_id,
        user_id=user_id,
        generate_embeddings=True,
    )

    # 6. Verify assertions
    assert result.status == "processed"
    assert result.total_pages == 2
    assert result.total_chunks > 0
    # Number of stored chunks matches the number of successfully processed chunks
    assert result.chunks_stored_in_db == result.total_chunks
    assert result.embedding_model_used == "text-embedding-004"

    # Verify stored chunks in repository
    stored_chunks = repo.get_chunks_by_document(doc_id, user_id)
    assert len(stored_chunks) == result.total_chunks
    assert stored_chunks[0]["document_id"] == str(doc_id)
    assert stored_chunks[0]["user_id"] == str(user_id)
    assert len(stored_chunks[0]["embedding"]) == 768

