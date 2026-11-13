"""Document Chunk domain models and vector representations."""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Default Gemini text-embedding-004 dimension
GEMINI_EMBEDDING_DIMENSION = 768

class DocumentChunkBase(BaseModel):
    document_id: UUID
    chunk_index: int = Field(ge=0, description="0-indexed position within the document")
    content: str = Field(min_length=1, description="Raw textual content of the chunk")
    page_number: int = Field(default=1, ge=1, description="Original page number where chunk appears")

class DocumentChunkCreate(DocumentChunkBase):
    user_id: UUID
    embedding: List[float] = Field(
        ...,
        description=f"Gemini text-embedding-004 vector (exact {GEMINI_EMBEDDING_DIMENSION} dimensions)"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimensions(cls, v: List[float]) -> List[float]:
        if len(v) != GEMINI_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected exactly {GEMINI_EMBEDDING_DIMENSION} "
                f"dimensions for Gemini embedding, got {len(v)}."
            )
        return v

class DocumentChunk(DocumentChunkBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    embedding: Optional[List[float]] = None
    created_at: datetime

class SimilaritySearchResult(BaseModel):
    """Result returned by vector similarity search (match_document_chunks)."""
    id: UUID
    document_id: UUID
    user_id: UUID
    chunk_index: int
    content: str
    page_number: int
    similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity score (0.0 to 1.0)")

class VectorSearchQuery(BaseModel):
    """Query model for vector search requests."""
    query_text: str = Field(min_length=1)
    filter_document_id: Optional[UUID] = None
    match_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    match_count: int = Field(default=5, ge=1, le=50)
