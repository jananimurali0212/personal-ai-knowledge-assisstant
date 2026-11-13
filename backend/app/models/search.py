"""Data models for semantic search queries and retrieval results."""
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

class SemanticSearchRequest(BaseModel):
    """Request payload for semantic vector search."""
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Natural-language search query (between 2 and 2000 characters)"
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document IDs to restrict search to selected documents"
    )
    match_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold between 0.0 (least similar) and 1.0 (exact match)"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve (default: 5)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Search query must contain at least 2 non-whitespace characters.")
        if len(cleaned) > 2000:
            raise ValueError("Search query is too long. Maximum allowed length is 2000 characters.")
        return cleaned

class SearchResultItem(BaseModel):
    """Individual retrieved document chunk with relevance score and metadata."""
    chunk_id: Optional[UUID] = None
    document_id: UUID
    document_name: str = Field(description="Name/filename of the parent document")
    page_number: int = Field(default=1, description="Page number where the chunk content originated")
    chunk_index: int = Field(description="Sequential position of chunk within the document")
    content: str = Field(description="Extracted textual content of the chunk")
    similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0 to 1.0; 1.0 is highest relevance)"
    )

class SemanticSearchResponse(BaseModel):
    """Response payload for semantic vector search."""
    query: str
    total_results: int
    match_threshold: float
    embedding_model: str
    results: List[SearchResultItem]

