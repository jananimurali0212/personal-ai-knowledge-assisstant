"""Data models for RAG Chat orchestration and question-answering endpoints."""
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class ChatSourceItem(BaseModel):
    """Detailed attribution and metadata for a retrieved source chunk used in the answer."""
    chunk_id: Optional[UUID] = Field(default=None, description="Unique chunk identifier")
    chunk_identifier: Optional[str] = Field(default=None, description="String representation of chunk identifier")
    document_id: UUID = Field(..., description="ID of the parent document")
    document_name: str = Field(..., description="Filename or title of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number where the chunk content originated (PDF only; None for DOCX/TXT)")
    chunk_index: int = Field(default=0, description="Sequential index of the chunk within the document")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0.0 to 1.0)")
    snippet: str = Field(..., description="Short text excerpt from the cited chunk")
    relevant_excerpt: str = Field(default="", description="Relevant excerpt supporting the answer")
    source_id: Optional[str] = Field(default=None, description="Unique citation identifier e.g. 'Source 1'")
    has_page_number: bool = Field(default=False, description="True if document provides verified page numbers (PDF)")
    file_type: Optional[str] = Field(default=None, description="File format e.g. 'pdf', 'docx', 'txt'")


class ChatRetrievalMetadata(BaseModel):
    """Telemetry and search metadata for the RAG query execution."""
    total_chunks_retrieved: int = Field(..., description="Number of relevant chunks retrieved and evaluated")
    match_threshold: float = Field(..., description="Similarity threshold applied for filtering")
    embedding_model: str = Field(..., description="Model used to embed the query")
    generation_model: str = Field(..., description="LLM used to generate the answer")
    grounded: bool = Field(..., description="True if answer is directly grounded in retrieved context")
    is_insufficient_context: bool = Field(..., description="True if no relevant context or context insufficient")
    tokens_used: Optional[int] = Field(default=None, description="Total generation tokens consumed")


class ChatAskRequest(BaseModel):
    """
    Request schema for POST /api/chat/ask.
    Accepts user question with optional document filters and conversation tracking.
    """
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Natural-language question asked by the authenticated user",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document UUIDs to restrict retrieval to selected documents",
    )
    conversation_id: Optional[UUID] = Field(
        default=None,
        description="Optional conversation thread identifier for tracking dialog context",
    )
    match_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Cosine similarity cutoff threshold (default: 0.3)",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of relevant chunks to retrieve for generation (default: 5)",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional LLM temperature override",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=8192,
        description="Optional max generation tokens override",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Question must contain at least 2 non-whitespace characters.")
        if len(cleaned) > 2000:
            raise ValueError("Question cannot exceed 2000 characters.")
        return cleaned


class ChatAskResponse(BaseModel):
    """
    Response schema for POST /api/chat/ask.
    Returns the grounded answer along with full supporting source references and metadata.
    """
    conversation_id: Optional[UUID] = Field(default=None, description="Conversation ID if provided in request")
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded answer generated using retrieved context")
    sources: List[ChatSourceItem] = Field(default_factory=list, description="List of source chunk citations")
    document_names: List[str] = Field(default_factory=list, description="Unique names of documents providing evidence")
    page_numbers: List[int] = Field(default_factory=list, description="Unique page numbers providing evidence")
    relevant_chunk_ids: List[UUID] = Field(default_factory=list, description="IDs of relevant chunks used in answer")
    retrieval_metadata: ChatRetrievalMetadata = Field(..., description="Telemetry and execution metrics")
