"""Data models for Gemini answer generation and RAG querying."""
from typing import List, Optional, Union, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator


class ContextChunk(BaseModel):
    """Structured representation of a retrieved document chunk with source metadata."""
    content: str = Field(..., min_length=1, description="Textual content of the chunk")
    document_name: str = Field(default="Document", description="Name/filename of the parent document")
    document_id: Optional[UUID] = Field(default=None, description="Unique ID of the parent document")
    chunk_id: Optional[UUID] = Field(default=None, description="Unique ID of the chunk")
    page_number: Optional[int] = Field(default=None, description="Page number where the chunk content originated (PDF only)")
    chunk_index: Optional[int] = Field(default=None, ge=0, description="Sequential index of the chunk")
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)")
    source_id: Optional[str] = Field(default=None, description="Human-readable citation identifier, e.g. 'Source 1'")
    file_type: Optional[str] = Field(default=None, description="File extension/type, e.g. 'pdf', 'docx', 'txt'")


class SourceCitation(BaseModel):
    """Source reference and attribution for a generated answer."""
    document_name: str = Field(description="Name/filename of the cited document")
    document_id: Optional[UUID] = Field(default=None, description="Document ID")
    chunk_id: Optional[UUID] = Field(default=None, description="Unique chunk ID")
    page_number: Optional[int] = Field(default=None, description="Source page number (PDF only)")
    chunk_index: Optional[int] = Field(default=None, description="Source chunk index")
    similarity: Optional[float] = Field(default=None, description="Relevance score of cited chunk")
    snippet: Optional[str] = Field(default=None, description="Brief snippet from the cited content")
    source_id: Optional[str] = Field(default=None, description="Citation tag, e.g. 'Source 1'")
    file_type: Optional[str] = Field(default=None, description="File extension/type")


class GenerateAnswerRequest(BaseModel):
    """
    Request payload for pure answer generation.
    Accepts question, retrieved document context, and optional source metadata.
    """
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User question to be answered using the provided context",
    )
    context: Union[List[ContextChunk], List[Dict[str, Any]], List[str], str] = Field(
        ...,
        description="Retrieved document context (structured chunks, raw string list, or unified context string)",
    )
    source_metadata: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional source metadata when context is provided as raw strings",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic and strictly factual)",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=8192,
        description="Maximum tokens for generated response",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Question must contain at least 2 non-whitespace characters.")
        if len(cleaned) > 2000:
            raise ValueError("Question is too long. Maximum allowed length is 2000 characters.")
        return cleaned


class GenerateAnswerResponse(BaseModel):
    """Structured response containing generated answer, grounding status, and citations."""
    question: str
    answer: str
    sources: List[SourceCitation]
    model: str
    grounded: bool = Field(
        ...,
        description="True if the answer is grounded in the retrieved context; False if context was insufficient",
    )
    is_insufficient_context: bool = Field(
        ...,
        description="True if the model indicated that the provided context was insufficient to answer",
    )
    tokens_used: Optional[int] = Field(default=None, description="Total tokens utilized in generation if available")
    finish_reason: Optional[str] = Field(default=None, description="Gemini candidate finish reason (e.g., STOP)")


class RAGQueryRequest(BaseModel):
    """
    End-to-end RAG query request.
    Performs retrieval over authenticated user documents and generates a grounded answer.
    """
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User question to be answered by searching private knowledge base",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document IDs to restrict search to selected documents",
    )
    match_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for chunk retrieval (default: 0.3)",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of relevant chunks to retrieve for generation",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Generation temperature override",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=8192,
        description="Maximum output tokens for generated answer",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Question must contain at least 2 non-whitespace characters.")
        return cleaned


class RAGQueryResponse(BaseModel):
    """End-to-end RAG response containing answer, citations, and retrieved chunk details."""
    question: str
    answer: str
    sources: List[SourceCitation]
    retrieved_chunks_count: int
    model: str
    grounded: bool
    is_insufficient_context: bool
