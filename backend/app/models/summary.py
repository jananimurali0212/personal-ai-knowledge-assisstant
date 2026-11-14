"""Pydantic domain models for AI-powered document summarization."""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class KeyTerminologyItem(BaseModel):
    """Specialized terminology item with definition."""
    term: str = Field(..., description="Key concept, technical term, or acronym.")
    definition: str = Field(..., description="Clear, factual definition extracted from context.")


class DocumentSummaryContent(BaseModel):
    """
    Structured document summary conforming strictly to requirements:
    1. Executive summary
    2. Main topics
    3. Important points
    4. Key terminology
    5. Important conclusions
    """
    executive_summary: str = Field(
        ...,
        description="Comprehensive high-level summary of the document's core thesis and purpose."
    )
    main_topics: List[str] = Field(
        default_factory=list,
        description="List of primary thematic subjects and modules addressed."
    )
    important_points: List[str] = Field(
        default_factory=list,
        description="Key factual takeaways, critical arguments, and supporting details."
    )
    key_terminology: List[KeyTerminologyItem] = Field(
        default_factory=list,
        description="List of specialized terms, technical variables, or concepts and their definitions."
    )
    important_conclusions: List[str] = Field(
        default_factory=list,
        description="Major conclusions, implications, or final outcomes of the document."
    )
    total_sections_analyzed: int = Field(
        default=1,
        description="Number of chunks or chunk sections analyzed in the summarization process."
    )
    is_hierarchical: bool = Field(
        default=False,
        description="True if document was summarized hierarchically via map-reduce."
    )
    model_used: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model identifier used for generation."
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the summary was generated."
    )


class DocumentSummaryRequest(BaseModel):
    """Request payload for triggering document summarization."""
    force_regenerate: bool = Field(
        default=False,
        description="If True, bypasses cached summary and forces a fresh Gemini re-summarization."
    )


class DocumentSummaryResponse(BaseModel):
    """API response model for document summarization."""
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    user_id: UUID
    file_name: str
    file_type: str
    status: str = Field(
        default="completed",
        description="'not_generated', 'generating', 'completed', or 'failed'"
    )
    summary: Optional[DocumentSummaryContent] = None
    cached: bool = Field(
        default=False,
        description="Indicates whether the returned summary was retrieved from cache without invoking Gemini."
    )
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
