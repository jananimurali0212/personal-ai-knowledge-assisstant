"""Data models for semantic, keyword, and hybrid search queries and retrieval results."""
from typing import List, Optional, Literal
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

SearchMode = Literal["hybrid", "vector", "keyword"]

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
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0.0 to 1.0; 1.0 is highest relevance)"
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Combined Reciprocal Rank Fusion (RRF) score"
    )
    vector_rank: Optional[int] = Field(
        default=None,
        description="1-based rank position in vector similarity retrieval"
    )
    keyword_rank: Optional[int] = Field(
        default=None,
        description="1-based rank position in full-text keyword retrieval"
    )
    keyword_score: Optional[float] = Field(
        default=None,
        description="Lexical rank score from ts_rank_cd or BM25"
    )
    match_type: Optional[str] = Field(
        default="vector",
        description="Type of match: 'hybrid' (both), 'vector' (semantic only), or 'keyword' (lexical only)"
    )
    file_type: Optional[str] = Field(
        default=None,
        description="File type extension ('pdf', 'docx', 'txt')"
    )
    rerank_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance logit/score for deep query-chunk relationship"
    )
    rerank_rank: Optional[int] = Field(
        default=None,
        description="1-based rank position after cross-encoder re-ranking"
    )
    initial_rank: Optional[int] = Field(
        default=None,
        description="1-based rank position prior to cross-encoder re-ranking"
    )


class SemanticSearchResponse(BaseModel):
    """Response payload for semantic vector search."""
    query: str
    total_results: int
    match_threshold: float
    embedding_model: str
    results: List[SearchResultItem]


class HybridSearchRequest(BaseModel):
    """Request payload for hybrid (vector + full-text keyword) search with RRF."""
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Search query string (supporting exact error codes, variables, or natural language)"
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document IDs to restrict search scope"
    )
    match_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity threshold for semantic candidate retrieval"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of merged results to return"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=500,
        description="Reciprocal Rank Fusion smoothing constant k (default: 60)"
    )
    search_mode: SearchMode = Field(
        default="hybrid",
        description="Retrieval strategy: 'hybrid' (RRF merge), 'vector' (semantic only), or 'keyword' (FTS only)"
    )
    vector_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Relative weight for vector rank in RRF"
    )
    keyword_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Relative weight for keyword rank in RRF"
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


class HybridSearchResponse(BaseModel):
    """Response payload for hybrid search containing fused rankings and retrieval metrics."""
    query: str
    search_mode: SearchMode
    total_results: int
    rrf_k: int
    embedding_model: str
    results: List[SearchResultItem]
    vector_candidate_count: int = 0
    keyword_candidate_count: int = 0


class ReRankRequest(BaseModel):
    """Request payload for cross-encoder second-stage re-ranking."""
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Search query to score candidate relationships against"
    )
    candidates: List[SearchResultItem] = Field(
        ...,
        description="First-stage candidate chunks retrieved from hybrid or vector search"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top re-ranked chunks to retain for the RAG prompt"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Optional minimum relevance score cutoff to exclude irrelevant chunks"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Cross-encoder model name override"
    )


class ReRankResponse(BaseModel):
    """Response payload for cross-encoder re-ranking."""
    query: str
    total_candidates: int
    total_returned: int
    model_name: str
    latency_ms: float
    results: List[SearchResultItem]
