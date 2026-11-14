"""Hybrid Search Service combining semantic vector retrieval and PostgreSQL full-text search.
Uses Reciprocal Rank Fusion (RRF) to merge, rank, and deduplicate candidates.
Supports exact programming error codes, variable names, rare technical terms, and natural language.
"""
import re
import math
import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from uuid import UUID
import httpx

from .embeddings import (
    GeminiEmbeddingService,
    default_embedding_service,
    EmbeddingError,
)
from .chunk_repository import (
    ChunkRepository,
    default_chunk_repository,
    _in_memory_chunks_store,
)
from .search import (
    SemanticSearchService,
    compute_cosine_similarity,
    default_semantic_search_service,
    SemanticSearchError,
)
from ..models.search import (
    SemanticSearchRequest,
    SearchResultItem,
    SemanticSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    SearchMode,
)
from ..config.settings import settings

logger = logging.getLogger(__name__)


class HybridSearchError(Exception):
    """Base exception for hybrid retrieval failures."""
    pass


def tokenize_for_lexical_search(text: str) -> List[str]:
    """
    Tokenizes text for lexical matching.
    Extracts words, error codes with underscores/hyphens, and camelCase identifiers.
    """
    if not text:
        return []

    # Preserve exact alphanumeric tokens including underscores and hyphens (error codes, var names)
    raw_tokens = re.findall(r'[a-zA-Z0-9_\-\.]+', text)
    tokens: Set[str] = set()

    for token in raw_tokens:
        clean = token.strip('.-_').lower()
        if len(clean) >= 2:
            tokens.add(clean)

        # Also split camelCase identifiers: e.g. "activeConversationId" -> "active", "conversation", "id"
        sub_words = re.findall(r'[a-z]+|[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
        for sw in sub_words:
            sw_clean = sw.lower()
            if len(sw_clean) >= 2:
                tokens.add(sw_clean)

    return list(tokens)


def compute_lexical_score(query: str, content: str) -> float:
    """
    Computes an in-memory lexical relevance score between a query and document chunk.
    Gives heavy weight to:
    1. Exact substring matches (error codes, exact variable names)
    2. Token overlap and frequency
    """
    if not query or not content:
        return 0.0

    content_lower = content.lower()
    query_lower = query.lower().strip()

    score = 0.0

    # 1. Exact phrase / substring match boost (crucial for exact error codes and identifiers)
    if query_lower in content_lower:
        # Boost based on query length relative to chunk
        score += 10.0 + (len(query_lower) / 10.0)

    # 2. Token overlap score
    query_tokens = tokenize_for_lexical_search(query)
    content_tokens = set(tokenize_for_lexical_search(content))

    if query_tokens:
        matched_tokens = 0
        for token in query_tokens:
            if token in content_tokens:
                matched_tokens += 1
                # Add extra weight for exact match inside content
                count = content_lower.count(token)
                score += 1.0 + min(count, 5) * 0.5

        coverage = matched_tokens / len(query_tokens)
        score += coverage * 5.0

    return score


def apply_reciprocal_rank_fusion(
    vector_results: List[SearchResultItem],
    keyword_results: List[SearchResultItem],
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
    limit: int = 5,
) -> List[SearchResultItem]:
    """
    Combines and ranks search results using Reciprocal Rank Fusion (RRF).
    Formula:
        RRF_Score(d) = (vector_weight / (rrf_k + vector_rank)) + (keyword_weight / (rrf_k + keyword_rank))

    - Guarantees deduplication across retrieval methods.
    - Preserves document IDs, page numbers, chunk indexes, and source metadata.
    - Tags items with 'hybrid', 'vector', or 'keyword' match type.
    """
    scored_items: Dict[str, Dict[str, Any]] = {}

    def get_chunk_key(item: SearchResultItem) -> str:
        if item.chunk_id:
            return str(item.chunk_id)
        return f"{item.document_id}:{item.chunk_index}"

    # 1. Process Vector Results (1-based ranking)
    for rank, item in enumerate(vector_results, 1):
        key = get_chunk_key(item)
        if key not in scored_items:
            scored_items[key] = {
                "item": item,
                "vector_rank": rank,
                "keyword_rank": None,
                "similarity": item.similarity,
                "keyword_score": None,
                "in_vector": True,
                "in_keyword": False,
            }
        else:
            scored_items[key]["vector_rank"] = rank
            scored_items[key]["similarity"] = item.similarity
            scored_items[key]["in_vector"] = True

    # 2. Process Keyword Results (1-based ranking)
    for rank, item in enumerate(keyword_results, 1):
        key = get_chunk_key(item)
        if key not in scored_items:
            scored_items[key] = {
                "item": item,
                "vector_rank": None,
                "keyword_rank": rank,
                "similarity": item.similarity,
                "keyword_score": item.keyword_score,
                "in_vector": False,
                "in_keyword": True,
            }
        else:
            scored_items[key]["keyword_rank"] = rank
            scored_items[key]["in_keyword"] = True
            if item.keyword_score is not None:
                scored_items[key]["keyword_score"] = item.keyword_score

    # 3. Calculate RRF Score for each unique chunk
    final_results: List[SearchResultItem] = []

    for key, data in scored_items.items():
        v_rank = data["vector_rank"]
        k_rank = data["keyword_rank"]

        v_score = (vector_weight / (rrf_k + v_rank)) if v_rank is not None else 0.0
        k_score = (keyword_weight / (rrf_k + k_rank)) if k_rank is not None else 0.0
        total_rrf = v_score + k_score

        base_item = data["item"]
        match_type = "hybrid" if (data["in_vector"] and data["in_keyword"]) else ("vector" if data["in_vector"] else "keyword")

        merged_item = SearchResultItem(
            chunk_id=base_item.chunk_id,
            document_id=base_item.document_id,
            document_name=base_item.document_name,
            page_number=base_item.page_number,
            chunk_index=base_item.chunk_index,
            content=base_item.content,
            similarity=data["similarity"],
            rrf_score=round(total_rrf, 6),
            vector_rank=v_rank,
            keyword_rank=k_rank,
            keyword_score=data["keyword_score"],
            match_type=match_type,
            file_type=base_item.file_type or ("pdf" if base_item.document_name.lower().endswith(".pdf") else "docx"),
        )
        final_results.append(merged_item)

    # 4. Sort descending by RRF score, breaking ties by vector similarity then keyword score
    final_results.sort(
        key=lambda x: (
            x.rrf_score or 0.0,
            x.similarity,
            x.keyword_score or 0.0,
        ),
        reverse=True
    )

    return final_results[:limit]


class HybridSearchService:
    """
    Dedicated Hybrid Search Orchestrator combining:
    1. Semantic vector similarity (Gemini embeddings + pgvector)
    2. Full-text keyword search (PostgreSQL tsvector / GIN index)
    3. Reciprocal Rank Fusion (RRF) with configurable smoothing factor k
    """

    def __init__(
        self,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        chunk_repository: Optional[ChunkRepository] = None,
        semantic_search_service: Optional[SemanticSearchService] = None,
    ):
        self.embedding_service = embedding_service or default_embedding_service
        self.chunk_repository = chunk_repository or default_chunk_repository
        self.semantic_search_service = semantic_search_service or default_semantic_search_service

    def search_vector(
        self,
        query: str,
        user_id: UUID,
        document_ids: Optional[List[UUID]] = None,
        match_threshold: float = 0.3,
        limit: int = 20,
    ) -> List[SearchResultItem]:
        """Performs vector similarity retrieval."""
        req = SemanticSearchRequest(
            query=query,
            document_ids=document_ids,
            match_threshold=match_threshold,
            limit=limit,
        )
        resp = self.semantic_search_service.search(req, user_id=user_id)
        for rank, item in enumerate(resp.results, 1):
            item.vector_rank = rank
            item.match_type = "vector"
        return resp.results

    def search_keyword(
        self,
        query: str,
        user_id: UUID,
        document_ids: Optional[List[UUID]] = None,
        limit: int = 20,
    ) -> List[SearchResultItem]:
        """Performs PostgreSQL full-text retrieval (or in-memory lexical search)."""
        if self.chunk_repository.is_configured():
            return self._search_supabase_keyword_rpc(query, user_id, document_ids, limit)
        return self._search_in_memory_keyword(query, user_id, document_ids, limit)

    def search_hybrid(
        self,
        request: HybridSearchRequest,
        user_id: UUID,
    ) -> HybridSearchResponse:
        """
        Executes hybrid search combining vector retrieval and full-text keyword retrieval.
        Merges results using Reciprocal Rank Fusion (RRF).
        """
        clean_query = request.query.strip()
        logger.info(
            f"Executing hybrid search for user={user_id}: '{clean_query[:60]}...' "
            f"mode={request.search_mode}, k={request.rrf_k}, limit={request.limit}"
        )

        # 1. Keyword-only mode
        if request.search_mode == "keyword":
            keyword_candidates = self.search_keyword(
                query=clean_query,
                user_id=user_id,
                document_ids=request.document_ids,
                limit=request.limit,
            )
            return HybridSearchResponse(
                query=clean_query,
                search_mode="keyword",
                total_results=len(keyword_candidates),
                rrf_k=request.rrf_k,
                embedding_model=self.embedding_service.model,
                results=keyword_candidates,
                vector_candidate_count=0,
                keyword_candidate_count=len(keyword_candidates),
            )

        # 2. Vector-only mode
        if request.search_mode == "vector":
            vector_candidates = self.search_vector(
                query=clean_query,
                user_id=user_id,
                document_ids=request.document_ids,
                match_threshold=request.match_threshold,
                limit=request.limit,
            )
            return HybridSearchResponse(
                query=clean_query,
                search_mode="vector",
                total_results=len(vector_candidates),
                rrf_k=request.rrf_k,
                embedding_model=self.embedding_service.model,
                results=vector_candidates,
                vector_candidate_count=len(vector_candidates),
                keyword_candidate_count=0,
            )

        # 3. Hybrid Mode (Vector + Keyword + RRF)
        # Attempt Supabase match_document_chunks_hybrid RPC if configured
        if self.chunk_repository.is_configured():
            try:
                rpc_results = self._search_supabase_hybrid_rpc(request, user_id)
                if rpc_results is not None:
                    v_count = len([r for r in rpc_results if r.vector_rank is not None])
                    k_count = len([r for r in rpc_results if r.keyword_rank is not None])
                    return HybridSearchResponse(
                        query=clean_query,
                        search_mode="hybrid",
                        total_results=len(rpc_results),
                        rrf_k=request.rrf_k,
                        embedding_model=self.embedding_service.model,
                        results=rpc_results,
                        vector_candidate_count=v_count,
                        keyword_candidate_count=k_count,
                    )
            except Exception as e:
                logger.warning(f"Supabase hybrid RPC failed ({e}). Falling back to separate candidate fusion: {e}")

        # Dual-candidate retrieval and Python RRF fusion
        candidate_pool_limit = min(max(request.limit * 2, 20), 50)

        # Retrieve vector candidates
        vector_candidates = self.search_vector(
            query=clean_query,
            user_id=user_id,
            document_ids=request.document_ids,
            match_threshold=request.match_threshold,
            limit=candidate_pool_limit,
        )

        # Retrieve keyword candidates
        keyword_candidates = self.search_keyword(
            query=clean_query,
            user_id=user_id,
            document_ids=request.document_ids,
            limit=candidate_pool_limit,
        )

        # Apply Reciprocal Rank Fusion
        fused_results = apply_reciprocal_rank_fusion(
            vector_results=vector_candidates,
            keyword_results=keyword_candidates,
            rrf_k=request.rrf_k,
            vector_weight=request.vector_weight,
            keyword_weight=request.keyword_weight,
            limit=request.limit,
        )

        return HybridSearchResponse(
            query=clean_query,
            search_mode="hybrid",
            total_results=len(fused_results),
            rrf_k=request.rrf_k,
            embedding_model=self.embedding_service.model,
            results=fused_results,
            vector_candidate_count=len(vector_candidates),
            keyword_candidate_count=len(keyword_candidates),
        )

    # --------------------------------------------------------------------------
    # SUPABASE RPC INTEGRATIONS
    # --------------------------------------------------------------------------

    def _search_supabase_hybrid_rpc(
        self,
        request: HybridSearchRequest,
        user_id: UUID,
    ) -> Optional[List[SearchResultItem]]:
        """Invokes public.match_document_chunks_hybrid in Supabase via PostgREST."""
        query_vector = self.embedding_service.embed_text(
            text=request.query,
            task_type="RETRIEVAL_QUERY"
        )

        endpoint = f"{self.chunk_repository.supabase_url}/rest/v1/rpc/match_document_chunks_hybrid"
        headers = {
            "apikey": self.chunk_repository.supabase_key,
            "Authorization": f"Bearer {self.chunk_repository.supabase_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_text": request.query,
            "query_embedding": query_vector,
            "match_threshold": request.match_threshold,
            "match_count": request.limit,
            "filter_user_id": str(user_id),
            "filter_document_ids": [str(d) for d in request.document_ids] if request.document_ids else None,
            "rrf_k": request.rrf_k,
            "vector_weight": request.vector_weight,
            "keyword_weight": request.keyword_weight,
        }

        with httpx.Client(timeout=15.0) as client:
            res = client.post(endpoint, json=payload, headers=headers)
            if res.status_code != 200:
                logger.error(f"match_document_chunks_hybrid returned {res.status_code}: {res.text}")
                return None

            data = res.json()
            results = []
            for item in data:
                v_rank = item.get("vector_rank")
                k_rank = item.get("keyword_rank")
                match_type = "hybrid" if (v_rank is not None and k_rank is not None) else ("vector" if v_rank is not None else "keyword")

                results.append(
                    SearchResultItem(
                        chunk_id=UUID(item["id"]) if item.get("id") else None,
                        document_id=UUID(item["document_id"]),
                        document_name=item.get("document_name") or "Document",
                        page_number=int(item.get("page_number", 1)),
                        chunk_index=int(item.get("chunk_index", 0)),
                        content=item["content"],
                        similarity=round(float(item.get("similarity", 0.0)), 4),
                        rrf_score=round(float(item["rrf_score"]), 6) if item.get("rrf_score") is not None else None,
                        vector_rank=v_rank,
                        keyword_rank=k_rank,
                        match_type=match_type,
                        file_type="pdf" if (item.get("document_name") or "").lower().endswith(".pdf") else "docx",
                    )
                )
            return results

    def _search_supabase_keyword_rpc(
        self,
        query: str,
        user_id: UUID,
        document_ids: Optional[List[UUID]],
        limit: int,
    ) -> List[SearchResultItem]:
        """Invokes public.match_document_chunks_keyword in Supabase."""
        endpoint = f"{self.chunk_repository.supabase_url}/rest/v1/rpc/match_document_chunks_keyword"
        headers = {
            "apikey": self.chunk_repository.supabase_key,
            "Authorization": f"Bearer {self.chunk_repository.supabase_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query_text": query,
            "match_count": limit,
            "filter_user_id": str(user_id),
            "filter_document_ids": [str(d) for d in document_ids] if document_ids else None,
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(endpoint, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    results = []
                    for rank, item in enumerate(data, 1):
                        results.append(
                            SearchResultItem(
                                chunk_id=UUID(item["id"]) if item.get("id") else None,
                                document_id=UUID(item["document_id"]),
                                document_name=item.get("document_name") or "Document",
                                page_number=int(item.get("page_number", 1)),
                                chunk_index=int(item.get("chunk_index", 0)),
                                content=item["content"],
                                similarity=0.0,
                                keyword_rank=rank,
                                keyword_score=round(float(item.get("rank", 0.0)), 4),
                                match_type="keyword",
                                file_type="pdf" if (item.get("document_name") or "").lower().endswith(".pdf") else "docx",
                            )
                        )
                    return results
        except Exception as e:
            logger.warning(f"Supabase keyword RPC failed ({e}). Falling back to in-memory: {e}")

        return self._search_in_memory_keyword(query, user_id, document_ids, limit)

    # --------------------------------------------------------------------------
    # IN-MEMORY LEXICAL SEARCH (LOCAL DEV & TESTS)
    # --------------------------------------------------------------------------

    def _search_in_memory_keyword(
        self,
        query: str,
        user_id: UUID,
        document_ids: Optional[List[UUID]],
        limit: int,
    ) -> List[SearchResultItem]:
        """In-memory lexical full-text matching supporting exact codes, variables, and terms."""
        user_id_str = str(user_id)
        doc_id_strs = [str(d) for d in document_ids] if document_ids else None

        candidates: List[Tuple[float, Dict[str, Any]]] = []

        for key, chunk_list in _in_memory_chunks_store.items():
            # Format is '{user_id}:{document_id}'
            owner_id = key.split(":")[0] if ":" in key else ""
            if owner_id != user_id_str:
                continue

            for chunk in chunk_list:
                c_doc_id = str(chunk.get("document_id"))
                if doc_id_strs and c_doc_id not in doc_id_strs:
                    continue

                content = chunk.get("content", "")
                score = compute_lexical_score(query, content)
                if score > 0.0:
                    candidates.append((score, chunk))

        # Sort descending by lexical score
        candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, c) in enumerate(candidates[:limit], 1):
            doc_name = c.get("document_name") or "Document"
            results.append(
                SearchResultItem(
                    chunk_id=UUID(c["id"]) if c.get("id") else None,
                    document_id=UUID(c["document_id"]),
                    document_name=doc_name,
                    page_number=int(c.get("page_number", 1)),
                    chunk_index=int(c.get("chunk_index", 0)),
                    content=c["content"],
                    similarity=0.0,
                    keyword_rank=rank,
                    keyword_score=round(score, 4),
                    match_type="keyword",
                    file_type="pdf" if doc_name.lower().endswith(".pdf") else "docx",
                )
            )

        return results


default_hybrid_search_service = HybridSearchService()
