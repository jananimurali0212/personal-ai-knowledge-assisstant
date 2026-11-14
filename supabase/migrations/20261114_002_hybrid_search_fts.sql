-- ==============================================================================
-- Migration: 20261114_002_hybrid_search_fts.sql
-- Description: Enable PostgreSQL Full-Text Search (FTS), add tsvector column to
--              document_chunks, create GIN index, and implement Keyword & Hybrid
--              Search with Reciprocal Rank Fusion (RRF).
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Add Generated tsvector Column for Full-Text Search
-- ------------------------------------------------------------------------------
-- Combines 'english' text search configuration (with stemming and stopword removal)
-- and 'simple' configuration (which preserves unstemmed lowercase tokens for
-- programming variable names, exact error codes, and technical identifiers).
ALTER TABLE public.document_chunks
ADD COLUMN IF NOT EXISTS fts tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', content) || to_tsvector('simple', content)
) STORED;

-- ------------------------------------------------------------------------------
-- 2. Create GIN Index for High-Performance Lexical Search
-- ------------------------------------------------------------------------------
-- Generalized Inverted Index (GIN) allows sub-millisecond full-text queries
-- across hundreds of thousands of document chunks.
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts_gin
ON public.document_chunks
USING gin (fts);

-- ------------------------------------------------------------------------------
-- 3. Stored Procedure: match_document_chunks_keyword (Pure Lexical Search)
-- ------------------------------------------------------------------------------
-- Performs full-text keyword retrieval using websearch_to_tsquery, ranked by ts_rank_cd.
CREATE OR REPLACE FUNCTION public.match_document_chunks_keyword (
    query_text text,
    match_count int DEFAULT 5,
    filter_user_id uuid DEFAULT NULL,
    filter_document_ids uuid[] DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    document_name TEXT,
    user_id UUID,
    chunk_index INTEGER,
    content TEXT,
    page_number INTEGER,
    rank FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    parsed_query tsquery;
BEGIN
    -- Build query matching both english stemming and exact simple tokens
    parsed_query := websearch_to_tsquery('english', query_text) || websearch_to_tsquery('simple', query_text);

    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        d.file_name AS document_name,
        dc.user_id,
        dc.chunk_index,
        dc.content,
        dc.page_number,
        ts_rank_cd(dc.fts, parsed_query)::FLOAT AS rank
    FROM public.document_chunks dc
    JOIN public.documents d ON dc.document_id = d.id AND d.user_id = dc.user_id
    WHERE
        -- Multi-tenant isolation: ensure user matches authenticated user or explicit filter
        (dc.user_id = COALESCE(filter_user_id, auth.uid()))
        -- Optional document filter (query specific documents or all user documents)
        AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
        -- Full-text matching condition
        AND dc.fts @@ parsed_query
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- ------------------------------------------------------------------------------
-- 4. Stored Procedure: match_document_chunks_hybrid (Hybrid Vector + Keyword Search)
-- ------------------------------------------------------------------------------
-- Combines pgvector cosine similarity retrieval with PostgreSQL full-text search
-- using Reciprocal Rank Fusion (RRF).
-- RRF formula: Score(d) = vector_weight / (rrf_k + rank_vector) + keyword_weight / (rrf_k + rank_keyword)
CREATE OR REPLACE FUNCTION public.match_document_chunks_hybrid (
    query_text text,
    query_embedding vector(768),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 5,
    filter_user_id uuid DEFAULT NULL,
    filter_document_ids uuid[] DEFAULT NULL,
    rrf_k int DEFAULT 60,
    vector_weight float DEFAULT 1.0,
    keyword_weight float DEFAULT 1.0
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    document_name TEXT,
    user_id UUID,
    chunk_index INTEGER,
    content TEXT,
    page_number INTEGER,
    similarity FLOAT,
    rrf_score FLOAT,
    vector_rank INT,
    keyword_rank INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    parsed_query tsquery;
BEGIN
    parsed_query := websearch_to_tsquery('english', query_text) || websearch_to_tsquery('simple', query_text);

    RETURN QUERY
    WITH vector_candidates AS (
        -- 1. Vector similarity candidates
        SELECT
            dc.id,
            dc.document_id,
            d.file_name AS document_name,
            dc.user_id,
            dc.chunk_index,
            dc.content,
            dc.page_number,
            (1 - (dc.embedding <=> query_embedding))::FLOAT AS similarity,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding ASC) AS v_rank
        FROM public.document_chunks dc
        JOIN public.documents d ON dc.document_id = d.id AND d.user_id = dc.user_id
        WHERE
            (dc.user_id = COALESCE(filter_user_id, auth.uid()))
            AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
            AND (1 - (dc.embedding <=> query_embedding)) >= match_threshold
        ORDER BY dc.embedding <=> query_embedding ASC
        LIMIT 50
    ),
    keyword_candidates AS (
        -- 2. Full-text keyword candidates
        SELECT
            dc.id,
            dc.document_id,
            d.file_name AS document_name,
            dc.user_id,
            dc.chunk_index,
            dc.content,
            dc.page_number,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(dc.fts, parsed_query) DESC) AS k_rank
        FROM public.document_chunks dc
        JOIN public.documents d ON dc.document_id = d.id AND d.user_id = dc.user_id
        WHERE
            (dc.user_id = COALESCE(filter_user_id, auth.uid()))
            AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
            AND dc.fts @@ parsed_query
        ORDER BY ts_rank_cd(dc.fts, parsed_query) DESC
        LIMIT 50
    ),
    combined_keys AS (
        SELECT id FROM vector_candidates
        UNION
        SELECT id FROM keyword_candidates
    )
    -- 3. Reciprocal Rank Fusion aggregation
    SELECT
        c.id,
        COALESCE(vc.document_id, kc.document_id) AS document_id,
        COALESCE(vc.document_name, kc.document_name) AS document_name,
        COALESCE(vc.user_id, kc.user_id) AS user_id,
        COALESCE(vc.chunk_index, kc.chunk_index) AS chunk_index,
        COALESCE(vc.content, kc.content) AS content,
        COALESCE(vc.page_number, kc.page_number) AS page_number,
        COALESCE(vc.similarity, 0.0)::FLOAT AS similarity,
        (
            COALESCE(vector_weight / (rrf_k + vc.v_rank), 0.0) +
            COALESCE(keyword_weight / (rrf_k + kc.k_rank), 0.0)
        )::FLOAT AS rrf_score,
        vc.v_rank::INT AS vector_rank,
        kc.k_rank::INT AS keyword_rank
    FROM combined_keys c
    LEFT JOIN vector_candidates vc ON c.id = vc.id
    LEFT JOIN keyword_candidates kc ON c.id = kc.id
    ORDER BY rrf_score DESC
    LIMIT match_count;
END;
$$;
