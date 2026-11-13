-- ==============================================================================
-- Migration: 20261113_002_update_match_document_chunks.sql
-- Description: Update match_document_chunks RPC function to include document_name (file_name),
--              support multi-document filtering (filter_document_ids), and enforce user isolation.
-- ==============================================================================

CREATE OR REPLACE FUNCTION public.match_document_chunks (
    query_embedding vector(768),
    match_threshold float DEFAULT 0.3,
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
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        d.file_name AS document_name,
        dc.user_id,
        dc.chunk_index,
        dc.content,
        dc.page_number,
        (1 - (dc.embedding <=> query_embedding))::FLOAT AS similarity
    FROM public.document_chunks dc
    JOIN public.documents d ON dc.document_id = d.id AND d.user_id = dc.user_id
    WHERE
        -- Mandatory multi-tenant isolation: user must match authenticated user or explicit filter
        (dc.user_id = COALESCE(filter_user_id, auth.uid()))
        -- Optional document filtering (supports single or multiple selected documents)
        AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
        -- Cosine similarity threshold filter
        AND (1 - (dc.embedding <=> query_embedding)) >= match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

