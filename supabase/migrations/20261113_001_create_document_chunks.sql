-- ==============================================================================
-- Migration: 20261113_001_create_document_chunks.sql
-- Description: Enable pgvector, create document_chunks table with 768-dim embeddings,
--              HNSW vector index, relational indexes, and RLS policies.
-- Model: Google Gemini text-embedding-004 (native 768 dimensions)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Enable pgvector Extension
-- ------------------------------------------------------------------------------
-- pgvector enables the `vector` data type and similarity search operators:
--   <=> : Cosine distance (1 - cosine similarity)
--   <-> : Euclidean (L2) distance
--   <#> : Negative inner product
CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------------------------
-- 2. Create document_chunks Table
-- ------------------------------------------------------------------------------
-- Stores chunked passages and their high-dimensional vector representations.
-- Designed for Gemini text-embedding-004 (768 dimensions).
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE DEFAULT auth.uid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER DEFAULT 1,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,

    -- Ensure a document cannot have duplicate chunks at the same index
    CONSTRAINT uq_document_chunk_index UNIQUE (document_id, chunk_index)
);

-- ------------------------------------------------------------------------------
-- 3. Add Indexes for Relational Filtering
-- ------------------------------------------------------------------------------
-- Fast lookups by document ID (e.g., retrieving all chunks of a file or cascading deletes)
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
    ON public.document_chunks(document_id);

-- Fast lookups by user ID (critical for RLS checks and multi-document user queries)
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id 
    ON public.document_chunks(user_id);

-- Composite index for ordered sequential reconstruction of document text
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_chunk_idx 
    ON public.document_chunks(document_id, chunk_index);

-- ------------------------------------------------------------------------------
-- 4. Add Vector Similarity Search Index (HNSW)
-- ------------------------------------------------------------------------------
-- HNSW (Hierarchical Navigable Small World) provides superior query throughput
-- and high recall without requiring training data or pre-population.
-- vector_cosine_ops indexes the cosine distance (<=>) for Gemini embeddings.
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw 
    ON public.document_chunks 
    USING hnsw (embedding vector_cosine_ops);

-- ------------------------------------------------------------------------------
-- 5. Enable Row Level Security (RLS) & Policies
-- ------------------------------------------------------------------------------
-- Enforce multi-tenant security at the database engine level so users can only
-- read, insert, update, and delete their own chunks.
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;

-- 5.1 SELECT Policy
DROP POLICY IF EXISTS "Users can view their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can view their own document chunks"
    ON public.document_chunks FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

-- 5.2 INSERT Policy
DROP POLICY IF EXISTS "Users can insert their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can insert their own document chunks"
    ON public.document_chunks FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- 5.3 UPDATE Policy
DROP POLICY IF EXISTS "Users can update their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can update their own document chunks"
    ON public.document_chunks FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

-- 5.4 DELETE Policy
DROP POLICY IF EXISTS "Users can delete their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can delete their own document chunks"
    ON public.document_chunks FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- ------------------------------------------------------------------------------
-- 6. RPC Function for Semantic Similarity Search
-- ------------------------------------------------------------------------------
-- Allows FastAPI backend and Supabase client to perform vector similarity queries
-- with cosine distance, similarity thresholding, user isolation, and optional document filters.
CREATE OR REPLACE FUNCTION public.match_document_chunks (
    query_embedding vector(768),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 5,
    filter_user_id uuid DEFAULT NULL,
    filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
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
        dc.user_id,
        dc.chunk_index,
        dc.content,
        dc.page_number,
        (1 - (dc.embedding <=> query_embedding))::FLOAT AS similarity
    FROM public.document_chunks dc
    WHERE
        -- Multi-tenant isolation: ensure user matches authenticated user or provided filter
        (dc.user_id = COALESCE(filter_user_id, auth.uid()))
        -- Optional document filter (query specific document or all user documents)
        AND (filter_document_id IS NULL OR dc.document_id = filter_document_id)
        -- Cosine similarity threshold filter
        AND (1 - (dc.embedding <=> query_embedding)) >= match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

