-- ==============================================================================
-- Personal AI Knowledge Assistant: Database Schema & Storage Configuration
-- ==============================================================================

-- ==============================================================================
-- PART 1: CORE EXTENSIONS
-- ==============================================================================

-- 1.1 Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1.2 Enable pgvector Extension for Vector Storage & Similarity Search
CREATE EXTENSION IF NOT EXISTS vector;

-- ==============================================================================
-- PART 2: DOCUMENTS METADATA TABLE (DAY 1)
-- ==============================================================================

-- Create the Documents Metadata Table
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE DEFAULT auth.uid(),
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL, -- 'pdf', 'docx', 'txt'
    file_size BIGINT NOT NULL DEFAULT 0,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded', -- 'uploaded', 'processing', 'ready', 'failed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Add Index for fast lookups by user and timestamp
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON public.documents(created_at DESC);

-- Enable Row Level Security (RLS) on Documents Table
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- Create RLS Policies for Documents Table
-- Users can ONLY access their own records based on their authenticated user ID
DROP POLICY IF EXISTS "Users can view their own documents" ON public.documents;
CREATE POLICY "Users can view their own documents"
    ON public.documents FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own documents" ON public.documents;
CREATE POLICY "Users can insert their own documents"
    ON public.documents FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own documents" ON public.documents;
CREATE POLICY "Users can update their own documents"
    ON public.documents FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own documents" ON public.documents;
CREATE POLICY "Users can delete their own documents"
    ON public.documents FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- ==============================================================================
-- PART 3: SUPABASE STORAGE CONFIGURATION (DAY 1)
-- ==============================================================================

-- Create the private bucket if it doesn't already exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'documents',
    'documents',
    FALSE, -- Private bucket: access requires authenticated RLS check
    52428800, -- 50 MB per file limit
    ARRAY[
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain'
    ]
)
ON CONFLICT (id) DO UPDATE SET
    public = FALSE,
    file_size_limit = 52428800,
    allowed_mime_types = ARRAY[
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain'
    ];

-- Storage RLS Policies
-- Files are organized into user-specific prefixes: `${user.id}/${documentId}/${fileName}`
-- Users can only upload, read, update, and delete files inside their own folder.

DROP POLICY IF EXISTS "Users can upload their own documents" ON storage.objects;
CREATE POLICY "Users can upload their own documents"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (
        bucket_id = 'documents' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

DROP POLICY IF EXISTS "Users can view and download their own documents" ON storage.objects;
CREATE POLICY "Users can view and download their own documents"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (
        bucket_id = 'documents' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

DROP POLICY IF EXISTS "Users can delete their own documents" ON storage.objects;
CREATE POLICY "Users can delete their own documents"
    ON storage.objects FOR DELETE
    TO authenticated
    USING (
        bucket_id = 'documents' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

DROP POLICY IF EXISTS "Users can update their own documents" ON storage.objects;
CREATE POLICY "Users can update their own documents"
    ON storage.objects FOR UPDATE
    TO authenticated
    USING (
        bucket_id = 'documents' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- ==============================================================================
-- PART 4: DOCUMENT CHUNKS & VECTOR STORAGE (DAY 2)
-- ==============================================================================

-- Create the Document Chunks Table
-- Configured for Google Gemini Embedding API (text-embedding-004: 768 dimensions)
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

-- Indexes for Relational Filtering
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
    ON public.document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id 
    ON public.document_chunks(user_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_chunk_idx 
    ON public.document_chunks(document_id, chunk_index);

-- Vector Similarity Search Index (HNSW with Cosine Distance)
-- Provides sub-linear search latency and high recall for 768-dim embeddings
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw 
    ON public.document_chunks 
    USING hnsw (embedding vector_cosine_ops);

-- Enable Row Level Security (RLS) on document_chunks Table
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;

-- RLS Policies for document_chunks Table
DROP POLICY IF EXISTS "Users can view their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can view their own document chunks"
    ON public.document_chunks FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can insert their own document chunks"
    ON public.document_chunks FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can update their own document chunks"
    ON public.document_chunks FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can delete their own document chunks"
    ON public.document_chunks FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- ==============================================================================
-- PART 5: SEMANTIC SEARCH RPC FUNCTION
-- ==============================================================================

-- Stored procedure for top-k cosine similarity search across user document chunks
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
        -- Multi-tenant isolation: ensure user matches authenticated user or provided filter
        (dc.user_id = COALESCE(filter_user_id, auth.uid()))
        -- Optional document filter (query specific documents or all user documents)
        AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
        -- Cosine similarity threshold filter
        AND (1 - (dc.embedding <=> query_embedding)) >= match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;
