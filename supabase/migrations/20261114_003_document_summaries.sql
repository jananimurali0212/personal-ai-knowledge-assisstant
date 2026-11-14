-- ==============================================================================
-- Migration: Add Document Summarization Tables & Metadata Columns
-- Migration Date: 2026-11-14
-- Purpose: Support AI-powered structured hierarchical summarization, caching,
--          and multi-tenant access control for document summaries.
-- ==============================================================================

-- 1. Extend documents metadata table with summary tracking columns
ALTER TABLE public.documents
ADD COLUMN IF NOT EXISTS summary JSONB DEFAULT NULL,
ADD COLUMN IF NOT EXISTS summary_status TEXT NOT NULL DEFAULT 'not_generated',
ADD COLUMN IF NOT EXISTS summarized_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;

-- Add index on summary_status for quick filtering
CREATE INDEX IF NOT EXISTS idx_documents_summary_status ON public.documents(summary_status);

-- 2. Create dedicated document_summaries table
CREATE TABLE IF NOT EXISTS public.document_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    executive_summary TEXT NOT NULL,
    main_topics JSONB NOT NULL DEFAULT '[]'::jsonb,
    important_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    key_terminology JSONB NOT NULL DEFAULT '[]'::jsonb,
    important_conclusions JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary_status TEXT NOT NULL DEFAULT 'completed',
    model_used TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    hierarchical BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    CONSTRAINT uq_document_summaries_document_id UNIQUE(document_id)
);

-- 3. Indexes for fast user lookup and join resolution
CREATE INDEX IF NOT EXISTS idx_document_summaries_user_id ON public.document_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_document_summaries_doc_id ON public.document_summaries(document_id);

-- 4. Enable Row Level Security (RLS) on document_summaries
ALTER TABLE public.document_summaries ENABLE ROW LEVEL SECURITY;

-- 5. RLS Policies: Authenticated users can only read and manage their own summaries
DROP POLICY IF EXISTS "Users can view their own document summaries" ON public.document_summaries;
CREATE POLICY "Users can view their own document summaries"
    ON public.document_summaries FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own document summaries" ON public.document_summaries;
CREATE POLICY "Users can insert their own document summaries"
    ON public.document_summaries FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own document summaries" ON public.document_summaries;
CREATE POLICY "Users can update their own document summaries"
    ON public.document_summaries FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own document summaries" ON public.document_summaries;
CREATE POLICY "Users can delete their own document summaries"
    ON public.document_summaries FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);
