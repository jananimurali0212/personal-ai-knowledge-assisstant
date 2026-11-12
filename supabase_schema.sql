-- ==============================================================================
-- Personal AI Knowledge Assistant: Database Schema & Storage Configuration
-- ==============================================================================

-- 1. Enable UUID Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Create the Documents Metadata Table
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

-- 3. Enable Row Level Security (RLS) on Documents Table
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- 4. Create RLS Policies for Documents Table
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

-- 5. Supabase Storage Configuration for Private 'documents' Bucket
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

-- 6. Storage RLS Policies
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

