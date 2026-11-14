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
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content) || to_tsvector('simple', content)
    ) STORED,
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

-- Full-Text Search Lexical Index (GIN)
-- Provides high-performance keyword and exact technical term search
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts_gin
    ON public.document_chunks 
    USING gin (fts);

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
-- PART 4.1: CONVERSATIONS AND CHAT MESSAGES TABLES & RLS
-- ==============================================================================

CREATE TABLE IF NOT EXISTS public.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE DEFAULT auth.uid(),
    title TEXT NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON public.conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON public.conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS public.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE DEFAULT auth.uid(),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON public.chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON public.chat_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conv_created ON public.chat_messages(conversation_id, created_at ASC);

ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own conversations" ON public.conversations;
CREATE POLICY "Users can view their own conversations"
    ON public.conversations FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can create their own conversations" ON public.conversations;
CREATE POLICY "Users can create their own conversations"
    ON public.conversations FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own conversations" ON public.conversations;
CREATE POLICY "Users can update their own conversations"
    ON public.conversations FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own conversations" ON public.conversations;
CREATE POLICY "Users can delete their own conversations"
    ON public.conversations FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view messages in their own conversations" ON public.chat_messages;
CREATE POLICY "Users can view messages in their own conversations"
    ON public.chat_messages FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert messages into their own conversations" ON public.chat_messages;
CREATE POLICY "Users can insert messages into their own conversations"
    ON public.chat_messages FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete messages in their own conversations" ON public.chat_messages;
CREATE POLICY "Users can delete messages in their own conversations"
    ON public.chat_messages FOR DELETE
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

-- ==============================================================================
-- PART 6: KEYWORD FULL-TEXT SEARCH RPC FUNCTION
-- ==============================================================================

-- Stored procedure for lexical full-text search across user document chunks
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
        (dc.user_id = COALESCE(filter_user_id, auth.uid()))
        AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids))
        AND dc.fts @@ parsed_query
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- ==============================================================================
-- PART 7: HYBRID SEARCH RPC FUNCTION (RECIPROCAL RANK FUSION)
-- ==============================================================================

-- Stored procedure combining vector similarity and keyword search with RRF
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

-- ==============================================================================
-- PART 8: DOCUMENT SUMMARIES & METADATA (DAY 4)
-- ==============================================================================

-- 8.1 Extend documents metadata table with summary tracking columns
ALTER TABLE public.documents
ADD COLUMN IF NOT EXISTS summary JSONB DEFAULT NULL,
ADD COLUMN IF NOT EXISTS summary_status TEXT NOT NULL DEFAULT 'not_generated',
ADD COLUMN IF NOT EXISTS summarized_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_summary_status ON public.documents(summary_status);

-- 8.2 Create dedicated document_summaries table
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

CREATE INDEX IF NOT EXISTS idx_document_summaries_user_id ON public.document_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_document_summaries_doc_id ON public.document_summaries(document_id);

ALTER TABLE public.document_summaries ENABLE ROW LEVEL SECURITY;

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

-- ==============================================================================
-- PART 9: QUIZZES, QUESTIONS, AND QUIZ ATTEMPTS (DAY 4)
-- ==============================================================================

-- 9.1 Quizzes Table
CREATE TABLE IF NOT EXISTS public.quizzes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    difficulty TEXT NOT NULL DEFAULT 'medium',
    total_questions INT NOT NULL DEFAULT 5,
    question_types JSONB NOT NULL DEFAULT '["multiple_choice", "true_false", "short_answer"]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quizzes_user_id ON public.quizzes(user_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_document_id ON public.quizzes(document_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_created_at ON public.quizzes(created_at DESC);

-- 9.2 Quiz Questions Table
CREATE TABLE IF NOT EXISTS public.quiz_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID NOT NULL REFERENCES public.quizzes(id) ON DELETE CASCADE,
    question_index INT NOT NULL,
    question_type TEXT NOT NULL,
    question_text TEXT NOT NULL,
    options JSONB DEFAULT '[]'::jsonb,
    correct_answer TEXT NOT NULL,
    explanation TEXT NOT NULL,
    source_excerpt TEXT NOT NULL,
    page_number INT DEFAULT NULL,
    chunk_id UUID DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quiz_questions_quiz_id ON public.quiz_questions(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_index ON public.quiz_questions(quiz_id, question_index);

-- 9.3 Quiz Attempts Table
CREATE TABLE IF NOT EXISTS public.quiz_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID NOT NULL REFERENCES public.quizzes(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user_answers JSONB NOT NULL DEFAULT '{}'::jsonb,
    score FLOAT NOT NULL DEFAULT 0.0,
    total_points FLOAT NOT NULL DEFAULT 0.0,
    percentage FLOAT NOT NULL DEFAULT 0.0,
    review_details JSONB NOT NULL DEFAULT '[]'::jsonb,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quiz_attempts_user_id ON public.quiz_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_id ON public.quiz_attempts(quiz_id);

-- 9.4 Enable Row Level Security (RLS)
ALTER TABLE public.quizzes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_attempts ENABLE ROW LEVEL SECURITY;

-- 9.5 RLS Policies for Quizzes
DROP POLICY IF EXISTS "Users can view their own quizzes" ON public.quizzes;
CREATE POLICY "Users can view their own quizzes"
    ON public.quizzes FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own quizzes" ON public.quizzes;
CREATE POLICY "Users can insert their own quizzes"
    ON public.quizzes FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own quizzes" ON public.quizzes;
CREATE POLICY "Users can update their own quizzes"
    ON public.quizzes FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own quizzes" ON public.quizzes;
CREATE POLICY "Users can delete their own quizzes"
    ON public.quizzes FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- 9.6 RLS Policies for Quiz Questions
DROP POLICY IF EXISTS "Users can view their own quiz questions" ON public.quiz_questions;
CREATE POLICY "Users can view their own quiz questions"
    ON public.quiz_questions FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.quizzes q
            WHERE q.id = quiz_questions.quiz_id
            AND q.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can insert their own quiz questions" ON public.quiz_questions;
CREATE POLICY "Users can insert their own quiz questions"
    ON public.quiz_questions FOR INSERT
    TO authenticated
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.quizzes q
            WHERE q.id = quiz_questions.quiz_id
            AND q.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can delete their own quiz questions" ON public.quiz_questions;
CREATE POLICY "Users can delete their own quiz questions"
    ON public.quiz_questions FOR DELETE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.quizzes q
            WHERE q.id = quiz_questions.quiz_id
            AND q.user_id = auth.uid()
        )
    );

-- 9.7 RLS Policies for Quiz Attempts
DROP POLICY IF EXISTS "Users can view their own quiz attempts" ON public.quiz_attempts;
CREATE POLICY "Users can view their own quiz attempts"
    ON public.quiz_attempts FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own quiz attempts" ON public.quiz_attempts;
CREATE POLICY "Users can insert their own quiz attempts"
    ON public.quiz_attempts FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own quiz attempts" ON public.quiz_attempts;
CREATE POLICY "Users can delete their own quiz attempts"
    ON public.quiz_attempts FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);
