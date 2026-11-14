-- ==============================================================================
-- Migration: Add Quizzes, Quiz Questions, and Quiz Attempts Tables
-- Migration Date: 2026-11-14
-- Purpose: Support AI-powered quiz generation from uploaded documents,
--          grounded questions, answer validation, scoring, and attempt tracking.
-- ==============================================================================

-- 1. Create quizzes table
CREATE TABLE IF NOT EXISTS public.quizzes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    difficulty TEXT NOT NULL DEFAULT 'medium', -- 'easy', 'medium', 'hard'
    total_questions INT NOT NULL DEFAULT 5,
    question_types JSONB NOT NULL DEFAULT '["multiple_choice", "true_false", "short_answer"]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indexes for quizzes
CREATE INDEX IF NOT EXISTS idx_quizzes_user_id ON public.quizzes(user_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_document_id ON public.quizzes(document_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_created_at ON public.quizzes(created_at DESC);

-- 2. Create quiz_questions table
CREATE TABLE IF NOT EXISTS public.quiz_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID NOT NULL REFERENCES public.quizzes(id) ON DELETE CASCADE,
    question_index INT NOT NULL,
    question_type TEXT NOT NULL, -- 'multiple_choice', 'true_false', 'short_answer'
    question_text TEXT NOT NULL,
    options JSONB DEFAULT '[]'::jsonb, -- array of options for multiple_choice & true_false
    correct_answer TEXT NOT NULL,
    explanation TEXT NOT NULL,
    source_excerpt TEXT NOT NULL,
    page_number INT DEFAULT NULL,
    chunk_id UUID DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indexes for quiz_questions
CREATE INDEX IF NOT EXISTS idx_quiz_questions_quiz_id ON public.quiz_questions(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_questions_index ON public.quiz_questions(quiz_id, question_index);

-- 3. Create quiz_attempts table (stores completed attempts & scores)
CREATE TABLE IF NOT EXISTS public.quiz_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID NOT NULL REFERENCES public.quizzes(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    user_answers JSONB NOT NULL DEFAULT '{}'::jsonb, -- { question_id: answer }
    score FLOAT NOT NULL DEFAULT 0.0,
    total_points FLOAT NOT NULL DEFAULT 0.0,
    percentage FLOAT NOT NULL DEFAULT 0.0,
    review_details JSONB NOT NULL DEFAULT '[]'::jsonb, -- detailed per-question grading breakdown
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indexes for quiz_attempts
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_user_id ON public.quiz_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_id ON public.quiz_attempts(quiz_id);

-- 4. Enable Row Level Security (RLS) on all quiz tables
ALTER TABLE public.quizzes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quiz_attempts ENABLE ROW LEVEL SECURITY;

-- 5. RLS Policies for public.quizzes
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

-- 6. RLS Policies for public.quiz_questions (join on quiz ownership)
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

-- 7. RLS Policies for public.quiz_attempts
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
