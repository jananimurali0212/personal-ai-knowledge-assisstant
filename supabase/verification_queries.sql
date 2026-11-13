-- ==============================================================================
-- Verification Queries: Personal AI Knowledge Assistant Schema
-- Run these queries in the Supabase SQL Editor to verify the Day 2 setup.
-- ==============================================================================

-- 1. Verify pgvector extension is enabled
SELECT 
    extname AS extension_name, 
    extversion AS version,
    installed
FROM pg_catalog.pg_available_extensions 
WHERE extname = 'vector';

-- 2. Verify document_chunks table columns and data types (ensure vector(768))
SELECT 
    column_name, 
    data_type, 
    udt_name,
    is_nullable, 
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'document_chunks'
ORDER BY ordinal_position;

-- 3. Verify Foreign Key relationships
SELECT
    tc.table_schema, 
    tc.constraint_name, 
    tc.table_name, 
    kcu.column_name, 
    ccu.table_schema AS foreign_table_schema,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
  AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
  AND ccu.table_schema = tc.table_schema
JOIN information_schema.referential_constraints AS rc
  ON tc.constraint_name = rc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' 
  AND tc.table_name = 'document_chunks';

-- 4. Verify all indexes (B-Tree + HNSW Vector Index)
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'document_chunks';

-- 5. Verify Row Level Security is active on document_chunks
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables
WHERE tablename IN ('documents', 'document_chunks');

-- 6. Verify Row Level Security Policies on document_chunks
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies
WHERE tablename = 'document_chunks';

-- 7. Verify match_document_chunks RPC function exists
SELECT 
    routine_name, 
    routine_type, 
    security_type,
    data_type
FROM information_schema.routines
WHERE routine_schema = 'public'
  AND routine_name = 'match_document_chunks';

-- 8. Verify existing documents table and data are preserved
SELECT 
    COUNT(*) AS total_existing_documents,
    COUNT(DISTINCT user_id) AS distinct_users_count
FROM public.documents;

