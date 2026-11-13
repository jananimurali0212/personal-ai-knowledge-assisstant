"""Unit tests and schema validation for Personal AI Knowledge Assistant database configuration."""
import os
import re
from uuid import uuid4
import pytest
from app.models.chunk import (
    DocumentChunkCreate,
    DocumentChunk,
    SimilaritySearchResult,
    VectorSearchQuery,
    GEMINI_EMBEDDING_DIMENSION,
)
from app.models.document import Document
from app.config.settings import settings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_gemini_embedding_dimension_is_768():
    """Verify that Gemini embedding dimension is 768."""
    assert GEMINI_EMBEDDING_DIMENSION == 768
    assert settings.GEMINI_EMBEDDING_DIMENSIONS == 768
    assert "text-embedding-004" in settings.GEMINI_EMBEDDING_MODEL

def test_chunk_model_accepts_exact_768_dimensions():
    """Verify that DocumentChunkCreate accepts exactly 768-dimensional embeddings."""
    valid_embedding = [0.01] * 768
    chunk = DocumentChunkCreate(
        user_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content="This is a test chunk content for vector indexing.",
        page_number=1,
        embedding=valid_embedding,
    )
    assert len(chunk.embedding) == 768
    assert chunk.chunk_index == 0
    assert chunk.page_number == 1

def test_chunk_model_rejects_incorrect_dimensions():
    """Verify that DocumentChunkCreate rejects embeddings with incorrect dimensions."""
    # Under-dimensioned (767)
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        DocumentChunkCreate(
            user_id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            content="Invalid chunk",
            page_number=1,
            embedding=[0.01] * 767,
        )

    # Over-dimensioned (1536, e.g. OpenAI dimension mistakenly used)
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        DocumentChunkCreate(
            user_id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            content="Invalid chunk",
            page_number=1,
            embedding=[0.01] * 1536,
        )

def test_migration_file_exists_and_valid():
    """Verify migration file contains all required table fields, pgvector, indexes, and RLS."""
    migration_path = os.path.join(PROJECT_ROOT, "supabase", "migrations", "20261113_001_create_document_chunks.sql")
    assert os.path.exists(migration_path), f"Migration file not found at {migration_path}"

    with open(migration_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # 1. pgvector extension
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql

    # 2. Table creation
    assert "CREATE TABLE IF NOT EXISTS public.document_chunks" in sql

    # 3. All 8 required fields
    required_fields = [
        "id UUID",
        "user_id UUID",
        "document_id UUID",
        "chunk_index INTEGER",
        "content TEXT",
        "page_number INTEGER",
        "embedding vector(768)",
        "created_at TIMESTAMP",
    ]
    for field in required_fields:
        assert re.search(re.escape(field), sql, re.IGNORECASE), f"Missing field in SQL: {field}"

    # 4. Foreign key relationships with cascade delete
    assert "REFERENCES auth.users(id) ON DELETE CASCADE" in sql
    assert "REFERENCES public.documents(id) ON DELETE CASCADE" in sql

    # 5. Indexes
    assert "idx_document_chunks_document_id" in sql
    assert "idx_document_chunks_user_id" in sql
    assert "idx_document_chunks_doc_chunk_idx" in sql
    assert "idx_document_chunks_embedding_hnsw" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql

    # 6. Row Level Security
    assert "ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY" in sql
    assert "Users can view their own document chunks" in sql
    assert "Users can insert their own document chunks" in sql
    assert "Users can update their own document chunks" in sql
    assert "Users can delete their own document chunks" in sql

    # 7. Semantic search RPC function
    assert "CREATE OR REPLACE FUNCTION public.match_document_chunks" in sql

def test_root_schema_preserves_documents_table():
    """Verify root supabase_schema.sql preserves existing documents table and data."""
    schema_path = os.path.join(PROJECT_ROOT, "supabase_schema.sql")
    assert os.path.exists(schema_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # Preserves documents table
    assert "CREATE TABLE IF NOT EXISTS public.documents" in sql
    assert "DROP TABLE public.documents" not in sql
    assert "TRUNCATE public.documents" not in sql

    # Preserves storage bucket and policies
    assert "INSERT INTO storage.buckets" in sql
    assert "storage.objects" in sql

    # Includes document_chunks
    assert "CREATE TABLE IF NOT EXISTS public.document_chunks" in sql
    assert "embedding vector(768)" in sql

def test_no_hardcoded_api_keys():
    """Ensure no real API keys are hardcoded in source code."""
    settings_path = os.path.join(PROJECT_ROOT, "backend", "app", "config", "settings.py")
    with open(settings_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "AIza" not in content  # Standard Google API key prefix
    assert "sk-" not in content

