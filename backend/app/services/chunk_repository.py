"""Supabase repository for storing and retrieving document chunks and vector embeddings."""
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
import httpx

from ..config.settings import settings
from ..models.chunk import DocumentChunkCreate

logger = logging.getLogger(__name__)

# In-memory storage cache used for test environments or when offline
_in_memory_chunks_store: Dict[str, List[Dict[str, Any]]] = {}

class ChunkRepositoryError(Exception):
    """Base exception for chunk storage operations."""
    pass

class ChunkRepository:
    """
    Manages persistence of document chunks and their 768-dimensional embeddings to Supabase.
    Implements idempotent upsert on conflict (document_id, chunk_index) to prevent duplicate chunks.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.supabase_url = (supabase_url or settings.SUPABASE_URL).rstrip("/")
        self.supabase_key = supabase_key or settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY

    def is_configured(self) -> bool:
        """Checks if valid Supabase connection credentials are present."""
        return bool(
            self.supabase_url 
            and self.supabase_key 
            and self.supabase_key not in ("your_supabase_anon_key_here", "")
            and len(self.supabase_key) > 20
        )

    def get_existing_chunks_count(self, document_id: UUID, user_id: UUID) -> int:
        """
        Returns the number of existing chunks already stored for this document.
        Used to prevent unnecessary re-embedding if chunks are already present.
        """
        doc_key = f"{user_id}:{document_id}"

        if not self.is_configured():
            return len(_in_memory_chunks_store.get(doc_key, []))

        try:
            endpoint = f"{self.supabase_url}/rest/v1/document_chunks"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Range-Unit": "items",
                "Range": "0-0",
                "Prefer": "count=exact",
            }
            params = {
                "document_id": f"eq.{document_id}",
                "user_id": f"eq.{user_id}",
                "select": "id",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code in (200, 206):
                    content_range = res.headers.get("content-range", "")
                    if "/" in content_range:
                        return int(content_range.split("/")[-1])
            return 0
        except Exception as e:
            logger.warning(f"Failed to check existing chunks in Supabase: {e}")
            return len(_in_memory_chunks_store.get(doc_key, []))

    def save_chunks(
        self,
        chunks: List[DocumentChunkCreate],
        overwrite: bool = True
    ) -> int:
        """
        Persists chunks and vector embeddings into public.document_chunks.
        Safe against duplicate insertions via merge-duplicates (upsert on document_id, chunk_index).
        """
        if not chunks:
            return 0

        document_id = chunks[0].document_id
        user_id = chunks[0].user_id
        doc_key = f"{user_id}:{document_id}"

        payload = [
            {
                "document_id": str(c.document_id),
                "user_id": str(c.user_id),
                "chunk_index": c.chunk_index,
                "content": c.content,
                "page_number": c.page_number,
                "embedding": c.embedding,
            }
            for c in chunks
        ]

        # Always update in-memory cache for fast lookup/test verification
        if overwrite or doc_key not in _in_memory_chunks_store:
            _in_memory_chunks_store[doc_key] = payload
        else:
            _in_memory_chunks_store[doc_key].extend(payload)

        if not self.is_configured():
            logger.info(
                f"Saved {len(payload)} chunks to in-memory store for document {document_id} "
                "(Supabase credentials not configured in environment)."
            )
            return len(payload)

        # Persist to Supabase PostgREST
        try:
            endpoint = f"{self.supabase_url}/rest/v1/document_chunks?on_conflict=document_id,chunk_index"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=representation",
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(endpoint, json=payload, headers=headers)
                if response.status_code not in (200, 201):
                    logger.error(f"Supabase chunks insert failed ({response.status_code}): {response.text}")
                    raise ChunkRepositoryError(
                        f"Supabase chunk insertion error ({response.status_code}): {response.text}"
                    )

            logger.info(f"Successfully persisted {len(payload)} chunks with embeddings to Supabase.")
            return len(payload)

        except Exception as e:
            logger.error(f"Failed to persist chunks to Supabase: {e}")
            raise ChunkRepositoryError(f"Database insertion failed: {str(e)}") from e

    def get_chunks_by_document(self, document_id: UUID, user_id: UUID) -> List[Dict[str, Any]]:
        """Retrieves all chunks for a document, enforcing user ownership."""
        doc_key = f"{user_id}:{document_id}"

        if not self.is_configured():
            return _in_memory_chunks_store.get(doc_key, [])

        try:
            endpoint = f"{self.supabase_url}/rest/v1/document_chunks"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            params = {
                "document_id": f"eq.{document_id}",
                "user_id": f"eq.{user_id}",
                "select": "id,document_id,user_id,chunk_index,content,page_number,created_at",
                "order": "chunk_index.asc",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code == 200:
                    return res.json()
            return _in_memory_chunks_store.get(doc_key, [])
        except Exception as e:
            logger.warning(f"Error fetching chunks from Supabase: {e}")
            return _in_memory_chunks_store.get(doc_key, [])

default_chunk_repository = ChunkRepository()

