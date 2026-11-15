"""Supabase repository for persisting and retrieving document summaries."""
import logging
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timezone
import httpx

from ..config.settings import settings

logger = logging.getLogger(__name__)

# In-memory storage cache for document summaries (used for tests and offline fallback)
_in_memory_summary_store: Dict[str, Dict[str, Any]] = {}


class SummaryRepositoryError(Exception):
    """Base exception for summary storage operations."""
    pass


class SummaryRepository:
    """
    Manages persistence and caching of document summaries in Supabase.
    - Persists structured summaries to public.document_summaries.
    - Syncs summary status and JSON to public.documents for rapid listing.
    - Enforces strict multi-tenant ownership.
    - Provides in-memory fallback for local development and automated testing.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.supabase_url = (
            supabase_url
            if supabase_url is not None
            else settings.SUPABASE_URL
        ).rstrip("/")
        self.supabase_key = (
            supabase_key
            if supabase_key is not None
            else (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY)
        )

    def is_configured(self) -> bool:
        """Checks if valid Supabase connection credentials are present."""
        return bool(
            self.supabase_url
            and self.supabase_key
            and self.supabase_key not in ("your_supabase_anon_key_here", "")
            and len(self.supabase_key) > 20
        )

    def get_summary(self, document_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieves existing summary for the document, enforcing user ownership.
        Returns None if not found or unauthorized.
        """
        doc_key = f"{user_id}:{document_id}"

        # In-memory cache lookup
        if doc_key in _in_memory_summary_store:
            cached = _in_memory_summary_store[doc_key]
            if str(cached.get("user_id")) == str(user_id):
                return cached

        if not self.is_configured():
            return None

        # Fetch from Supabase document_summaries
        try:
            endpoint = f"{self.supabase_url}/rest/v1/document_summaries"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            params = {
                "document_id": f"eq.{document_id}",
                "user_id": f"eq.{user_id}",
                "select": "*",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code == 200:
                    records = res.json()
                    if records and len(records) > 0:
                        rec = records[0]
                        # Cache locally
                        _in_memory_summary_store[doc_key] = rec
                        return rec
            return None
        except Exception as e:
            logger.warning(f"Error checking existing summary in Supabase: {e}")
            return _in_memory_summary_store.get(doc_key)

    def save_summary(
        self,
        document_id: UUID,
        user_id: UUID,
        summary_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Upserts structured summary into public.document_summaries and
        updates public.documents status and summary column.
        """
        doc_key = f"{user_id}:{document_id}"
        now_iso = datetime.now(timezone.utc).isoformat()

        record = {
            "document_id": str(document_id),
            "user_id": str(user_id),
            "executive_summary": summary_data.get("executive_summary", ""),
            "main_topics": summary_data.get("main_topics", []),
            "important_points": summary_data.get("important_points", []),
            "key_terminology": summary_data.get("key_terminology", []),
            "important_conclusions": summary_data.get("important_conclusions", []),
            "action_items": summary_data.get("action_items", []),
            "summary_status": "completed",
            "model_used": summary_data.get("model_used", "gemini-2.5-flash"),
            "tokens_used": summary_data.get("tokens_used", 0),
            "hierarchical": summary_data.get("is_hierarchical", False),
            "created_at": summary_data.get("generated_at", now_iso),
            "updated_at": now_iso,
        }

        # Store in-memory
        _in_memory_summary_store[doc_key] = record

        if not self.is_configured():
            logger.info(f"Saved summary in-memory for document {document_id}")
            return record

        try:
            # 1. Upsert into document_summaries
            endpoint = f"{self.supabase_url}/rest/v1/document_summaries?on_conflict=document_id"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=representation",
            }
            with httpx.Client(timeout=15.0) as client:
                res = client.post(endpoint, json=record, headers=headers)
                if res.status_code not in (200, 201):
                    logger.warning(f"document_summaries upsert status {res.status_code}: {res.text}")

            # 2. Sync to documents table
            doc_endpoint = f"{self.supabase_url}/rest/v1/documents?id=eq.{document_id}&user_id=eq.{user_id}"
            doc_payload = {
                "summary": summary_data,
                "summary_status": "completed",
                "summarized_at": now_iso,
            }
            with httpx.Client(timeout=10.0) as client:
                client.patch(doc_endpoint, json=doc_payload, headers=headers)

            return record
        except Exception as e:
            logger.error(f"Failed to persist summary to Supabase: {e}")
            return record

    def update_summary_status(
        self,
        document_id: UUID,
        user_id: UUID,
        status: str,
        error_message: Optional[str] = None,
    ):
        """Updates summary_status on documents table and in-memory cache."""
        doc_key = f"{user_id}:{document_id}"
        if doc_key in _in_memory_summary_store:
            _in_memory_summary_store[doc_key]["summary_status"] = status
            if error_message:
                _in_memory_summary_store[doc_key]["error_message"] = error_message

        if not self.is_configured():
            return

        try:
            endpoint = f"{self.supabase_url}/rest/v1/documents?id=eq.{document_id}&user_id=eq.{user_id}"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            payload = {"summary_status": status}
            with httpx.Client(timeout=5.0) as client:
                client.patch(endpoint, json=payload, headers=headers)
        except Exception as e:
            logger.debug(f"Failed to sync summary status to Supabase: {e}")

    def delete_summary(self, document_id: UUID, user_id: UUID) -> bool:
        """
        Deletes the summary for a document from storage and cache,
        and resets document summary_status to 'not_generated'.
        """
        doc_key = f"{user_id}:{document_id}"
        _in_memory_summary_store.pop(doc_key, None)

        if not self.is_configured():
            logger.info(f"Deleted summary in-memory for document {document_id}")
            return True

        try:
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
            }

            # 1. Delete from document_summaries
            endpoint = f"{self.supabase_url}/rest/v1/document_summaries?document_id=eq.{document_id}&user_id=eq.{user_id}"
            with httpx.Client(timeout=10.0) as client:
                client.delete(endpoint, headers=headers)

            # 2. Reset on documents table
            doc_endpoint = f"{self.supabase_url}/rest/v1/documents?id=eq.{document_id}&user_id=eq.{user_id}"
            doc_payload = {
                "summary": None,
                "summary_status": "not_generated",
                "summarized_at": None,
            }
            with httpx.Client(timeout=10.0) as client:
                client.patch(doc_endpoint, json=doc_payload, headers=headers)

            return True
        except Exception as e:
            logger.error(f"Failed to delete summary from Supabase for document {document_id}: {e}")
            return True


default_summary_repository = SummaryRepository()
