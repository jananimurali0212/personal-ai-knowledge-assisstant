"""Supabase repository for storing and retrieving persistent conversations and chat messages."""
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
import httpx

from ..config.settings import settings
from ..models.conversation import (
    ConversationRecord,
    ChatMessageRecord,
    generate_conversation_title,
)

logger = logging.getLogger(__name__)

# In-memory storage stores used for testing or when Supabase credentials are not configured
_in_memory_conversations_store: Dict[str, Dict[str, Any]] = {}
_in_memory_messages_store: Dict[str, List[Dict[str, Any]]] = {}


class ConversationRepositoryError(Exception):
    """Base exception for conversation storage operations."""
    pass


class ConversationRepository:
    """
    Manages persistence of conversations and chat messages to Supabase.
    Enforces user isolation, timestamp tracking, and cascade deletion.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        raw_url = supabase_url if supabase_url is not None else settings.SUPABASE_URL
        self.supabase_url = raw_url.rstrip("/") if raw_url else ""
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

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    # --------------------------------------------------------------------------
    # CONVERSATIONS OPERATIONS
    # --------------------------------------------------------------------------

    def create_conversation(
        self,
        user_id: UUID,
        title: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
    ) -> ConversationRecord:
        """Creates a new conversation record for the user."""
        cid = conversation_id or uuid4()
        clean_title = (title or "New Conversation").strip()
        now = datetime.now(timezone.utc)

        if not self.is_configured():
            record = {
                "id": str(cid),
                "user_id": str(user_id),
                "title": clean_title,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            _in_memory_conversations_store[str(cid)] = record
            return ConversationRecord(
                id=cid,
                user_id=user_id,
                title=clean_title,
                created_at=now,
                updated_at=now,
                message_count=0,
            )

        try:
            endpoint = f"{self.supabase_url}/rest/v1/conversations"
            payload = {
                "id": str(cid),
                "user_id": str(user_id),
                "title": clean_title,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(endpoint, headers=self._headers(), json=payload)
                if res.status_code in (200, 201):
                    data = res.json()
                    item = data[0] if isinstance(data, list) and len(data) > 0 else payload
                    return ConversationRecord(
                        id=UUID(item["id"]),
                        user_id=UUID(item["user_id"]),
                        title=item["title"],
                        created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                        message_count=0,
                    )
                logger.warning(f"Supabase create_conversation failed with {res.status_code}: {res.text}")
        except Exception as e:
            logger.warning(f"Failed to create conversation in Supabase: {e}")

        # Fallback to in-memory
        record = {
            "id": str(cid),
            "user_id": str(user_id),
            "title": clean_title,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        _in_memory_conversations_store[str(cid)] = record
        return ConversationRecord(
            id=cid,
            user_id=user_id,
            title=clean_title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(self, conversation_id: UUID, user_id: UUID) -> Optional[ConversationRecord]:
        """Fetches a conversation by ID, verifying user ownership."""
        cid_str = str(conversation_id)
        uid_str = str(user_id)

        if not self.is_configured():
            item = _in_memory_conversations_store.get(cid_str)
            if item and item.get("user_id") == uid_str:
                messages = _in_memory_messages_store.get(cid_str, [])
                return ConversationRecord(
                    id=UUID(item["id"]),
                    user_id=UUID(item["user_id"]),
                    title=item["title"],
                    created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                    message_count=len(messages),
                )
            return None

        try:
            endpoint = f"{self.supabase_url}/rest/v1/conversations"
            params = {
                "id": f"eq.{cid_str}",
                "user_id": f"eq.{uid_str}",
                "select": "*",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=self._headers(), params=params)
                if res.status_code == 200:
                    data = res.json()
                    if data and len(data) > 0:
                        item = data[0]
                        # Count messages
                        msg_count = self.get_messages_count(conversation_id, user_id)
                        return ConversationRecord(
                            id=UUID(item["id"]),
                            user_id=UUID(item["user_id"]),
                            title=item["title"],
                            created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                            updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                            message_count=msg_count,
                        )
        except Exception as e:
            logger.warning(f"Failed to fetch conversation from Supabase: {e}")

        # Check in-memory store
        item = _in_memory_conversations_store.get(cid_str)
        if item and item.get("user_id") == uid_str:
            messages = _in_memory_messages_store.get(cid_str, [])
            return ConversationRecord(
                id=UUID(item["id"]),
                user_id=UUID(item["user_id"]),
                title=item["title"],
                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                message_count=len(messages),
            )
        return None

    def list_conversations(self, user_id: UUID, limit: int = 50) -> List[ConversationRecord]:
        """Lists all conversations owned by the user, ordered by updated_at descending."""
        uid_str = str(user_id)

        if not self.is_configured():
            user_convs = [
                c for c in _in_memory_conversations_store.values()
                if c.get("user_id") == uid_str
            ]
            user_convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            result = []
            for item in user_convs[:limit]:
                cid = item["id"]
                messages = _in_memory_messages_store.get(cid, [])
                result.append(
                    ConversationRecord(
                        id=UUID(item["id"]),
                        user_id=UUID(item["user_id"]),
                        title=item["title"],
                        created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                        message_count=len(messages),
                    )
                )
            return result

        try:
            endpoint = f"{self.supabase_url}/rest/v1/conversations"
            params = {
                "user_id": f"eq.{uid_str}",
                "select": "*",
                "order": "updated_at.desc",
                "limit": str(limit),
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=self._headers(), params=params)
                if res.status_code == 200:
                    data = res.json()
                    convs = []
                    for item in data:
                        convs.append(
                            ConversationRecord(
                                id=UUID(item["id"]),
                                user_id=UUID(item["user_id"]),
                                title=item["title"],
                                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                                updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                                message_count=None,
                            )
                        )
                    return convs
        except Exception as e:
            logger.warning(f"Failed to list conversations from Supabase: {e}")

        # Fallback to in-memory store
        user_convs = [
            c for c in _in_memory_conversations_store.values()
            if c.get("user_id") == uid_str
        ]
        user_convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        result = []
        for item in user_convs[:limit]:
            cid = item["id"]
            messages = _in_memory_messages_store.get(cid, [])
            result.append(
                ConversationRecord(
                    id=UUID(item["id"]),
                    user_id=UUID(item["user_id"]),
                    title=item["title"],
                    created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                    message_count=len(messages),
                )
            )
        return result

    def update_conversation_title(
        self,
        conversation_id: UUID,
        user_id: UUID,
        title: str,
    ) -> Optional[ConversationRecord]:
        """Updates the conversation title, verifying ownership."""
        cid_str = str(conversation_id)
        uid_str = str(user_id)
        clean_title = title.strip()
        now = datetime.now(timezone.utc).isoformat()

        if not self.is_configured():
            item = _in_memory_conversations_store.get(cid_str)
            if item and item.get("user_id") == uid_str:
                item["title"] = clean_title
                item["updated_at"] = now
                messages = _in_memory_messages_store.get(cid_str, [])
                return ConversationRecord(
                    id=UUID(item["id"]),
                    user_id=UUID(item["user_id"]),
                    title=clean_title,
                    created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(now.replace("Z", "+00:00")),
                    message_count=len(messages),
                )
            return None

        try:
            endpoint = f"{self.supabase_url}/rest/v1/conversations"
            params = {
                "id": f"eq.{cid_str}",
                "user_id": f"eq.{uid_str}",
            }
            payload = {
                "title": clean_title,
                "updated_at": now,
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.patch(endpoint, headers=self._headers(), params=params, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    if data and len(data) > 0:
                        item = data[0]
                        return ConversationRecord(
                            id=UUID(item["id"]),
                            user_id=UUID(item["user_id"]),
                            title=item["title"],
                            created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                            updated_at=datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00")),
                        )
        except Exception as e:
            logger.warning(f"Failed to update conversation title in Supabase: {e}")

        # Update in-memory
        item = _in_memory_conversations_store.get(cid_str)
        if item and item.get("user_id") == uid_str:
            item["title"] = clean_title
            item["updated_at"] = now
            messages = _in_memory_messages_store.get(cid_str, [])
            return ConversationRecord(
                id=UUID(item["id"]),
                user_id=UUID(item["user_id"]),
                title=clean_title,
                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(now.replace("Z", "+00:00")),
                message_count=len(messages),
            )
        return None

    def touch_conversation(self, conversation_id: UUID, user_id: UUID) -> None:
        """Updates updated_at on the conversation to reflect new activity."""
        cid_str = str(conversation_id)
        uid_str = str(user_id)
        now = datetime.now(timezone.utc).isoformat()

        if cid_str in _in_memory_conversations_store:
            _in_memory_conversations_store[cid_str]["updated_at"] = now

        if not self.is_configured():
            return

        try:
            endpoint = f"{self.supabase_url}/rest/v1/conversations"
            params = {
                "id": f"eq.{cid_str}",
                "user_id": f"eq.{uid_str}",
            }
            payload = {"updated_at": now}
            with httpx.Client(timeout=5.0) as client:
                client.patch(endpoint, headers=self._headers(), params=params, json=payload)
        except Exception as e:
            logger.warning(f"Failed to touch conversation timestamp: {e}")

    def delete_conversation(self, conversation_id: UUID, user_id: UUID) -> bool:
        """Deletes a conversation and all its messages, verifying ownership."""
        cid_str = str(conversation_id)
        uid_str = str(user_id)

        in_mem_deleted = False
        if cid_str in _in_memory_conversations_store:
            if _in_memory_conversations_store[cid_str].get("user_id") == uid_str:
                del _in_memory_conversations_store[cid_str]
                _in_memory_messages_store.pop(cid_str, None)
                in_mem_deleted = True

        if not self.is_configured():
            return in_mem_deleted

        try:
            endpoint = f"{self.supabase_url}/rest/v1/conversations"
            params = {
                "id": f"eq.{cid_str}",
                "user_id": f"eq.{uid_str}",
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.delete(endpoint, headers=self._headers(), params=params)
                if res.status_code in (200, 204):
                    return True
        except Exception as e:
            logger.warning(f"Failed to delete conversation in Supabase: {e}")

        return in_mem_deleted

    # --------------------------------------------------------------------------
    # CHAT MESSAGES OPERATIONS
    # --------------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        message_id: Optional[UUID] = None,
    ) -> ChatMessageRecord:
        """
        Saves a message (user or assistant) in Supabase.
        Associates message with the conversation and updates conversation timestamp.
        """
        mid = message_id or uuid4()
        now = datetime.now(timezone.utc)
        clean_content = content.strip()
        safe_sources = sources or []
        cid_str = str(conversation_id)

        # Ensure in-memory conversation exists
        if cid_str not in _in_memory_conversations_store:
            # Auto-create if missing
            self.create_conversation(user_id=user_id, conversation_id=conversation_id)

        # Store in-memory
        msg_record = {
            "id": str(mid),
            "conversation_id": cid_str,
            "user_id": str(user_id),
            "role": role,
            "content": clean_content,
            "sources": safe_sources,
            "created_at": now.isoformat(),
        }
        if cid_str not in _in_memory_messages_store:
            _in_memory_messages_store[cid_str] = []
        _in_memory_messages_store[cid_str].append(msg_record)

        # Touch timestamp
        self.touch_conversation(conversation_id, user_id)

        if not self.is_configured():
            return ChatMessageRecord(
                id=mid,
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=clean_content,
                sources=safe_sources,
                created_at=now,
            )

        try:
            endpoint = f"{self.supabase_url}/rest/v1/chat_messages"
            payload = {
                "id": str(mid),
                "conversation_id": cid_str,
                "user_id": str(user_id),
                "role": role,
                "content": clean_content,
                "sources": safe_sources,
                "created_at": now.isoformat(),
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.post(endpoint, headers=self._headers(), json=payload)
                if res.status_code in (200, 201):
                    data = res.json()
                    item = data[0] if isinstance(data, list) and len(data) > 0 else payload
                    return ChatMessageRecord(
                        id=UUID(item["id"]),
                        conversation_id=UUID(item["conversation_id"]),
                        user_id=UUID(item["user_id"]),
                        role=item["role"],
                        content=item["content"],
                        sources=item.get("sources", []),
                        created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                    )
                logger.warning(f"Supabase add_message returned {res.status_code}: {res.text}")
        except Exception as e:
            logger.warning(f"Failed to persist message in Supabase: {e}")

        return ChatMessageRecord(
            id=mid,
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=clean_content,
            sources=safe_sources,
            created_at=now,
        )

    def get_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        limit: int = 100,
    ) -> List[ChatMessageRecord]:
        """Retrieves messages for a conversation, ordered chronologically (created_at ASC)."""
        cid_str = str(conversation_id)
        uid_str = str(user_id)

        if not self.is_configured():
            msgs = _in_memory_messages_store.get(cid_str, [])
            filtered = [m for m in msgs if m.get("user_id") == uid_str]
            filtered.sort(key=lambda x: x.get("created_at", ""))
            return [
                ChatMessageRecord(
                    id=UUID(m["id"]),
                    conversation_id=UUID(m["conversation_id"]),
                    user_id=UUID(m["user_id"]),
                    role=m["role"],
                    content=m["content"],
                    sources=m.get("sources", []),
                    created_at=datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")),
                )
                for m in filtered[:limit]
            ]

        try:
            endpoint = f"{self.supabase_url}/rest/v1/chat_messages"
            params = {
                "conversation_id": f"eq.{cid_str}",
                "user_id": f"eq.{uid_str}",
                "select": "*",
                "order": "created_at.asc",
                "limit": str(limit),
            }
            with httpx.Client(timeout=10.0) as client:
                res = client.get(endpoint, headers=self._headers(), params=params)
                if res.status_code == 200:
                    data = res.json()
                    return [
                        ChatMessageRecord(
                            id=UUID(m["id"]),
                            conversation_id=UUID(m["conversation_id"]),
                            user_id=UUID(m["user_id"]),
                            role=m["role"],
                            content=m["content"],
                            sources=m.get("sources", []),
                            created_at=datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")),
                        )
                        for m in data
                    ]
        except Exception as e:
            logger.warning(f"Failed to fetch messages from Supabase: {e}")

        # Fallback to in-memory
        msgs = _in_memory_messages_store.get(cid_str, [])
        filtered = [m for m in msgs if m.get("user_id") == uid_str]
        filtered.sort(key=lambda x: x.get("created_at", ""))
        return [
            ChatMessageRecord(
                id=UUID(m["id"]),
                conversation_id=UUID(m["conversation_id"]),
                user_id=UUID(m["user_id"]),
                role=m["role"],
                content=m["content"],
                sources=m.get("sources", []),
                created_at=datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")),
            )
            for m in filtered[:limit]
        ]

    def get_recent_history(
        self,
        conversation_id: UUID,
        user_id: UUID,
        limit: int = 6,
    ) -> List[Dict[str, str]]:
        """
        Retrieves the most recent N messages formatted as [{"role": "...", "content": "..."}, ...]
        for conversational follow-up context. Limits history to prevent token explosion.
        """
        all_msgs = self.get_messages(conversation_id, user_id, limit=50)
        recent = all_msgs[-limit:] if len(all_msgs) > limit else all_msgs
        return [{"role": m.role, "content": m.content} for m in recent]

    def get_messages_count(self, conversation_id: UUID, user_id: UUID) -> int:
        """Counts total messages in a conversation."""
        cid_str = str(conversation_id)
        if not self.is_configured():
            msgs = _in_memory_messages_store.get(cid_str, [])
            return len(msgs)

        try:
            endpoint = f"{self.supabase_url}/rest/v1/chat_messages"
            params = {
                "conversation_id": f"eq.{cid_str}",
                "user_id": f"eq.{str(user_id)}",
                "select": "id",
            }
            headers = self._headers()
            headers["Prefer"] = "count=exact"
            headers["Range"] = "0-0"
            with httpx.Client(timeout=5.0) as client:
                res = client.get(endpoint, headers=headers, params=params)
                if res.status_code in (200, 206):
                    cr = res.headers.get("content-range", "")
                    if "/" in cr:
                        return int(cr.split("/")[-1])
        except Exception:
            pass

        return len(_in_memory_messages_store.get(cid_str, []))


default_conversation_repository = ConversationRepository()
