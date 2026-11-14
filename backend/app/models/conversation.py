"""Pydantic models for persistent conversation history and chat messages."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import re
from pydantic import BaseModel, Field, ConfigDict


def generate_conversation_title(question: str, max_length: int = 50) -> str:
    """
    Automatically generates a concise, readable conversation title from the first question.
    Strips leading punctuation and question marks, cleans extra spaces, and caps length.
    """
    if not question or not question.strip():
        return "New Conversation"

    clean = question.strip()
    # Strip wrapping quotes
    clean = re.sub(r'^["\'`]+|["\'`]+$', '', clean).strip()

    # If question starts with common prefixes, keep it natural
    clean = re.sub(r'\s+', ' ', clean)

    if len(clean) <= max_length:
        return clean

    # Truncate at word boundary if possible
    truncated = clean[:max_length].rstrip()
    last_space = truncated.rfind(' ')
    if last_space > 20:
        truncated = truncated[:last_space]

    return f"{truncated}..."


class ConversationBase(BaseModel):
    title: str = Field(default="New Conversation", description="User-visible conversation title")


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, description="Optional title. If omitted, generated from first question")


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=120, description="Updated conversation title")


class ConversationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = Field(default=None, description="Total messages in conversation")


class ChatMessageRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    user_id: UUID
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Grounding source citations and metadata")
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    conversation: ConversationRecord
    messages: List[ChatMessageRecord] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    conversations: List[ConversationRecord]
    total: int
