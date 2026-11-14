"""Conversations and persistent chat history routes."""
import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_authenticated_user
from ..models.conversation import (
    ConversationRecord,
    ConversationCreate,
    ConversationUpdate,
    ConversationDetailResponse,
    ConversationListResponse,
    ChatMessageRecord,
)
from ..services.conversation_repository import (
    default_conversation_repository,
    ConversationRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=List[ConversationRecord])
def list_user_conversations(
    limit: int = 50,
    current_user: UUID = Depends(get_authenticated_user),
    repo: ConversationRepository = Depends(lambda: default_conversation_repository),
):
    """
    List all conversations owned by the authenticated user.
    Results are sorted by updated_at descending (most recent first).
    """
    try:
        conversations = repo.list_conversations(user_id=current_user, limit=limit)
        return conversations
    except Exception as e:
        logger.exception(f"Failed to list conversations for user={current_user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history.",
        )


@router.post("", response_model=ConversationRecord, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: Optional[ConversationCreate] = None,
    current_user: UUID = Depends(get_authenticated_user),
    repo: ConversationRepository = Depends(lambda: default_conversation_repository),
):
    """
    Create a new conversation for the authenticated user.
    """
    title = payload.title if payload and payload.title else "New Conversation"
    try:
        conv = repo.create_conversation(user_id=current_user, title=title)
        return conv
    except Exception as e:
        logger.exception(f"Failed to create conversation for user={current_user}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation.",
        )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_details(
    conversation_id: UUID,
    current_user: UUID = Depends(get_authenticated_user),
    repo: ConversationRepository = Depends(lambda: default_conversation_repository),
):
    """
    Load an existing conversation and its full message history.
    Strictly verifies ownership before returning data.
    """
    conv = repo.get_conversation(conversation_id=conversation_id, user_id=current_user)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )

    try:
        messages = repo.get_messages(conversation_id=conversation_id, user_id=current_user)
        return ConversationDetailResponse(
            conversation=conv,
            messages=messages,
        )
    except Exception as e:
        logger.exception(f"Failed to load messages for conversation={conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load conversation messages.",
        )


@router.patch("/{conversation_id}", response_model=ConversationRecord)
def update_conversation_title(
    conversation_id: UUID,
    payload: ConversationUpdate,
    current_user: UUID = Depends(get_authenticated_user),
    repo: ConversationRepository = Depends(lambda: default_conversation_repository),
):
    """
    Update the title of an existing conversation.
    Verifies user ownership before updating.
    """
    conv = repo.update_conversation_title(
        conversation_id=conversation_id,
        user_id=current_user,
        title=payload.title,
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )
    return conv


@router.delete("/{conversation_id}", status_code=status.HTTP_200_OK)
def delete_user_conversation(
    conversation_id: UUID,
    current_user: UUID = Depends(get_authenticated_user),
    repo: ConversationRepository = Depends(lambda: default_conversation_repository),
):
    """
    Delete a conversation and all its associated messages.
    Verifies user ownership before deleting.
    """
    deleted = repo.delete_conversation(conversation_id=conversation_id, user_id=current_user)
    if not deleted:
        # Check if conversation existed
        conv = repo.get_conversation(conversation_id=conversation_id, user_id=current_user)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied.",
            )

    return {"deleted": True, "conversation_id": str(conversation_id)}
