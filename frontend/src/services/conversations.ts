/**
 * Conversation history and chat messages service for frontend
 */
import {
  ConversationRecord,
  ConversationDetailResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const getHeaders = (userId?: string, token?: string): Record<string, string> => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (userId) {
    headers['X-User-ID'] = userId;
  }
  return headers;
};

/**
 * Fetch all conversations for the authenticated user.
 */
export const fetchUserConversations = async (
  userId?: string,
  token?: string
): Promise<ConversationRecord[]> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/conversations`, {
      method: 'GET',
      headers: getHeaders(userId, token),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to fetch conversations (status ${res.status})`);
    }

    return await res.json();
  } catch (error) {
    console.error('Error fetching conversations:', error);
    return [];
  }
};

/**
 * Fetch a single conversation and its full message history.
 */
export const fetchConversationDetails = async (
  conversationId: string,
  userId?: string,
  token?: string
): Promise<ConversationDetailResponse | null> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`, {
      method: 'GET',
      headers: getHeaders(userId, token),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to load conversation (status ${res.status})`);
    }

    return await res.json();
  } catch (error) {
    console.error(`Error loading conversation ${conversationId}:`, error);
    return null;
  }
};

/**
 * Create a new conversation explicitly.
 */
export const createUserConversation = async (
  title?: string,
  userId?: string,
  token?: string
): Promise<ConversationRecord | null> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/conversations`, {
      method: 'POST',
      headers: getHeaders(userId, token),
      body: JSON.stringify({ title }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to create conversation (status ${res.status})`);
    }

    return await res.json();
  } catch (error) {
    console.error('Error creating conversation:', error);
    return null;
  }
};

/**
 * Rename/update the title of an existing conversation.
 */
export const updateUserConversationTitle = async (
  conversationId: string,
  title: string,
  userId?: string,
  token?: string
): Promise<ConversationRecord | null> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`, {
      method: 'PATCH',
      headers: getHeaders(userId, token),
      body: JSON.stringify({ title }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to rename conversation (status ${res.status})`);
    }

    return await res.json();
  } catch (error) {
    console.error(`Error renaming conversation ${conversationId}:`, error);
    return null;
  }
};

/**
 * Delete a conversation and its messages.
 */
export const deleteUserConversation = async (
  conversationId: string,
  userId?: string,
  token?: string
): Promise<boolean> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`, {
      method: 'DELETE',
      headers: getHeaders(userId, token),
    });

    return res.ok;
  } catch (error) {
    console.error(`Error deleting conversation ${conversationId}:`, error);
    return false;
  }
};
