/**
 * ReviewAI Backend API Client
 */

import { ChatAskRequest, ChatAskResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export const checkBackendHealth = async (): Promise<HealthResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed with status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Backend health check error:', error);
    throw error;
  }
};

/**
 * Sends a natural language question to the RAG chat endpoint (POST /api/chat/ask).
 */
export const askRAGQuestion = async (
  payload: ChatAskRequest,
  userId?: string,
  token?: string
): Promise<ChatAskResponse> => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (userId) {
    headers['X-User-ID'] = userId;
  }

  const response = await fetch(`${API_BASE_URL}/api/chat/ask`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Chat query failed with status ${response.status}`);
  }

  return await response.json();
};

export interface StreamRAGCallbacks {
  onStart?: (data: { conversation_id: string; status: string }) => void;
  onToken?: (token: string) => void;
  onSources?: (data: { sources: any[]; document_names: string[]; page_numbers: number[] }) => void;
  onComplete?: (data: { conversation_id: string; grounded: boolean; is_insufficient_context: boolean; finish_reason?: string }) => void;
  onError?: (error: string) => void;
}

/**
 * Sends a natural language question to the real-time RAG SSE streaming endpoint (POST /api/chat/stream).
 * Yields tokens progressively, reports citations on completion, and supports cancellation via AbortSignal.
 */
export const streamRAGQuestion = async (
  payload: ChatAskRequest,
  callbacks: StreamRAGCallbacks,
  userId?: string,
  token?: string,
  signal?: AbortSignal
): Promise<void> => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (userId) {
    headers['X-User-ID'] = userId;
  }

  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    let message = `Streaming request failed with status: ${response.status}`;
    try {
      const parsed = JSON.parse(errorText);
      if (parsed.detail) message = parsed.detail;
    } catch {}
    callbacks.onError?.(message);
    throw new Error(message);
  }

  if (!response.body) {
    throw new Error('Streaming response body is null');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';

      for (const block of blocks) {
        if (!block.trim()) continue;
        const lines = block.split('\n');
        let eventType = 'token';
        let dataStr = '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            dataStr = line.replace('data:', '').trim();
          }
        }

        if (!dataStr) continue;

        try {
          const parsed = JSON.parse(dataStr);
          switch (eventType) {
            case 'start':
              callbacks.onStart?.(parsed);
              break;
            case 'token':
              callbacks.onToken?.(parsed.token || '');
              break;
            case 'sources':
              callbacks.onSources?.(parsed);
              break;
            case 'complete':
              callbacks.onComplete?.(parsed);
              break;
            case 'error':
              callbacks.onError?.(parsed.error || 'Streaming error');
              break;
          }
        } catch {
          // Ignore malformed JSON chunks
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return;
    }
    callbacks.onError?.(err.message || 'Stream connection was interrupted');
    throw err;
  }
};

