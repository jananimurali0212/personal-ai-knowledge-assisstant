/**
 * API client service for AI-powered Quiz generation, taking, and reviewing.
 */
import {
  GenerateQuizRequest,
  QuizResponse,
  QuizSummary,
  QuizSubmitRequest,
  QuizSubmitResponse,
  QuizAttemptHistoryItem,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface ApiResult<T> {
  data: T | null;
  error: string | null;
}

const buildHeaders = (userId: string, token?: string): Record<string, string> => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-User-ID': userId,
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
};

/**
 * Generates a new quiz for an uploaded document.
 * Endpoint: POST /api/documents/{documentId}/generate-quiz
 */
export const generateQuiz = async (
  documentId: string,
  request: GenerateQuizRequest,
  userId: string,
  token?: string
): Promise<ApiResult<QuizResponse>> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/${documentId}/generate-quiz`, {
      method: 'POST',
      headers: buildHeaders(userId, token),
      body: JSON.stringify(request),
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Quiz generation failed with status ${res.status}`,
      };
    }

    return { data: body as QuizResponse, error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while generating quiz.',
    };
  }
};

/**
 * Lists all quizzes previously generated for a document.
 * Endpoint: GET /api/documents/{documentId}/quizzes
 */
export const fetchDocumentQuizzes = async (
  documentId: string,
  userId: string,
  token?: string
): Promise<ApiResult<QuizSummary[]>> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/${documentId}/quizzes`, {
      method: 'GET',
      headers: buildHeaders(userId, token),
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Failed to fetch quizzes (status ${res.status})`,
      };
    }

    return { data: body as QuizSummary[], error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while fetching quizzes.',
    };
  }
};

/**
 * Fetches an interactive quiz session (student view).
 * Endpoint: GET /api/quizzes/{quizId}
 */
export const fetchQuizStudentView = async (
  quizId: string,
  userId: string,
  token?: string
): Promise<ApiResult<QuizResponse>> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/quizzes/${quizId}`, {
      method: 'GET',
      headers: buildHeaders(userId, token),
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Failed to fetch quiz (status ${res.status})`,
      };
    }

    return { data: body as QuizResponse, error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while fetching quiz.',
    };
  }
};

/**
 * Submits student answers for grading and returns review with explanations.
 * Endpoint: POST /api/quizzes/{quizId}/submit
 */
export const submitQuiz = async (
  quizId: string,
  request: QuizSubmitRequest,
  userId: string,
  token?: string
): Promise<ApiResult<QuizSubmitResponse>> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/quizzes/${quizId}/submit`, {
      method: 'POST',
      headers: buildHeaders(userId, token),
      body: JSON.stringify(request),
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Failed to submit quiz (status ${res.status})`,
      };
    }

    return { data: body as QuizSubmitResponse, error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while submitting quiz.',
    };
  }
};

/**
 * Retrieves past attempts for a quiz.
 * Endpoint: GET /api/quizzes/{quizId}/attempts
 */
export const fetchQuizAttempts = async (
  quizId: string,
  userId: string,
  token?: string
): Promise<ApiResult<QuizAttemptHistoryItem[]>> => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/quizzes/${quizId}/attempts`, {
      method: 'GET',
      headers: buildHeaders(userId, token),
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Failed to fetch attempts (status ${res.status})`,
      };
    }

    return { data: body as QuizAttemptHistoryItem[], error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while fetching quiz attempts.',
    };
  }
};
