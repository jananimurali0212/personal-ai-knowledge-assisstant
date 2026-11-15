import { supabase } from './supabase';
import { DocumentRecord, DocumentStats, DocumentSummaryResponse } from '../types';

export const SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.txt'];
export const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Normalizes and extracts file extension from filename.
 */
export const getFileExtension = (filename: string): string => {
  const parts = filename.split('.');
  if (parts.length <= 1) return '';
  return parts[parts.length - 1].toLowerCase();
};

/**
 * Checks if a file is supported (PDF, DOCX, TXT).
 */
export const isSupportedFile = (file: File): { valid: boolean; reason?: string } => {
  const ext = `.${getFileExtension(file.name)}`;
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      reason: `Unsupported file type "${ext}". Only PDF, DOCX, and TXT files are accepted.`,
    };
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      valid: false,
      reason: `File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of 25MB.`,
    };
  }

  if (file.size === 0) {
    return {
      valid: false,
      reason: 'The selected file is empty (0 bytes).',
    };
  }

  return { valid: true };
};

/**
 * Formats bytes into a human-readable string.
 */
export const formatBytes = (bytes: number, decimals: number = 1): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

/**
 * Calculates aggregate stats from loaded documents.
 */
export const calculateDocumentStats = (documents: DocumentRecord[]): DocumentStats => {
  let totalSizeBytes = 0;
  let pdfCount = 0;
  let docxCount = 0;
  let txtCount = 0;

  for (const doc of documents) {
    totalSizeBytes += Number(doc.file_size) || 0;
    const type = (doc.file_type || '').toLowerCase();
    if (type === 'pdf') pdfCount++;
    else if (type === 'docx') docxCount++;
    else if (type === 'txt') txtCount++;
  }

  return {
    totalDocuments: documents.length,
    totalSizeBytes,
    pdfCount,
    docxCount,
    txtCount,
  };
};

/**
 * Fetches all documents owned by the authenticated user from Supabase.
 */
export const fetchUserDocuments = async (
  userId: string
): Promise<{ data: DocumentRecord[]; error: string | null; isSchemaMissing?: boolean }> => {
  if (!userId) {
    return { data: [], error: 'User is not authenticated.' };
  }

  try {
    const { data, error } = await supabase
      .from('documents')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      const isMissing = error.code === '42P01' || error.message?.includes('does not exist');
      return {
        data: [],
        error: isMissing
          ? 'The "documents" table does not exist in your Supabase database yet. Please run the provided SQL schema script in your Supabase SQL Editor.'
          : error.message,
        isSchemaMissing: isMissing,
      };
    }

    return { data: (data as DocumentRecord[]) || [], error: null };
  } catch (err: any) {
    return { data: [], error: err.message || 'Failed to fetch user documents.' };
  }
};

/**
 * Uploads a document file to Supabase Storage and records metadata in the documents table.
 */
export const uploadUserDocument = async (
  file: File,
  userId: string
): Promise<{ data: DocumentRecord | null; error: string | null }> => {
  if (!userId) {
    return { data: null, error: 'User is not authenticated.' };
  }

  const validation = isSupportedFile(file);
  if (!validation.valid) {
    return { data: null, error: validation.reason || 'Invalid file.' };
  }

  const documentId = typeof crypto !== 'undefined' && crypto.randomUUID 
    ? crypto.randomUUID() 
    : `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

  const ext = getFileExtension(file.name);
  // Sanitize filename to avoid URL and storage key collisions
  const cleanName = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
  const storagePath = `${userId}/${documentId}/${cleanName}`;

  try {
    // 1. Upload to Supabase Storage in private 'documents' bucket
    const { error: storageError } = await supabase.storage
      .from('documents')
      .upload(storagePath, file, {
        cacheControl: '3600',
        upsert: false,
        contentType: file.type || undefined,
      });

    if (storageError) {
      if (storageError.message?.toLowerCase().includes('bucket not found')) {
        return {
          data: null,
          error: 'The "documents" storage bucket has not been created in Supabase yet. Please run the provided SQL script or create the "documents" bucket in your Supabase dashboard.',
        };
      }
      return { data: null, error: `Storage upload failed: ${storageError.message}` };
    }

    // 2. Insert metadata into 'documents' table
    const recordPayload = {
      id: documentId,
      user_id: userId,
      file_name: file.name,
      file_type: ext,
      file_size: file.size,
      storage_path: storagePath,
      status: 'uploaded',
    };

    const { data: insertedData, error: dbError } = await supabase
      .from('documents')
      .insert([recordPayload])
      .select()
      .single();

    if (dbError) {
      // Rollback uploaded storage object on DB insertion failure
      await supabase.storage.from('documents').remove([storagePath]);
      return {
        data: null,
        error: `Database insertion failed: ${dbError.message}`,
      };
    }

    return { data: insertedData as DocumentRecord, error: null };
  } catch (err: any) {
    return { data: null, error: err.message || 'An unexpected error occurred during document upload.' };
  }
};

/**
 * Deletes a document from Supabase Storage and removes its database metadata.
 */
export const deleteUserDocument = async (
  document: DocumentRecord,
  userId: string
): Promise<{ error: string | null }> => {
  if (!userId || document.user_id !== userId) {
    return { error: 'Unauthorized to delete this document.' };
  }

  try {
    // 1. Remove from Supabase Storage
    const { error: storageError } = await supabase.storage
      .from('documents')
      .remove([document.storage_path]);

    if (storageError) {
      console.warn('Storage deletion warning:', storageError.message);
    }

    // 2. Delete database row
    const { error: dbError } = await supabase
      .from('documents')
      .delete()
      .eq('id', document.id)
      .eq('user_id', userId);

    if (dbError) {
      return { error: `Failed to delete document record: ${dbError.message}` };
    }

    return { error: null };
  } catch (err: any) {
    return { error: err.message || 'An unexpected error occurred while deleting the document.' };
  }
};

/**
 * Generates a temporary secure signed download URL for an uploaded document.
 */
export const getDocumentDownloadUrl = async (
  storagePath: string,
  expiresInSeconds: number = 60
): Promise<{ url: string | null; error: string | null }> => {
  try {
    const { data, error } = await supabase.storage
      .from('documents')
      .createSignedUrl(storagePath, expiresInSeconds);

    if (error) {
      return { url: null, error: error.message };
    }

    return { url: data.signedUrl, error: null };
  } catch (err: any) {
    return { url: null, error: err.message || 'Failed to generate download URL.' };
  }
};

/**
 * Triggers AI document summarization via POST /api/documents/{documentId}/summarize.
 */
export const summarizeDocument = async (
  documentId: string,
  userId: string,
  forceRegenerate: boolean = false,
  token?: string
): Promise<{ data: DocumentSummaryResponse | null; error: string | null }> => {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-User-ID': userId,
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const res = await fetch(`${API_BASE_URL}/api/documents/${documentId}/summarize`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ force_regenerate: forceRegenerate }),
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Summarization failed with status ${res.status}`,
      };
    }

    return { data: body as DocumentSummaryResponse, error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while generating document summary.',
    };
  }
};

/**
 * Retrieves the saved summary for a document without regenerating via GET /api/documents/{documentId}/summary.
 */
export const fetchDocumentSummary = async (
  documentId: string,
  userId: string,
  token?: string
): Promise<{ data: DocumentSummaryResponse | null; error: string | null }> => {
  try {
    const headers: Record<string, string> = {
      'X-User-ID': userId,
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const res = await fetch(`${API_BASE_URL}/api/documents/${documentId}/summary`, {
      method: 'GET',
      headers,
    });

    const body = await res.json();
    if (!res.ok) {
      return {
        data: null,
        error: body?.detail || `Failed to fetch summary (status ${res.status})`,
      };
    }

    return { data: body as DocumentSummaryResponse, error: null };
  } catch (err: any) {
    return {
      data: null,
      error: err?.message || 'Network error occurred while fetching document summary.',
    };
  }
};

/**
 * Deletes the saved summary for a document via DELETE /api/documents/{documentId}/summary.
 */
export const deleteDocumentSummary = async (
  documentId: string,
  userId: string,
  token?: string
): Promise<{ success: boolean; error: string | null }> => {
  try {
    const headers: Record<string, string> = {
      'X-User-ID': userId,
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const res = await fetch(`${API_BASE_URL}/api/documents/${documentId}/summary`, {
      method: 'DELETE',
      headers,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return {
        success: false,
        error: body?.detail || `Failed to delete summary (status ${res.status})`,
      };
    }

    return { success: true, error: null };
  } catch (err: any) {
    return {
      success: false,
      error: err?.message || 'Network error occurred while deleting document summary.',
    };
  }
};

