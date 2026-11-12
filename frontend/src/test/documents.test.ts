import { describe, it, expect, vi, beforeEach } from 'vitest';
import { 
  isSupportedFile, 
  formatBytes, 
  calculateDocumentStats, 
  fetchUserDocuments, 
  uploadUserDocument, 
  deleteUserDocument,
  MAX_FILE_SIZE_BYTES
} from '../services/documents';
import { supabase } from '../services/supabase';
import { DocumentRecord } from '../types';

describe('Document Management Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('File Validation & Formatting', () => {
    it('accepts PDF, DOCX, and TXT files', () => {
      const pdf = new File(['%PDF-1.4 test'], 'paper.pdf', { type: 'application/pdf' });
      const docx = new File(['PK docx test'], 'notes.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
      const txt = new File(['plain text'], 'readme.txt', { type: 'text/plain' });

      expect(isSupportedFile(pdf).valid).toBe(true);
      expect(isSupportedFile(docx).valid).toBe(true);
      expect(isSupportedFile(txt).valid).toBe(true);
    });

    it('rejects unsupported extensions', () => {
      const exe = new File(['bin'], 'malware.exe', { type: 'application/octet-stream' });
      const js = new File(['console.log(1)'], 'script.js', { type: 'application/javascript' });

      const exeRes = isSupportedFile(exe);
      expect(exeRes.valid).toBe(false);
      expect(exeRes.reason).toContain('Only PDF, DOCX, and TXT');

      const jsRes = isSupportedFile(js);
      expect(jsRes.valid).toBe(false);
    });

    it('rejects files larger than 25MB', () => {
      const largeFile = new File([''], 'huge.pdf', { type: 'application/pdf' });
      Object.defineProperty(largeFile, 'size', { value: MAX_FILE_SIZE_BYTES + 1024 });

      const res = isSupportedFile(largeFile);
      expect(res.valid).toBe(false);
      expect(res.reason).toContain('exceeds the maximum allowed limit of 25MB');
    });

    it('formats bytes correctly', () => {
      expect(formatBytes(0)).toBe('0 B');
      expect(formatBytes(1024)).toBe('1 KB');
      expect(formatBytes(1048576)).toBe('1 MB');
      expect(formatBytes(5242880)).toBe('5 MB');
    });

    it('calculates document stats correctly', () => {
      const mockDocs: DocumentRecord[] = [
        {
          id: '1',
          user_id: 'u-1',
          file_name: 'test.pdf',
          file_type: 'pdf',
          file_size: 1048576,
          storage_path: 'u-1/1/test.pdf',
          status: 'ready',
          created_at: new Date().toISOString(),
        },
        {
          id: '2',
          user_id: 'u-1',
          file_name: 'doc.docx',
          file_type: 'docx',
          file_size: 2097152,
          storage_path: 'u-1/2/doc.docx',
          status: 'uploaded',
          created_at: new Date().toISOString(),
        },
        {
          id: '3',
          user_id: 'u-1',
          file_name: 'notes.txt',
          file_type: 'txt',
          file_size: 512,
          storage_path: 'u-1/3/notes.txt',
          status: 'uploaded',
          created_at: new Date().toISOString(),
        },
      ];

      const stats = calculateDocumentStats(mockDocs);
      expect(stats.totalDocuments).toBe(3);
      expect(stats.pdfCount).toBe(1);
      expect(stats.docxCount).toBe(1);
      expect(stats.txtCount).toBe(1);
      expect(stats.totalSizeBytes).toBe(1048576 + 2097152 + 512);
    });
  });

  describe('Supabase Operations', () => {
    it('fetches documents for the authenticated user with RLS filtering', async () => {
      const mockDocs = [
        {
          id: 'doc-1',
          user_id: 'user-abc',
          file_name: 'manual.pdf',
          file_type: 'pdf',
          file_size: 1000,
          storage_path: 'user-abc/doc-1/manual.pdf',
          status: 'uploaded',
          created_at: '2026-11-12T10:00:00Z',
        },
      ];

      const mockOrder = vi.fn().mockResolvedValue({ data: mockDocs, error: null });
      const mockEq = vi.fn().mockReturnValue({ order: mockOrder });
      const mockSelect = vi.fn().mockReturnValue({ eq: mockEq });
      vi.spyOn(supabase, 'from').mockReturnValue({ select: mockSelect } as any);

      const result = await fetchUserDocuments('user-abc');

      expect(supabase.from).toHaveBeenCalledWith('documents');
      expect(mockSelect).toHaveBeenCalledWith('*');
      expect(mockEq).toHaveBeenCalledWith('user_id', 'user-abc');
      expect(result.data).toEqual(mockDocs);
      expect(result.error).toBeNull();
    });

    it('uploads file to storage and saves metadata to PostgreSQL', async () => {
      const file = new File(['sample content'], 'guide.pdf', { type: 'application/pdf' });
      const userId = 'user-xyz';

      // Mock storage upload
      const mockUpload = vi.fn().mockResolvedValue({ data: { path: 'path' }, error: null });
      vi.spyOn(supabase.storage, 'from').mockReturnValue({ upload: mockUpload } as any);

      // Mock DB insert
      const mockSingle = vi.fn().mockResolvedValue({
        data: {
          id: 'new-id',
          user_id: userId,
          file_name: 'guide.pdf',
          file_type: 'pdf',
          file_size: file.size,
          storage_path: `${userId}/new-id/guide.pdf`,
          status: 'uploaded',
        },
        error: null,
      });
      const mockSelect = vi.fn().mockReturnValue({ single: mockSingle });
      const mockInsert = vi.fn().mockReturnValue({ select: mockSelect });
      vi.spyOn(supabase, 'from').mockReturnValue({ insert: mockInsert } as any);

      const result = await uploadUserDocument(file, userId);

      expect(supabase.storage.from).toHaveBeenCalledWith('documents');
      expect(mockUpload).toHaveBeenCalled();
      expect(supabase.from).toHaveBeenCalledWith('documents');
      expect(result.data?.file_name).toBe('guide.pdf');
      expect(result.error).toBeNull();
    });

    it('deletes document from Supabase storage and documents table', async () => {
      const doc: DocumentRecord = {
        id: 'del-1',
        user_id: 'user-xyz',
        file_name: 'old.txt',
        file_type: 'txt',
        file_size: 100,
        storage_path: 'user-xyz/del-1/old.txt',
        status: 'uploaded',
        created_at: '2026-11-12T10:00:00Z',
      };

      const mockRemove = vi.fn().mockResolvedValue({ data: [], error: null });
      vi.spyOn(supabase.storage, 'from').mockReturnValue({ remove: mockRemove } as any);

      const mockEqUser = vi.fn().mockResolvedValue({ data: null, error: null });
      const mockEqId = vi.fn().mockReturnValue({ eq: mockEqUser });
      const mockDelete = vi.fn().mockReturnValue({ eq: mockEqId });
      vi.spyOn(supabase, 'from').mockReturnValue({ delete: mockDelete } as any);

      const res = await deleteUserDocument(doc, 'user-xyz');

      expect(supabase.storage.from).toHaveBeenCalledWith('documents');
      expect(mockRemove).toHaveBeenCalledWith(['user-xyz/del-1/old.txt']);
      expect(supabase.from).toHaveBeenCalledWith('documents');
      expect(res.error).toBeNull();
    });
  });
});

