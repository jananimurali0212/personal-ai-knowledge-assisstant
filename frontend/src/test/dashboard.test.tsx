import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { DocumentUploadZone } from '../components/DocumentUploadZone';
import { DocumentTable } from '../components/DocumentTable';
import * as AuthContextModule from '../context/AuthContext';
import * as DocumentsModule from '../services/documents';
import { DocumentRecord } from '../types';

describe('Dashboard and Document UI Suite', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Dashboard Page Overview', () => {
    it('renders dashboard metric cards, quick upload, and recent documents list', async () => {
      vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
        user: { id: 'test-user-123', email: 'test@example.com' } as any,
        session: {} as any,
        loading: false,
        isConfigured: true,
        signIn: vi.fn(),
        signUp: vi.fn(),
        signOut: vi.fn(),
      });

      const mockDocs: DocumentRecord[] = [
        {
          id: 'doc-1',
          user_id: 'test-user-123',
          file_name: 'AI_Architecture.pdf',
          file_type: 'pdf',
          file_size: 1048576,
          storage_path: 'test-user-123/doc-1/AI_Architecture.pdf',
          status: 'ready',
          created_at: '2026-11-12T12:00:00Z',
        },
      ];

      vi.spyOn(DocumentsModule, 'fetchUserDocuments').mockResolvedValue({
        data: mockDocs,
        error: null,
      });

      render(
        <MemoryRouter>
          <DashboardPage onNavigate={vi.fn()} />
        </MemoryRouter>
      );

      // Verify Dashboard Header
      expect(screen.getByRole('heading', { name: /Knowledge Assistant Dashboard/i })).toBeInTheDocument();

      // Verify Metrics
      expect(screen.getByText(/Total Documents/i)).toBeInTheDocument();
      expect(screen.getByText(/Storage Used/i)).toBeInTheDocument();
      expect(screen.getByText(/Security Model/i)).toBeInTheDocument();

      // Wait for document to appear
      expect(await screen.findByText('AI_Architecture.pdf')).toBeInTheDocument();
      expect(screen.getAllByText('1 MB').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Ready').length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Document Upload Zone', () => {
    it('selects and previews supported file (PDF/DOCX/TXT)', async () => {
      const handleSuccess = vi.fn();

      render(
        <DocumentUploadZone
          userId="user-123"
          onUploadSuccess={handleSuccess}
          compact={false}
        />
      );

      expect(screen.getByText(/Upload knowledge document/i)).toBeInTheDocument();

      const file = new File(['dummy pdf content'], 'research.pdf', { type: 'application/pdf' });
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;

      fireEvent.change(input, { target: { files: [file] } });

      expect(await screen.findByText('research.pdf')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Upload to Cloud/i })).toBeInTheDocument();
    });

    it('rejects unsupported file formats with user-friendly error message', async () => {
      render(
        <DocumentUploadZone
          userId="user-123"
          onUploadSuccess={vi.fn()}
          compact={false}
        />
      );

      const invalidFile = new File(['alert(1)'], 'script.js', { type: 'application/javascript' });
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;

      fireEvent.change(input, { target: { files: [invalidFile] } });

      expect(
        await screen.findByText(/Unsupported file type "\.js"\. Only PDF, DOCX, and TXT files are accepted\./i)
      ).toBeInTheDocument();
    });

    it('executes upload on button click and invokes callback', async () => {
      const handleSuccess = vi.fn();
      const mockDoc: DocumentRecord = {
        id: 'new-doc-id',
        user_id: 'user-123',
        file_name: 'notes.txt',
        file_type: 'txt',
        file_size: 120,
        storage_path: 'user-123/new-doc-id/notes.txt',
        status: 'uploaded',
        created_at: new Date().toISOString(),
      };

      vi.spyOn(DocumentsModule, 'uploadUserDocument').mockResolvedValue({
        data: mockDoc,
        error: null,
      });

      render(
        <DocumentUploadZone
          userId="user-123"
          onUploadSuccess={handleSuccess}
          compact={false}
        />
      );

      const file = new File(['hello world'], 'notes.txt', { type: 'text/plain' });
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;

      fireEvent.change(input, { target: { files: [file] } });

      const uploadBtn = await screen.findByRole('button', { name: /Upload to Cloud/i });
      fireEvent.click(uploadBtn);

      await waitFor(() => {
        expect(handleSuccess).toHaveBeenCalledWith(mockDoc);
      });

      expect(await screen.findByText(/"notes\.txt" uploaded successfully!/i)).toBeInTheDocument();
    });
  });

  describe('Document Table Component', () => {
    it('displays file metadata: name, type, size, date, status, and actions', () => {
      const mockDocs: DocumentRecord[] = [
        {
          id: 'doc-xyz',
          user_id: 'user-123',
          file_name: 'Specification.docx',
          file_type: 'docx',
          file_size: 2097152,
          storage_path: 'user-123/doc-xyz/Specification.docx',
          status: 'uploaded',
          created_at: '2026-11-12T10:00:00Z',
        },
      ];

      render(
        <DocumentTable
          documents={mockDocs}
          isLoading={false}
          errorMessage={null}
          onDeleteDocument={vi.fn()}
        />
      );

      expect(screen.getByText('Specification.docx')).toBeInTheDocument();
      expect(screen.getByText('DOCX')).toBeInTheDocument();
      expect(screen.getByText('2 MB')).toBeInTheDocument();
      expect(screen.getByText('Stored')).toBeInTheDocument();
      expect(screen.getByLabelText('Download document')).toBeInTheDocument();
      expect(screen.getByLabelText('Delete document')).toBeInTheDocument();
    });

    it('triggers delete confirmation modal when trash icon is clicked', async () => {
      const mockDocs: DocumentRecord[] = [
        {
          id: 'doc-del',
          user_id: 'user-123',
          file_name: 'Draft.txt',
          file_type: 'txt',
          file_size: 500,
          storage_path: 'user-123/doc-del/Draft.txt',
          status: 'uploaded',
          created_at: '2026-11-12T10:00:00Z',
        },
      ];

      render(
        <DocumentTable
          documents={mockDocs}
          isLoading={false}
          errorMessage={null}
          onDeleteDocument={vi.fn()}
        />
      );

      const deleteBtn = screen.getByLabelText('Delete document');
      fireEvent.click(deleteBtn);

      expect(screen.getByRole('heading', { name: /Delete Document/i })).toBeInTheDocument();
      expect(screen.getByText(/Are you sure you want to permanently delete/i)).toBeInTheDocument();
    });
  });
});
