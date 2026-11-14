import React, { useState } from 'react';
import { 
  FileText, 
  Trash2, 
  Download, 
  CheckCircle, 
  Clock, 
  AlertCircle,
  Loader2,
  FileSearch,
  Database,
  Sparkles,
  BookOpen,
  HelpCircle,
} from 'lucide-react';
import { DocumentRecord, DocumentSummaryResponse } from '../types';
import { formatBytes, getDocumentDownloadUrl, summarizeDocument } from '../services/documents';
import { DeleteConfirmModal } from './DeleteConfirmModal';
import { DocumentSummaryModal } from './DocumentSummaryModal';
import { QuizModal } from './QuizModal';

interface DocumentTableProps {
  documents: DocumentRecord[];
  isLoading: boolean;
  errorMessage: string | null;
  isSchemaMissing?: boolean;
  onDeleteDocument: (doc: DocumentRecord) => Promise<boolean>;
  onRefresh?: () => void;
  title?: string;
  limit?: number;
  emptyAction?: React.ReactNode;
}

export const DocumentTable: React.FC<DocumentTableProps> = ({
  documents,
  isLoading,
  errorMessage,
  isSchemaMissing = false,
  onDeleteDocument,
  onRefresh,
  title = 'Knowledge Base Documents',
  limit,
  emptyAction,
}) => {
  const [docToDelete, setDocToDelete] = useState<DocumentRecord | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  // Document Summarization state
  const [selectedDocForSummary, setSelectedDocForSummary] = useState<DocumentRecord | null>(null);
  const [summaryResponse, setSummaryResponse] = useState<DocumentSummaryResponse | null>(null);
  const [isSummarizing, setIsSummarizing] = useState<boolean>(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  // Document Quiz state
  const [selectedDocForQuiz, setSelectedDocForQuiz] = useState<DocumentRecord | null>(null);

  const handleOpenSummary = async (doc: DocumentRecord) => {
    setSelectedDocForSummary(doc);
    setSummaryError(null);

    // If document already contains a loaded summary, show it immediately
    if (doc.summary) {
      setSummaryResponse({
        document_id: doc.id,
        user_id: doc.user_id,
        file_name: doc.file_name,
        file_type: doc.file_type,
        status: 'completed',
        summary: doc.summary,
        cached: true,
      });
      return;
    }

    // Otherwise trigger or fetch summary from backend
    setIsSummarizing(true);
    setSummaryResponse(null);

    const { data, error } = await summarizeDocument(doc.id, doc.user_id, false);
    setIsSummarizing(false);

    if (error || !data) {
      setSummaryError(error || 'Failed to generate document summary.');
    } else {
      setSummaryResponse(data);
      // Update local doc record summary if possible
      doc.summary = data.summary;
      doc.summary_status = 'completed';
    }
  };

  const handleRegenerateSummary = async () => {
    if (!selectedDocForSummary) return;
    setIsSummarizing(true);
    setSummaryError(null);

    const { data, error } = await summarizeDocument(
      selectedDocForSummary.id,
      selectedDocForSummary.user_id,
      true
    );
    setIsSummarizing(false);

    if (error || !data) {
      setSummaryError(error || 'Failed to regenerate summary.');
    } else {
      setSummaryResponse(data);
      selectedDocForSummary.summary = data.summary;
      selectedDocForSummary.summary_status = 'completed';
    }
  };

  const displayedDocs = limit ? documents.slice(0, limit) : documents;

  const handleDeleteConfirm = async () => {
    if (!docToDelete) return;
    setIsDeleting(true);
    const success = await onDeleteDocument(docToDelete);
    setIsDeleting(false);
    if (success) {
      setDocToDelete(null);
    }
  };

  const handleDownload = async (doc: DocumentRecord) => {
    setDownloadingId(doc.id);
    try {
      const { url, error } = await getDocumentDownloadUrl(doc.storage_path);
      if (error || !url) {
        alert(`Failed to generate secure download link: ${error || 'Unknown error'}`);
      } else {
        const link = document.createElement('a');
        link.href = url;
        link.download = doc.file_name;
        link.target = '_blank';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } finally {
      setDownloadingId(null);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const d = new Date(dateString);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  const getFormatBadge = (type: string) => {
    const ext = type.toLowerCase();
    if (ext === 'pdf') {
      return (
        <span className="inline-flex items-center rounded-md border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[10px] font-bold text-rose-300">
          PDF
        </span>
      );
    }
    if (ext === 'docx') {
      return (
        <span className="inline-flex items-center rounded-md border border-blue-500/30 bg-blue-500/10 px-2 py-0.5 text-[10px] font-bold text-blue-300">
          DOCX
        </span>
      );
    }
    return (
      <span className="inline-flex items-center rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-300">
        TXT
      </span>
    );
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ready':
        return (
          <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
            <CheckCircle className="h-3 w-3" /> Ready
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-300">
            <Clock className="h-3 w-3 animate-spin" /> Processing
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 rounded-full border border-purple-500/30 bg-purple-500/10 px-2 py-0.5 text-[10px] font-medium text-purple-300">
            <CheckCircle className="h-3 w-3" /> Stored
          </span>
        );
    }
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/80 backdrop-blur-sm shadow-xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 px-6 py-4">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <FileText className="h-4 w-4 text-purple-400" />
            <span>{title}</span>
            <span className="rounded-full bg-purple-500/10 px-2 py-0.5 text-[10px] font-semibold text-purple-300 border border-purple-500/20">
              {documents.length}
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time metadata saved in PostgreSQL with Row Level Security
          </p>
        </div>

        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="self-start sm:self-auto rounded-xl border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
          >
            {isLoading ? 'Refreshing...' : 'Refresh List'}
          </button>
        )}
      </div>

      {/* Schema Missing Notice */}
      {isSchemaMissing && (
        <div className="m-5 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-200">
          <div className="flex items-start gap-3">
            <Database className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-300">Supabase Table Setup Required</p>
              <p className="mt-1 text-amber-200/90 leading-relaxed">
                The <code className="bg-amber-950/60 px-1.5 py-0.5 rounded font-mono text-amber-100">documents</code> table or <code className="bg-amber-950/60 px-1.5 py-0.5 rounded font-mono text-amber-100">documents</code> storage bucket is not created yet in your Supabase project.
                Run the script in <span className="font-semibold text-white">supabase_schema.sql</span> (located at your project root) in your Supabase SQL Editor.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {errorMessage && !isSchemaMissing && (
        <div className="m-5 flex items-center gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Loading state */}
      {isLoading && documents.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-slate-400">
          <Loader2 className="h-7 w-7 animate-spin text-purple-400" />
          <p className="mt-3 text-xs font-medium text-slate-300">Loading your knowledge documents...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && documents.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <FileSearch className="h-7 w-7" />
          </div>
          <h4 className="mt-4 text-sm font-semibold text-white">No documents uploaded yet</h4>
          <p className="mt-1 text-xs text-slate-400 max-w-sm">
            Upload your first PDF, DOCX, or TXT file to start training your Personal AI Knowledge Assistant.
          </p>
          {emptyAction && <div className="mt-4">{emptyAction}</div>}
        </div>
      )}

      {/* Document List View */}
      {displayedDocs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800/80 bg-slate-900/50 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              <tr>
                <th className="py-3.5 pl-6 pr-4">Document</th>
                <th className="py-3.5 px-4">Type</th>
                <th className="py-3.5 px-4">Size</th>
                <th className="py-3.5 px-4 hidden sm:table-cell">Uploaded At</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 pl-4 pr-6 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-medium">
              {displayedDocs.map((doc) => (
                <tr
                  key={doc.id}
                  className="transition-colors hover:bg-slate-800/40 group"
                >
                  {/* File Name */}
                  <td className="py-3.5 pl-6 pr-4">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-800 border border-slate-700/80 text-purple-400 group-hover:border-purple-500/40 transition-colors">
                        <FileText className="h-4 w-4" />
                      </div>
                      <div className="min-w-0">
                        <span className="block truncate font-semibold text-slate-100 max-w-xs sm:max-w-md">
                          {doc.file_name}
                        </span>
                        <span className="block text-[10px] text-slate-400 font-mono truncate max-w-xs">
                          {doc.id}
                        </span>
                      </div>
                    </div>
                  </td>

                  {/* Type */}
                  <td className="py-3.5 px-4 whitespace-nowrap">
                    {getFormatBadge(doc.file_type)}
                  </td>

                  {/* Size */}
                  <td className="py-3.5 px-4 text-slate-300 font-mono whitespace-nowrap">
                    {formatBytes(doc.file_size)}
                  </td>

                  {/* Upload Date */}
                  <td className="py-3.5 px-4 text-slate-400 hidden sm:table-cell whitespace-nowrap">
                    {formatDate(doc.created_at)}
                  </td>

                  {/* Processing Status */}
                  <td className="py-3.5 px-4 whitespace-nowrap">
                    {getStatusBadge(doc.status)}
                  </td>

                  {/* Actions */}
                  <td className="py-3.5 pl-4 pr-6 text-right whitespace-nowrap">
                    <div className="inline-flex items-center gap-2">
                      {/* Summarize Document Button */}
                      <button
                        type="button"
                        onClick={() => handleOpenSummary(doc)}
                        className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all ${
                          doc.summary || doc.summary_status === 'completed'
                            ? 'border border-purple-500/40 bg-purple-500/15 text-purple-200 hover:bg-purple-500/25 shadow-sm shadow-purple-500/10'
                            : 'border border-slate-700/80 bg-slate-800/80 text-slate-300 hover:border-purple-500/40 hover:bg-slate-700 hover:text-white'
                        }`}
                        title={doc.summary || doc.summary_status === 'completed' ? "View AI Summary" : "Generate AI Summary"}
                      >
                        {doc.summary || doc.summary_status === 'completed' ? (
                          <>
                            <BookOpen className="h-3.5 w-3.5 text-purple-400" />
                            <span>Summary</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-3.5 w-3.5 text-purple-400" />
                            <span>Summarize</span>
                          </>
                        )}
                      </button>

                      {/* Quiz Button */}
                      <button
                        type="button"
                        onClick={() => setSelectedDocForQuiz(doc)}
                        className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all border border-slate-700/80 bg-slate-800/80 text-slate-300 hover:border-indigo-500/40 hover:bg-slate-700 hover:text-indigo-200"
                        title="Generate Grounded AI Quiz"
                      >
                        <HelpCircle className="h-3.5 w-3.5 text-indigo-400" />
                        <span>Quiz</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleDownload(doc)}
                        disabled={downloadingId === doc.id}
                        className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-purple-300 transition-colors"
                        title="Download file via secure signed URL"
                        aria-label="Download document"
                      >
                        {downloadingId === doc.id ? (
                          <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                        ) : (
                          <Download className="h-4 w-4" />
                        )}
                      </button>

                      <button
                        type="button"
                        onClick={() => setDocToDelete(doc)}
                        className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-rose-400 transition-colors"
                        title="Delete document"
                        aria-label="Delete document"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        document={docToDelete}
        isOpen={Boolean(docToDelete)}
        isLoading={isDeleting}
        onClose={() => setDocToDelete(null)}
        onConfirm={handleDeleteConfirm}
      />

      {/* Document AI Summary Modal */}
      <DocumentSummaryModal
        isOpen={Boolean(selectedDocForSummary)}
        onClose={() => setSelectedDocForSummary(null)}
        document={selectedDocForSummary}
        summaryResponse={summaryResponse}
        isLoading={isSummarizing}
        errorMessage={summaryError}
        onRegenerate={handleRegenerateSummary}
      />

      {/* Document AI Quiz Modal */}
      <QuizModal
        isOpen={Boolean(selectedDocForQuiz)}
        onClose={() => setSelectedDocForQuiz(null)}
        document={selectedDocForQuiz}
        userId={selectedDocForQuiz?.user_id || ''}
      />
    </div>
  );
};
