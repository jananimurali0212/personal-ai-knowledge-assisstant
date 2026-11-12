import React, { useState, useEffect, useCallback } from 'react';
import { 
  FileText, 
  HardDrive, 
  UploadCloud, 
  Sparkles, 
  ArrowRight, 
  ShieldCheck, 
  Bot
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { NavPage, DocumentRecord, DocumentStats } from '../types';
import { 
  fetchUserDocuments, 
  deleteUserDocument, 
  calculateDocumentStats, 
  formatBytes 
} from '../services/documents';
import { DocumentUploadZone } from '../components/DocumentUploadZone';
import { DocumentTable } from '../components/DocumentTable';

interface DashboardPageProps {
  onNavigate: (page: NavPage) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigate }) => {
  const { user } = useAuth();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [stats, setStats] = useState<DocumentStats>({
    totalDocuments: 0,
    totalSizeBytes: 0,
    pdfCount: 0,
    docxCount: 0,
    txtCount: 0,
  });
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSchemaMissing, setIsSchemaMissing] = useState<boolean>(false);

  const loadDocuments = useCallback(async () => {
    if (!user) return;
    setIsLoading(true);
    setErrorMessage(null);

    const { data, error, isSchemaMissing: missing } = await fetchUserDocuments(user.id);

    if (error) {
      setErrorMessage(error);
      setIsSchemaMissing(Boolean(missing));
    } else {
      setDocuments(data);
      setStats(calculateDocumentStats(data));
      setIsSchemaMissing(false);
    }
    setIsLoading(false);
  }, [user]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleUploadSuccess = (newDoc: DocumentRecord) => {
    setDocuments((prev) => {
      const updated = [newDoc, ...prev];
      setStats(calculateDocumentStats(updated));
      return updated;
    });
  };

  const handleDeleteDocument = async (doc: DocumentRecord): Promise<boolean> => {
    if (!user) return false;
    const { error } = await deleteUserDocument(doc, user.id);
    if (error) {
      alert(`Deletion failed: ${error}`);
      return false;
    }
    setDocuments((prev) => {
      const updated = prev.filter((d) => d.id !== doc.id);
      setStats(calculateDocumentStats(updated));
      return updated;
    });
    return true;
  };

  return (
    <div className="space-y-8 max-w-7xl">
      {/* Welcome & AI Knowledge Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-purple-500/20 bg-gradient-to-r from-purple-950/40 via-indigo-950/40 to-slate-900/60 p-6 sm:p-8 backdrop-blur-md shadow-2xl">
        <div className="absolute -right-12 -top-12 h-64 w-64 rounded-full bg-purple-600/10 blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-purple-500/30 bg-purple-500/10 px-3 py-1 text-xs font-semibold text-purple-300">
              <Sparkles className="h-3.5 w-3.5 text-purple-400" />
              <span>Personal AI Knowledge Hub</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
              Knowledge Assistant Dashboard
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed">
              Upload your research papers, notes, and manuals in PDF, DOCX, or TXT format. 
              Files are privately secured in Supabase Storage and ready to power your AI assistant.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => onNavigate('documents')}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-purple-600/25 transition-all hover:scale-[1.02] active:scale-95"
            >
              <FileText className="h-4 w-4" />
              <span>My Documents</span>
              <ArrowRight className="h-4 w-4 ml-1" />
            </button>
            <button
              onClick={() => onNavigate('assistant')}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2.5 text-xs font-medium text-slate-300 transition-all hover:bg-slate-700 hover:text-white"
            >
              <Bot className="h-4 w-4 text-purple-400" />
              <span>AI Assistant</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Total Documents */}
        <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-6 backdrop-blur-sm transition-all hover:border-purple-500/30">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Documents</span>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
              <FileText className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white tracking-tight">{stats.totalDocuments}</span>
            <span className="text-xs text-slate-400">files</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            {stats.pdfCount} PDF · {stats.docxCount} DOCX · {stats.txtCount} TXT
          </p>
        </div>

        {/* Storage Usage */}
        <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-6 backdrop-blur-sm transition-all hover:border-purple-500/30">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Storage Used</span>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <HardDrive className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white tracking-tight">
              {formatBytes(stats.totalSizeBytes)}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Private bucket storage allocated
          </p>
        </div>

        {/* Privacy & RLS */}
        <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-6 backdrop-blur-sm transition-all hover:border-purple-500/30">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Security Model</span>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <ShieldCheck className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-xl font-bold text-white tracking-tight">RLS Active</span>
          </div>
          <p className="mt-1 text-[11px] text-emerald-400/90 flex items-center gap-1">
            <span>Isolated per authenticated UID</span>
          </p>
        </div>

        {/* Knowledge Status */}
        <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-6 backdrop-blur-sm transition-all hover:border-purple-500/30">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Assistant State</span>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Bot className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-4 flex items-baseline gap-2">
            <span className="text-xl font-bold text-white tracking-tight">
              {stats.totalDocuments > 0 ? 'Ready' : 'Awaiting Docs'}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            {stats.totalDocuments > 0 ? `${stats.totalDocuments} sources connected` : 'Upload docs to begin'}
          </p>
        </div>
      </div>

      {/* Quick Upload Section */}
      <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-6 backdrop-blur-sm shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <UploadCloud className="h-4 w-4 text-purple-400" />
              <span>Quick Document Upload</span>
            </h3>
            <p className="text-xs text-slate-400">Upload PDF, DOCX, or TXT directly to your private knowledge store</p>
          </div>
        </div>

        {user && (
          <DocumentUploadZone
            userId={user.id}
            onUploadSuccess={handleUploadSuccess}
            compact={true}
          />
        )}
      </div>

      {/* Recently Uploaded Documents */}
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <h3 className="text-sm font-semibold text-white">Recently Uploaded Documents</h3>
          {documents.length > 5 && (
            <button
              onClick={() => onNavigate('documents')}
              className="text-xs font-medium text-purple-400 hover:text-purple-300 flex items-center gap-1"
            >
              <span>View all ({documents.length})</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        <DocumentTable
          documents={documents}
          isLoading={isLoading}
          errorMessage={errorMessage}
          isSchemaMissing={isSchemaMissing}
          onDeleteDocument={handleDeleteDocument}
          onRefresh={loadDocuments}
          limit={5}
          title="Recent Documents"
          emptyAction={
            <button
              onClick={() => onNavigate('documents')}
              className="inline-flex items-center gap-2 rounded-xl bg-purple-600 px-4 py-2 text-xs font-semibold text-white hover:bg-purple-500 shadow-md shadow-purple-600/20"
            >
              <UploadCloud className="h-4 w-4" />
              <span>Go to My Documents</span>
            </button>
          }
        />
      </div>
    </div>
  );
};
