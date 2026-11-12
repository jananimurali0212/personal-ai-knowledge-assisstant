import React, { useState, useEffect, useCallback } from 'react';
import { 
  FileText, 
  Search, 
  Plus, 
  Filter, 
  ChevronUp
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { DocumentRecord, DocumentStats } from '../types';
import { 
  fetchUserDocuments, 
  deleteUserDocument, 
  calculateDocumentStats 
} from '../services/documents';
import { DocumentUploadZone } from '../components/DocumentUploadZone';
import { DocumentTable } from '../components/DocumentTable';

export const MyDocumentsPage: React.FC = () => {
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
  const [showUploadZone, setShowUploadZone] = useState<boolean>(true);

  // Filter & Search state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('all');

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

  // Filter and search computation
  const filteredDocuments = documents.filter((doc) => {
    const matchesSearch = doc.file_name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = typeFilter === 'all' || doc.file_type.toLowerCase() === typeFilter.toLowerCase();
    return matchesSearch && matchesType;
  });

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <FileText className="h-5 w-5 text-purple-400" />
            <span>My Knowledge Documents</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Manage your personal repository of uploaded PDF, DOCX, and TXT knowledge sources
          </p>
        </div>

        <button
          onClick={() => setShowUploadZone(!showUploadZone)}
          className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md shadow-purple-600/20 hover:opacity-95 transition-all self-start sm:self-auto"
        >
          {showUploadZone ? <ChevronUp className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
          <span>{showUploadZone ? 'Hide Uploader' : 'Upload New Document'}</span>
        </button>
      </div>

      {/* Upload Zone (Expandable) */}
      {showUploadZone && user && (
        <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-6 backdrop-blur-sm shadow-xl animate-fadeIn">
          <DocumentUploadZone
            userId={user.id}
            onUploadSuccess={handleUploadSuccess}
            compact={false}
          />
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 rounded-2xl border border-slate-800 bg-[#0F172A]/70 p-4">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
            <Search className="h-4 w-4 text-slate-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search document names..."
            className="w-full rounded-xl border border-slate-800 bg-slate-900/80 py-2 pl-10 pr-3.5 text-xs text-slate-100 placeholder-slate-400 outline-none transition-all focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
          />
        </div>

        {/* Format Filter Tabs */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400 flex items-center gap-1 mr-1">
            <Filter className="h-3 w-3" /> Filter:
          </span>
          <button
            onClick={() => setTypeFilter('all')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              typeFilter === 'all'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'bg-slate-800/80 text-slate-400 hover:text-white'
            }`}
          >
            All ({documents.length})
          </button>
          <button
            onClick={() => setTypeFilter('pdf')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              typeFilter === 'pdf'
                ? 'bg-rose-600 text-white shadow-sm'
                : 'bg-slate-800/80 text-slate-400 hover:text-white'
            }`}
          >
            PDF ({stats.pdfCount})
          </button>
          <button
            onClick={() => setTypeFilter('docx')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              typeFilter === 'docx'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-800/80 text-slate-400 hover:text-white'
            }`}
          >
            DOCX ({stats.docxCount})
          </button>
          <button
            onClick={() => setTypeFilter('txt')}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              typeFilter === 'txt'
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'bg-slate-800/80 text-slate-400 hover:text-white'
            }`}
          >
            TXT ({stats.txtCount})
          </button>
        </div>
      </div>

      {/* Document Table */}
      <DocumentTable
        documents={filteredDocuments}
        isLoading={isLoading}
        errorMessage={errorMessage}
        isSchemaMissing={isSchemaMissing}
        onDeleteDocument={handleDeleteDocument}
        onRefresh={loadDocuments}
        title={
          typeFilter === 'all'
            ? 'All Uploaded Documents'
            : `${typeFilter.toUpperCase()} Documents`
        }
      />
    </div>
  );
};
