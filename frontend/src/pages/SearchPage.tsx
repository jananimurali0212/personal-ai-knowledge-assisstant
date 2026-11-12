import React, { useState, useEffect, useCallback } from 'react';
import { Search, FileText } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { DocumentRecord } from '../types';
import { fetchUserDocuments, deleteUserDocument } from '../services/documents';
import { DocumentTable } from '../components/DocumentTable';
import { useNavigate } from 'react-router-dom';

export const SearchPage: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    if (!user) return;
    setIsLoading(true);
    const { data, error } = await fetchUserDocuments(user.id);
    if (error) {
      setErrorMessage(error);
    } else {
      setDocuments(data || []);
    }
    setIsLoading(false);
  }, [user]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleDelete = async (doc: DocumentRecord): Promise<boolean> => {
    if (!user) return false;
    const { error } = await deleteUserDocument(doc, user.id);
    if (error) {
      alert(`Deletion failed: ${error}`);
      return false;
    }
    setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
    return true;
  };

  const matchingDocuments = documents.filter((doc) => {
    const q = searchQuery.toLowerCase();
    return (
      doc.file_name.toLowerCase().includes(q) ||
      doc.file_type.toLowerCase().includes(q) ||
      doc.status.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Top Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Search className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white">Knowledge Search</h2>
            <p className="text-xs text-slate-400">
              Quickly locate uploaded PDF, DOCX, and TXT files across your knowledge library
            </p>
          </div>
        </div>
      </div>

      {/* Search Input Box */}
      <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/80 p-5 shadow-xl">
        <div className="relative">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4">
            <Search className="h-5 w-5 text-slate-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by file name, format (pdf, docx, txt), or status..."
            className="w-full rounded-xl border border-slate-700/80 bg-slate-900/90 py-3 pl-12 pr-4 text-sm text-slate-100 placeholder-slate-400 outline-none transition-all focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
            autoFocus
          />
        </div>
        {searchQuery && (
          <p className="mt-2 text-xs text-slate-400 pl-1">
            Found <span className="font-semibold text-purple-300">{matchingDocuments.length}</span> matching {matchingDocuments.length === 1 ? 'document' : 'documents'}.
          </p>
        )}
      </div>

      {/* Results Table */}
      <DocumentTable
        documents={matchingDocuments}
        isLoading={isLoading}
        errorMessage={errorMessage}
        onDeleteDocument={handleDelete}
        onRefresh={loadDocuments}
        title={searchQuery ? `Search Results for "${searchQuery}"` : 'All Indexed Documents'}
        emptyAction={
          <button
            onClick={() => navigate('/documents')}
            className="inline-flex items-center gap-1.5 rounded-xl bg-purple-600 px-4 py-2 text-xs font-semibold text-white hover:bg-purple-500"
          >
            <FileText className="h-4 w-4" />
            <span>Upload New Documents</span>
          </button>
        }
      />
    </div>
  );
};
