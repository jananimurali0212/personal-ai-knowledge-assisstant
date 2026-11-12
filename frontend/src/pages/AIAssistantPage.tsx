import React, { useState, useEffect } from 'react';
import { 
  Bot, 
  Sparkles, 
  Send, 
  FileText, 
  Layers, 
  ArrowRight
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { DocumentRecord } from '../types';
import { fetchUserDocuments } from '../services/documents';
import { useNavigate } from 'react-router-dom';

export const AIAssistantPage: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [promptInput, setPromptInput] = useState('');

  useEffect(() => {
    if (!user) return;
    fetchUserDocuments(user.id).then(({ data }) => {
      setDocuments(data || []);
    });
  }, [user]);

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Top Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white">AI Knowledge Assistant</h2>
            <p className="text-xs text-slate-400">
              Interactive conversational intelligence grounded in your private documents
            </p>
          </div>
        </div>
      </div>

      {/* Assistant Status Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/80 p-6 backdrop-blur-sm shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-600 text-white shadow-md shadow-purple-600/25">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Knowledge Grounding Status</h3>
              <p className="text-xs text-slate-400">
                {documents.length > 0
                  ? `${documents.length} verified knowledge sources currently indexed in your account`
                  : 'No documents uploaded yet in your knowledge base'}
              </p>
            </div>
          </div>

          <button
            onClick={() => navigate('/documents')}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800/80 px-3.5 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white transition-colors self-start sm:self-auto"
          >
            <FileText className="h-3.5 w-3.5 text-purple-400" />
            <span>Manage Documents</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Attached Knowledge Sources chips */}
        <div className="mt-5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
            Active Sources for RAG Querying
          </span>
          {documents.length > 0 ? (
            <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900/90 px-2.5 py-1 text-xs text-slate-300"
                >
                  <FileText className="h-3.5 w-3.5 text-purple-400 shrink-0" />
                  <span className="truncate max-w-xs">{doc.file_name}</span>
                  <span className="text-[10px] text-slate-400 uppercase font-mono">({doc.file_type})</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-800 p-4 text-center text-xs text-slate-400">
              Upload documents under <span className="text-purple-400 font-semibold cursor-pointer" onClick={() => navigate('/documents')}>My Documents</span> to provide context to this assistant.
            </div>
          )}
        </div>
      </div>

      {/* Upcoming RAG Pipeline Roadmap Info Card */}
      <div className="rounded-2xl border border-purple-500/20 bg-gradient-to-r from-purple-950/20 via-indigo-950/20 to-slate-900/40 p-6 backdrop-blur-sm">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
          <Layers className="h-4 w-4 text-purple-400" />
          <span>Knowledge Processing Architecture</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-300">
          <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-3.5">
            <span className="block font-semibold text-purple-300 mb-1">Step 1: Secure Storage</span>
            <p className="text-slate-400 text-[11px]">
              Active: Files stored privately in Supabase Storage with Row Level Security.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-3.5">
            <span className="block font-semibold text-indigo-300 mb-1">Step 2: Text & Embeddings</span>
            <p className="text-slate-400 text-[11px]">
              Upcoming: Parsing PDF, DOCX, and TXT with chunking and pgvector embeddings.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-3.5">
            <span className="block font-semibold text-blue-300 mb-1">Step 3: Reasoning & Chat</span>
            <p className="text-slate-400 text-[11px]">
              Upcoming: Gemini LLM generation with exact line and page source citations.
            </p>
          </div>
        </div>
      </div>

      {/* Mock Chat Input Box */}
      <div className="rounded-2xl border border-slate-800 bg-[#0F172A]/80 p-4 shadow-xl">
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={promptInput}
            onChange={(e) => setPromptInput(e.target.value)}
            placeholder={
              documents.length > 0
                ? "Ask a question about your uploaded documents (chat engine arriving in next milestone)..."
                : "Upload documents to begin querying your knowledge base..."
            }
            className="flex-1 rounded-xl border border-slate-800 bg-slate-900/90 py-2.5 px-4 text-xs text-slate-100 placeholder-slate-400 outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
          />
          <button
            type="button"
            onClick={() => {
              if (promptInput.trim()) {
                alert('Document text extraction and embeddings pipeline will be enabled in the upcoming milestone.');
                setPromptInput('');
              }
            }}
            className="inline-flex items-center gap-2 rounded-xl bg-purple-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-purple-500 shadow-md shadow-purple-600/20 transition-all active:scale-95"
          >
            <Send className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Send Query</span>
          </button>
        </div>
      </div>
    </div>
  );
};
