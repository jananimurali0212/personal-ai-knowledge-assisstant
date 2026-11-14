import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, 
  Sparkles, 
  Send, 
  FileText, 
  ExternalLink,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Trash2,
  Quote,
  Layers,
  ChevronDown,
  ChevronUp,
  FileCheck,
  Shield,
  ArrowRight,
  UploadCloud,
  MessageSquare,
  Plus,
  Clock,
  Edit2,
  Check,
  X,
  Menu,
  Square
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { 
  DocumentRecord, 
  ChatMessage, 
  ChatSourceItem, 
  ConversationRecord 
} from '../types';
import { fetchUserDocuments, getDocumentDownloadUrl } from '../services/documents';
import { askRAGQuestion, streamRAGQuestion } from '../services/api';
import { 
  fetchUserConversations, 
  fetchConversationDetails, 
  deleteUserConversation,
  updateUserConversationTitle 
} from '../services/conversations';
import { MarkdownRenderer } from '../components/MarkdownRenderer';
import { useNavigate, useSearchParams } from 'react-router-dom';

export const AIAssistantPage: React.FC = () => {
  const { user, session } = useAuth();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const urlConvId = searchParams.get('c');

  // Documents & Knowledge Source state
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // Conversations & History state
  const [conversations, setConversations] = useState<ConversationRecord[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [activeConversationTitle, setActiveConversationTitle] = useState<string>('New Conversation');
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [isLoadingConversationDetail, setIsLoadingConversationDetail] = useState(false);
  const [showHistoryDrawer, setShowHistoryDrawer] = useState(false);

  // Renaming state
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState('');

  // Delete modal state
  const [deletingConvId, setDeletingConvId] = useState<string | null>(null);

  // Chat message & interaction state
  const [promptInput, setPromptInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [openingDocId, setOpeningDocId] = useState<string | null>(null);
  const [expandedSnippets, setExpandedSnippets] = useState<Record<string, boolean>>({});
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Real-time SSE Streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [streamingSources, setStreamingSources] = useState<ChatSourceItem[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Abort ongoing stream on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Fetch user uploaded documents
  useEffect(() => {
    if (!user) return;
    setIsLoadingDocs(true);
    fetchUserDocuments(user.id)
      .then(({ data }) => {
        setDocuments(data || []);
      })
      .finally(() => {
        setIsLoadingDocs(false);
      });
  }, [user]);

  // Load user conversation history list
  const reloadConversations = async () => {
    if (!user) return;
    setIsLoadingConversations(true);
    try {
      const list = await fetchUserConversations(user.id);
      setConversations(list);
    } finally {
      setIsLoadingConversations(false);
    }
  };

  useEffect(() => {
    reloadConversations();
  }, [user]);

  // Load conversation from URL search param (?c=<conversation_id>) if provided
  useEffect(() => {
    if (!user || !urlConvId) return;
    if (urlConvId === activeConversationId) return;
    setIsLoadingConversationDetail(true);
    fetchConversationDetails(urlConvId, user.id)
      .then((detail) => {
        if (detail && detail.conversation) {
          setActiveConversationId(detail.conversation.id);
          setActiveConversationTitle(detail.conversation.title);
          const loaded: ChatMessage[] = (detail.messages || []).map((m) => {
            const isAssistant = m.role === 'assistant';
            const isInsufficient = 
              m.content.includes('does not contain sufficient information') || 
              m.content.includes('insufficient information');
            return {
              id: m.id,
              sender: m.role as 'user' | 'assistant',
              text: m.content,
              timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              sources: m.sources || [],
              grounded: isAssistant && !isInsufficient && (m.sources && m.sources.length > 0),
              is_insufficient_context: isAssistant && isInsufficient,
            };
          });
          setMessages(loaded);
        }
      })
      .finally(() => {
        setIsLoadingConversationDetail(false);
      });
  }, [user, urlConvId]);

  // Auto-scroll when new messages arrive or when streaming response tokens
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSubmitting, streamingText]);

  // Toggle selection of specific knowledge documents
  const toggleDocumentFilter = (docId: string) => {
    setSelectedDocIds((prev) => 
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const toggleSnippetExpand = (key: string) => {
    setExpandedSnippets((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Start a fresh new conversation
  const handleStartNewConversation = () => {
    setActiveConversationId(null);
    setActiveConversationTitle('New Conversation');
    setMessages([]);
    setSelectedDocIds([]);
    setPromptInput('');
    setIsEditingTitle(false);
    setShowHistoryDrawer(false);
    setSearchParams({});
  };

  // Reopen an existing conversation from history
  const handleReopenConversation = async (conv: ConversationRecord) => {
    if (!user) return;
    setActiveConversationId(conv.id);
    setActiveConversationTitle(conv.title);
    setIsLoadingConversationDetail(true);
    setShowHistoryDrawer(false);
    setSearchParams({ c: conv.id });

    try {
      const detail = await fetchConversationDetails(conv.id, user.id);
      if (detail && detail.messages) {
        const loaded: ChatMessage[] = detail.messages.map((m) => {
          const isAssistant = m.role === 'assistant';
          const isInsufficient = 
            m.content.includes('does not contain sufficient information') || 
            m.content.includes('insufficient information');
          return {
            id: m.id,
            sender: m.role as 'user' | 'assistant',
            text: m.content,
            timestamp: new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            sources: m.sources || [],
            grounded: isAssistant && !isInsufficient && (m.sources && m.sources.length > 0),
            is_insufficient_context: isAssistant && isInsufficient,
          };
        });
        setMessages(loaded);
      } else {
        setMessages([]);
      }
    } finally {
      setIsLoadingConversationDetail(false);
    }
  };

  // Delete a conversation
  const handleDeleteConversation = async (convId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!user) return;

    const confirmed = window.confirm("Are you sure you want to delete this conversation? This will remove all messages permanently.");
    if (!confirmed) return;

    setDeletingConvId(convId);
    try {
      const success = await deleteUserConversation(convId, user.id);
      if (success) {
        setConversations((prev) => prev.filter((c) => c.id !== convId));
        if (activeConversationId === convId) {
          handleStartNewConversation();
        }
      } else {
        alert("Could not delete conversation. Please try again.");
      }
    } finally {
      setDeletingConvId(null);
    }
  };

  // Save renamed title
  const handleSaveRenamedTitle = async () => {
    if (!activeConversationId || !user || !editedTitle.trim()) {
      setIsEditingTitle(false);
      return;
    }
    const clean = editedTitle.trim();
    try {
      const updated = await updateUserConversationTitle(activeConversationId, clean, user.id);
      if (updated) {
        setActiveConversationTitle(updated.title);
        setConversations((prev) =>
          prev.map((c) => (c.id === updated.id ? { ...c, title: updated.title } : c))
        );
      }
    } finally {
      setIsEditingTitle(false);
    }
  };

  // Citation document viewer: opens signed URL and navigates directly to #page=N for PDFs
  const handleOpenCitationDocument = async (citation: ChatSourceItem) => {
    const key = `${citation.document_id}-${citation.chunk_index}`;
    setOpeningDocId(key);

    try {
      const targetDoc = documents.find(
        (d) => d.id === citation.document_id || d.file_name === citation.document_name
      );

      if (targetDoc && targetDoc.storage_path) {
        const { url, error } = await getDocumentDownloadUrl(targetDoc.storage_path, 120);
        if (error || !url) {
          alert(`Could not generate secure view URL: ${error || 'Unknown error'}`);
          return;
        }

        let finalUrl = url;
        const isPdf = 
          targetDoc.file_type?.toLowerCase() === 'pdf' || 
          citation.document_name.toLowerCase().endsWith('.pdf');

        if (isPdf && citation.page_number) {
          finalUrl = `${url}#page=${citation.page_number}`;
        }

        window.open(finalUrl, '_blank', 'noopener,noreferrer');
      } else {
        alert(
          `Document: ${citation.document_name}\n` +
          `${citation.has_page_number && citation.page_number ? `Page: ${citation.page_number}\n` : ''}` +
          `Excerpt:\n"${citation.snippet}"`
        );
      }
    } finally {
      setOpeningDocId(null);
    }
  };

  // Helper to render the citations tray for completed or streaming messages
  const renderCitationsTray = (sources: ChatSourceItem[]) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="mt-5 pt-4 border-t border-indigo-100/90 space-y-3.5 bg-indigo-50/30 -mx-5 -mb-5 md:-mx-6 md:-mb-6 p-4 md:p-5 rounded-b-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs">
          <div className="flex items-center gap-2 font-bold text-slate-800">
            <BookOpen className="h-4 w-4 text-indigo-600" />
            <span>Supporting Citations ({sources.length})</span>
          </div>
          <span className="text-[11px] text-slate-500">
            Click any citation to open the source document
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {sources.map((citation, cIdx) => {
            const snippetKey = `${citation.document_id}-${citation.chunk_index}-${cIdx}`;
            const isExpanded = !!expandedSnippets[snippetKey];
            const isPdf = 
              citation.file_type === 'pdf' || 
              citation.document_name.toLowerCase().endsWith('.pdf');
            const isOpening = 
              openingDocId === `${citation.document_id}-${citation.chunk_index}`;

            return (
              <div
                key={cIdx}
                className="group relative rounded-2xl border border-indigo-100 bg-white p-3.5 hover:border-indigo-300 hover:shadow-md transition-all shadow-xs space-y-2 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                      <span className="rounded-md bg-indigo-100 text-indigo-700 font-bold px-2 py-0.5 text-[10px]">
                        {citation.source_id || `Source ${cIdx + 1}`}
                      </span>
                      <span className={`rounded-md px-1.5 py-0.5 text-[9px] font-bold uppercase ${
                        isPdf 
                          ? 'bg-rose-50 text-rose-600 border border-rose-200' 
                          : 'bg-blue-50 text-blue-600 border border-blue-200'
                      }`}>
                        {citation.file_type || (isPdf ? 'PDF' : 'DOC')}
                      </span>
                    </div>

                    {isPdf && citation.page_number ? (
                      <span className="inline-flex items-center gap-1 rounded-md bg-purple-50 border border-purple-200/70 px-2 py-0.5 text-[10px] font-semibold text-purple-700">
                        <FileCheck className="h-3 w-3 text-purple-600" />
                        <span>Page {citation.page_number}</span>
                      </span>
                    ) : (
                      <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[10px] text-slate-500 font-medium">
                        Unpaginated
                      </span>
                    )}
                  </div>

                  <div 
                    className="font-bold text-xs text-slate-800 truncate mt-2" 
                    title={citation.document_name}
                  >
                    {citation.document_name}
                  </div>

                  <div className="relative text-[11px] text-slate-600 bg-slate-50/80 p-2.5 rounded-xl border border-slate-200/60 mt-1.5">
                    <Quote className="h-3 w-3 text-indigo-400 absolute top-2 right-2 opacity-60" />
                    <p className={`pr-4 leading-relaxed ${!isExpanded ? 'line-clamp-2' : ''}`}>
                      "{citation.snippet}"
                    </p>
                    {citation.snippet.length > 90 && (
                      <button
                        type="button"
                        onClick={() => toggleSnippetExpand(snippetKey)}
                        className="mt-1 text-[10px] font-semibold text-indigo-600 hover:text-indigo-800 inline-flex items-center gap-0.5"
                      >
                        <span>{isExpanded ? 'Show less' : 'Read more'}</span>
                        {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                      </button>
                    )}
                  </div>
                </div>

                <div className="flex items-center justify-between pt-1 border-t border-slate-100">
                  <span className="text-[10px] text-slate-400 font-mono">
                    Relevance: {(citation.similarity * 100).toFixed(0)}%
                  </span>

                  <button
                    type="button"
                    onClick={() => handleOpenCitationDocument(citation)}
                    disabled={isOpening}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 hover:text-indigo-900 border border-indigo-200/70 px-2.5 py-1 text-[11px] font-semibold transition-all shadow-2xs"
                  >
                    {isOpening ? (
                      <Loader2 className="h-3 w-3 animate-spin text-indigo-600" />
                    ) : (
                      <ExternalLink className="h-3 w-3 text-indigo-600" />
                    )}
                    <span>
                      {isPdf && citation.page_number
                        ? `Page ${citation.page_number}`
                        : 'Open File'}
                    </span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Stop ongoing real-time stream
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    // If we have accumulated partial text, preserve it as an assistant response
    if (streamingText.trim()) {
      const partialMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: streamingText + ' _[generation stopped]_',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: streamingSources,
        grounded: streamingSources.length > 0,
        is_insufficient_context: false,
      };
      setMessages((prev) => [...prev, partialMessage]);
    }
    setIsStreaming(false);
    setIsSubmitting(false);
    setStreamingText('');
    setStreamingSources([]);
  };

  // Send question or follow-up question via real-time SSE streaming
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const query = promptInput.trim();
    if (!query || isSubmitting) return;

    const userMessageId = `user-${Date.now()}`;
    const newUserMessage: ChatMessage = {
      id: userMessageId,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setPromptInput('');
    setIsSubmitting(true);
    setIsStreaming(true);
    setStreamingText('');
    setStreamingSources([]);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let accumulatedText = '';
    let accumulatedSources: ChatSourceItem[] = [];
    let completedNormally = false;

    try {
      await streamRAGQuestion(
        {
          question: query,
          document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
          conversation_id: activeConversationId || undefined,
          match_threshold: 0.3,
          limit: 5,
        },
        {
          onStart: (data) => {
            if (data.conversation_id) {
              setActiveConversationId(data.conversation_id);
              setSearchParams({ c: data.conversation_id });
            }
          },
          onToken: (token) => {
            accumulatedText += token;
            setStreamingText((prev) => prev + token);
          },
          onSources: (data) => {
            accumulatedSources = data.sources || [];
            setStreamingSources(accumulatedSources);
          },
          onComplete: (data) => {
            completedNormally = true;
            if (data.conversation_id) {
              setActiveConversationId(data.conversation_id);
              setSearchParams({ c: data.conversation_id });
            }

            const assistantMessage: ChatMessage = {
              id: `assistant-${Date.now()}`,
              sender: 'assistant',
              text: accumulatedText,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              sources: accumulatedSources,
              document_names: accumulatedSources.map((s) => s.document_name),
              page_numbers: accumulatedSources.map((s) => s.page_number).filter(Boolean) as number[],
              grounded: data.grounded,
              is_insufficient_context: data.is_insufficient_context,
            };

            setMessages((prev) => [...prev, assistantMessage]);
            setIsStreaming(false);
            setStreamingText('');
            setStreamingSources([]);
            reloadConversations();
          },
          onError: (errMsg) => {
            throw new Error(errMsg);
          },
        },
        user?.id,
        session?.access_token,
        controller.signal
      );
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // Handled cleanly via user cancellation
        return;
      }
      if (!completedNormally) {
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          sender: 'assistant',
          text: `Error generating response: ${err.message || 'Failed to connect to the knowledge assistant service.'}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          grounded: false,
          is_insufficient_context: true,
        };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } finally {
      abortControllerRef.current = null;
      setIsSubmitting(false);
      setIsStreaming(false);
      setStreamingText('');
      setStreamingSources([]);
    }
  };

  const sampleQuestions = [
    "What is inheritance in Java and what relationship does it establish?",
    "Explain that in simpler terms.",
    "Give me an example of extending a class.",
  ];

  return (
    <div className="max-w-7xl mx-auto pb-12 font-sans">
      
      {/* 2-Column Responsive Layout: History Sidebar + Chat Main Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* ========================================================================= */}
        {/* CONVERSATION HISTORY SIDEBAR (Desktop: 4 cols, Mobile: Drawer modal) */}
        {/* ========================================================================= */}
        <div className={`lg:col-span-4 space-y-4 ${
          showHistoryDrawer 
            ? 'fixed inset-0 z-50 p-4 bg-slate-900/60 backdrop-blur-sm flex flex-col justify-end lg:static lg:p-0 lg:bg-transparent' 
            : 'hidden lg:block'
        }`}>
          <div className="rounded-3xl border border-indigo-100 bg-white shadow-lg shadow-indigo-100/30 overflow-hidden flex flex-col max-h-[85vh]">
            
            {/* Sidebar Header */}
            <div className="p-4 border-b border-indigo-50 bg-gradient-to-r from-white via-indigo-50/20 to-purple-50/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-indigo-600" />
                <span className="font-bold text-sm text-slate-800">Chat History</span>
                <span className="rounded-full bg-indigo-50 text-indigo-600 border border-indigo-100 px-2 py-0.5 text-[11px] font-semibold">
                  {conversations.length}
                </span>
              </div>

              {/* Close button on mobile drawer */}
              <button
                type="button"
                onClick={() => setShowHistoryDrawer(false)}
                className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 lg:hidden"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* "+ New Conversation" Button */}
            <div className="p-3 border-b border-slate-100 bg-slate-50/50">
              <button
                type="button"
                onClick={handleStartNewConversation}
                className="w-full inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 px-4 py-2.5 text-xs font-semibold text-white shadow-md shadow-indigo-500/20 hover:from-indigo-500 hover:to-purple-500 transition-all active:scale-95"
              >
                <Plus className="h-4 w-4" />
                <span>New Conversation</span>
              </button>
            </div>

            {/* Conversations List */}
            <div className="flex-1 overflow-y-auto p-3 space-y-1.5 min-h-[220px]">
              {isLoadingConversations ? (
                <div className="py-8 text-center text-xs text-slate-400 flex items-center justify-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
                  <span>Loading history...</span>
                </div>
              ) : conversations.length === 0 ? (
                <div className="py-8 text-center px-4 space-y-1">
                  <MessageSquare className="h-8 w-8 text-indigo-200 mx-auto" />
                  <p className="text-xs font-medium text-slate-600">No previous chats</p>
                  <p className="text-[11px] text-slate-400">
                    Ask a question to start your first grounded conversation.
                  </p>
                </div>
              ) : (
                conversations.map((conv) => {
                  const isActive = activeConversationId === conv.id;
                  const isDeleting = deletingConvId === conv.id;
                  const timeFormatted = new Date(conv.updated_at).toLocaleDateString([], {
                    month: 'short',
                    day: 'numeric',
                  });

                  return (
                    <div
                      key={conv.id}
                      onClick={() => handleReopenConversation(conv)}
                      className={`group relative flex items-center justify-between rounded-2xl p-3 text-xs cursor-pointer transition-all border ${
                        isActive
                          ? 'border-indigo-300 bg-indigo-50/80 text-indigo-950 font-semibold shadow-xs'
                          : 'border-slate-100 bg-white hover:border-indigo-200 hover:bg-slate-50/80 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2.5 min-w-0 pr-2">
                        <MessageSquare className={`h-4 w-4 shrink-0 ${
                          isActive ? 'text-indigo-600' : 'text-slate-400 group-hover:text-indigo-500'
                        }`} />
                        <div className="min-w-0">
                          <p className="truncate text-xs font-medium" title={conv.title}>
                            {conv.title}
                          </p>
                          <p className="text-[10px] text-slate-400 font-mono">
                            {timeFormatted} {conv.message_count ? `· ${conv.message_count} msgs` : ''}
                          </p>
                        </div>
                      </div>

                      {/* Delete Conversation Button */}
                      <button
                        type="button"
                        onClick={(e) => handleDeleteConversation(conv.id, e)}
                        disabled={isDeleting}
                        className="rounded-lg p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors opacity-70 group-hover:opacity-100 shrink-0"
                        title="Delete conversation"
                      >
                        {isDeleting ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Trash2 className="h-3.5 w-3.5" />
                        )}
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* CHAT MAIN VIEWPORT (Desktop: 8 cols) */}
        {/* ========================================================================= */}
        <div className="lg:col-span-8 space-y-4">
          <div className="rounded-3xl border border-indigo-100 bg-white shadow-xl shadow-indigo-100/40 overflow-hidden">
            
            {/* Top Bar with Title, Rename, and Mobile History Toggle */}
            <div className="border-b border-indigo-100/80 bg-gradient-to-r from-white via-indigo-50/30 to-purple-50/20 px-6 py-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  {/* Mobile toggle for history drawer */}
                  <button
                    type="button"
                    onClick={() => setShowHistoryDrawer(true)}
                    className="rounded-xl border border-indigo-200 bg-white p-2 text-indigo-600 lg:hidden shadow-2xs"
                    title="View history"
                  >
                    <Clock className="h-4 w-4" />
                  </button>

                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-500/20 shrink-0">
                    <Bot className="h-5 w-5" />
                  </div>

                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      {isEditingTitle ? (
                        <div className="flex items-center gap-1.5">
                          <input
                            type="text"
                            value={editedTitle}
                            onChange={(e) => setEditedTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleSaveRenamedTitle();
                              if (e.key === 'Escape') setIsEditingTitle(false);
                            }}
                            className="rounded-lg border border-indigo-300 px-2.5 py-1 text-xs font-semibold text-slate-800 outline-none focus:ring-2 focus:ring-indigo-200"
                            autoFocus
                          />
                          <button
                            type="button"
                            onClick={handleSaveRenamedTitle}
                            className="rounded p-1 text-emerald-600 hover:bg-emerald-50"
                          >
                            <Check className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setIsEditingTitle(false)}
                            className="rounded p-1 text-slate-400 hover:bg-slate-100"
                          >
                            <X className="h-4 w-4" />
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 group">
                          <h2 className="text-base sm:text-lg font-bold tracking-tight text-slate-900 truncate max-w-[260px] sm:max-w-md">
                            {activeConversationTitle}
                          </h2>
                          {activeConversationId && (
                            <button
                              type="button"
                              onClick={() => {
                                setEditedTitle(activeConversationTitle);
                                setIsEditingTitle(true);
                              }}
                              className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-indigo-600 transition-opacity p-0.5"
                              title="Rename conversation"
                            >
                              <Edit2 className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </div>
                      )}

                      <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 border border-indigo-200/80 px-2 py-0.5 text-[10px] font-semibold text-indigo-700 shrink-0">
                        <Sparkles className="h-3 w-3 text-indigo-500" />
                        Grounded RAG
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Contextual follow-up enabled · Gemini 2.5 Flash
                    </p>
                  </div>
                </div>

                {/* Header Action Buttons */}
                <div className="flex items-center gap-2 self-end sm:self-auto">
                  <button
                    type="button"
                    onClick={handleStartNewConversation}
                    className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-indigo-50/60 px-3 py-1.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-100 transition-colors shadow-2xs"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    <span>New Chat</span>
                  </button>

                  {activeConversationId && (
                    <button
                      type="button"
                      onClick={() => handleDeleteConversation(activeConversationId)}
                      className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-rose-600 hover:border-rose-200 hover:bg-rose-50/40 transition-colors shadow-2xs"
                      title="Delete this conversation"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      <span className="hidden sm:inline">Delete</span>
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Knowledge Source Filter Bar */}
            <div className="border-b border-indigo-50 bg-slate-50/60 px-6 py-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-700">
                  <Layers className="h-3.5 w-3.5 text-indigo-600" />
                  <span>Knowledge Scope:</span>
                  <span className="rounded-full bg-white border border-indigo-100 px-2 py-0.5 text-[11px] font-medium text-indigo-600">
                    {selectedDocIds.length > 0 ? `${selectedDocIds.length} Selected` : 'All Documents Active'}
                  </span>
                </div>

                <button
                  onClick={() => navigate('/documents')}
                  className="text-xs font-medium text-indigo-600 hover:text-indigo-800 transition-colors flex items-center gap-1 self-start sm:self-auto"
                >
                  <span>Manage Documents ({documents.length})</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
              </div>

              {/* Document selection pills */}
              {documents.length > 0 && (
                <div className="mt-2.5 flex flex-wrap gap-2 items-center">
                  <button
                    type="button"
                    onClick={() => setSelectedDocIds([])}
                    className={`rounded-xl px-2.5 py-1 text-xs font-medium transition-all ${
                      selectedDocIds.length === 0
                        ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-2xs'
                        : 'bg-white border border-slate-200 text-slate-600 hover:border-indigo-300 hover:text-slate-900'
                    }`}
                  >
                    All Documents ({documents.length})
                  </button>

                  {documents.map((doc) => {
                    const isSelected = selectedDocIds.includes(doc.id);
                    const isPdf = doc.file_type?.toLowerCase() === 'pdf';
                    const isDocx = doc.file_type?.toLowerCase() === 'docx';
                    return (
                      <button
                        key={doc.id}
                        type="button"
                        onClick={() => toggleDocumentFilter(doc.id)}
                        className={`inline-flex items-center gap-1.5 rounded-xl px-2.5 py-1 text-xs font-medium transition-all border ${
                          isSelected
                            ? 'border-indigo-400 bg-indigo-100/70 text-indigo-900 shadow-2xs'
                            : 'border-slate-200 bg-white text-slate-600 hover:border-indigo-200 hover:text-slate-900'
                        }`}
                      >
                        <FileText className={`h-3 w-3 ${isPdf ? 'text-rose-500' : isDocx ? 'text-blue-500' : 'text-amber-500'}`} />
                        <span className="truncate max-w-[140px]">{doc.file_name}</span>
                        <span className="text-[9px] uppercase font-mono px-1 py-0.2 rounded bg-slate-100 text-slate-500">
                          {doc.file_type}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Chat Messages Body */}
            <div className="p-6 md:p-8 space-y-6 min-h-[420px] bg-slate-50/40">
              
              {/* Empty state: No uploaded documents */}
              {!isLoadingDocs && documents.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-indigo-200 bg-indigo-50/30 p-10 text-center space-y-4 max-w-lg mx-auto my-8">
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-100 text-indigo-600 shadow-sm">
                    <UploadCloud className="h-7 w-7" />
                  </div>
                  <div className="space-y-1.5">
                    <h3 className="text-base font-bold text-slate-900">No Documents Uploaded Yet</h3>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      Upload PDFs, Word documents, or text files to begin asking grounded questions.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => navigate('/documents')}
                    className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-5 py-2.5 text-xs font-semibold text-white shadow-md shadow-indigo-500/25 hover:from-indigo-500 hover:to-purple-500 transition-all active:scale-95"
                  >
                    <FileText className="h-4 w-4" />
                    <span>Upload Documents Now</span>
                  </button>
                </div>
              ) : isLoadingConversationDetail ? (
                <div className="py-20 text-center space-y-3">
                  <Loader2 className="h-6 w-6 animate-spin text-indigo-600 mx-auto" />
                  <p className="text-xs text-slate-500">Loading conversation history and citations...</p>
                </div>
              ) : messages.length === 0 ? (
                /* Welcome Onboarding State */
                <div className="rounded-3xl border border-indigo-100 bg-white p-8 text-center space-y-5 max-w-xl mx-auto shadow-sm">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-50 to-purple-100 text-indigo-600 border border-indigo-200/50 shadow-inner">
                    <Sparkles className="h-6 w-6 text-indigo-600" />
                  </div>
                  <div className="space-y-1.5">
                    <h3 className="text-base sm:text-lg font-bold text-slate-900">
                      Personal AI Knowledge Assistant
                    </h3>
                    <p className="text-xs sm:text-sm text-slate-500 max-w-md mx-auto leading-relaxed">
                      Ask questions and follow-ups grounded in your documents. Answers include verifiable page citations.
                    </p>
                  </div>

                  {/* Suggested Prompts */}
                  <div className="pt-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-3">
                      Try a sample question or follow-up
                    </span>
                    <div className="flex flex-col gap-2">
                      {sampleQuestions.map((question, idx) => (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => setPromptInput(question)}
                          className="group flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 p-3 text-left text-xs font-medium text-slate-700 hover:border-indigo-300 hover:bg-indigo-50/40 hover:text-indigo-900 transition-all shadow-2xs"
                        >
                          <span>"{question}"</span>
                          <ArrowRight className="h-3.5 w-3.5 text-slate-400 group-hover:text-indigo-600 group-hover:translate-x-0.5 transition-all shrink-0 ml-2" />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                /* Chat Messages Thread */
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-3.5 ${
                      msg.sender === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {msg.sender === 'assistant' && (
                      <div className="flex h-9 w-9 shrink-0 select-none items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-600/20 mt-1">
                        <Bot className="h-4 w-4" />
                      </div>
                    )}

                    <div
                      className={`max-w-[88%] md:max-w-[82%] space-y-3.5 shadow-sm transition-all ${
                        msg.sender === 'user'
                          ? 'rounded-2xl rounded-tr-sm bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-5 py-3.5 shadow-md shadow-indigo-500/15'
                          : 'rounded-2xl rounded-tl-sm border border-indigo-100 bg-white text-slate-800 p-5 md:p-6 shadow-indigo-100/30'
                      }`}
                    >
                      {/* Bubble Header */}
                      <div className={`flex items-center justify-between gap-4 text-[11px] pb-2 border-b ${
                        msg.sender === 'user' ? 'border-white/20 text-indigo-100' : 'border-indigo-50 text-slate-500'
                      }`}>
                        <span className="font-semibold tracking-wide">
                          {msg.sender === 'user' ? 'You' : 'AI Knowledge Assistant'}
                        </span>
                        <span className="font-mono text-[10px] opacity-80">{msg.timestamp}</span>
                      </div>

                      {/* Content */}
                      {msg.sender === 'user' ? (
                        <div className="text-xs sm:text-sm leading-relaxed whitespace-pre-wrap font-sans font-medium text-white">
                          {msg.text}
                        </div>
                      ) : (
                        <MarkdownRenderer content={msg.text} />
                      )}

                      {/* Grounding Status Indicator */}
                      {msg.sender === 'assistant' && (
                        <div className="pt-1 flex items-center gap-2 text-[11px]">
                          {msg.is_insufficient_context ? (
                            <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-50 border border-amber-200/80 px-2 py-0.5 text-amber-700 font-medium">
                              <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                              <span>Insufficient context in uploaded documents</span>
                            </span>
                          ) : msg.grounded ? (
                            <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-50 border border-emerald-200/80 px-2 py-0.5 text-emerald-700 font-medium">
                              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                              <span>Grounded in retrieved sources</span>
                            </span>
                          ) : null}
                        </div>
                      )}

                      {/* Citations Tray */}
                      {msg.sources && msg.sources.length > 0 && !msg.is_insufficient_context && (
                        renderCitationsTray(msg.sources)
                      )}
                    </div>
                  </div>
                ))
              )}

              {/* Real-time Streaming Response / Thinking Indicator */}
              {isStreaming && (
                <div className="flex gap-3.5 justify-start">
                  <div className="flex h-9 w-9 shrink-0 select-none items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-600/20 mt-1">
                    <Bot className="h-4 w-4" />
                  </div>

                  {streamingText === '' ? (
                    /* Initial retrieval / thinking phase */
                    <div className="rounded-2xl rounded-tl-sm border border-indigo-100 bg-white p-4 shadow-sm text-xs text-slate-700 flex items-center gap-3">
                      <div className="flex space-x-1">
                        <div className="h-2 w-2 rounded-full bg-indigo-600 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="h-2 w-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="font-medium text-slate-600">
                        Searching knowledge base & connecting real-time stream...
                      </span>
                    </div>
                  ) : (
                    /* Live progressive token streaming bubble */
                    <div className="max-w-[88%] md:max-w-[82%] space-y-3.5 shadow-sm transition-all rounded-2xl rounded-tl-sm border border-indigo-100 bg-white text-slate-800 p-5 md:p-6 shadow-indigo-100/30">
                      <div className="flex items-center justify-between gap-4 text-[11px] pb-2 border-b border-indigo-50 text-slate-500">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold tracking-wide text-slate-700">AI Knowledge Assistant</span>
                          <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 border border-indigo-200/80 px-2 py-0.5 text-[9px] font-semibold text-indigo-700 animate-pulse">
                            <Sparkles className="h-2.5 w-2.5 text-indigo-500" />
                            Streaming Live
                          </span>
                        </div>
                        <span className="font-mono text-[10px] text-indigo-600 font-semibold animate-pulse">
                          Generating...
                        </span>
                      </div>

                      <div className="relative text-xs sm:text-sm leading-relaxed text-slate-800">
                        <MarkdownRenderer content={streamingText} />
                        <span className="inline-block w-1.5 h-4 ml-1 bg-indigo-600 animate-pulse align-middle" />
                      </div>

                      {streamingSources.length > 0 && renderCitationsTray(streamingSources)}
                    </div>
                  )}
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Query Input Box */}
            <div className="border-t border-indigo-100/90 bg-white p-4 md:p-5">
              <form onSubmit={handleSendMessage} className="flex items-center gap-3">
                <div className="relative flex-1">
                  <input
                    type="text"
                    value={promptInput}
                    onChange={(e) => setPromptInput(e.target.value)}
                    disabled={isSubmitting || documents.length === 0}
                    placeholder={
                      documents.length > 0
                        ? messages.length > 0
                          ? "Ask a follow-up (e.g. 'Explain that in simpler terms' or 'Give me an example')..."
                          : "Ask a question about your uploaded documents..."
                        : "Please upload documents to begin asking questions..."
                    }
                    className="w-full rounded-2xl border border-slate-200 bg-slate-50/70 py-3.5 pl-4 pr-10 text-xs sm:text-sm text-slate-900 placeholder-slate-400 outline-none focus:border-indigo-500 focus:bg-white focus:ring-4 focus:ring-indigo-100 transition-all disabled:opacity-50"
                  />
                </div>
                
                {isStreaming && (
                  <button
                    type="button"
                    onClick={handleStopStreaming}
                    className="inline-flex items-center gap-1.5 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3.5 text-xs sm:text-sm font-semibold text-rose-700 hover:bg-rose-100 transition-all active:scale-95 shrink-0 shadow-2xs"
                    title="Stop response generation"
                  >
                    <Square className="h-3.5 w-3.5 fill-current" />
                    <span className="hidden sm:inline">Stop</span>
                  </button>
                )}

                <button
                  type="submit"
                  disabled={!promptInput.trim() || isSubmitting || documents.length === 0}
                  className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 px-5 sm:px-6 py-3.5 text-xs sm:text-sm font-semibold text-white hover:from-indigo-500 hover:to-purple-500 disabled:opacity-40 shadow-md shadow-indigo-600/25 transition-all active:scale-95 shrink-0"
                >
                  {isSubmitting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                  <span className="hidden sm:inline">Ask Assistant</span>
                </button>
              </form>

              {/* Footer Metadata */}
              <div className="mt-2.5 flex items-center justify-between text-[11px] text-slate-400 px-1">
                <span className="flex items-center gap-1">
                  <Shield className="h-3 w-3 text-indigo-500" />
                  Multi-tenant isolated with Supabase RLS
                </span>
                <span>Contextual Follow-ups · Formatted Citations</span>
              </div>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
};
