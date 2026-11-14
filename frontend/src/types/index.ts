export type NavPage = 
  | 'dashboard' 
  | 'documents' 
  | 'assistant' 
  | 'search' 
  | 'settings'
  | 'new-review' 
  | 'history';

export type DocumentProcessingStatus = 'uploaded' | 'processing' | 'ready' | 'failed';

export interface KeyTerminologyItem {
  term: string;
  definition: string;
}

export interface DocumentSummaryContent {
  executive_summary: string;
  main_topics: string[];
  important_points: string[];
  key_terminology: KeyTerminologyItem[];
  important_conclusions: string[];
  total_sections_analyzed: number;
  is_hierarchical: boolean;
  model_used: string;
  generated_at: string;
}

export interface DocumentSummaryResponse {
  document_id: string;
  user_id: string;
  file_name: string;
  file_type: string;
  status: 'not_generated' | 'generating' | 'completed' | 'failed' | string;
  summary?: DocumentSummaryContent | null;
  cached?: boolean;
  error_message?: string | null;
  created_at?: string;
}

export interface DocumentRecord {
  id: string;
  user_id: string;
  file_name: string;
  file_type: 'pdf' | 'docx' | 'txt' | string;
  file_size: number;
  storage_path: string;
  status: DocumentProcessingStatus;
  summary?: DocumentSummaryContent | null;
  summary_status?: 'not_generated' | 'generating' | 'completed' | 'failed' | string;
  summarized_at?: string | null;
  created_at: string;
  updated_at?: string;
}

export interface DocumentStats {
  totalDocuments: number;
  totalSizeBytes: number;
  pdfCount: number;
  docxCount: number;
  txtCount: number;
}

export interface DocumentChunk {
  id: string;
  user_id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  page_number: number;
  embedding?: number[];
  created_at: string;
}

export interface ChunkSimilarityMatch {
  id: string;
  document_id: string;
  user_id: string;
  chunk_index: number;
  content: string;
  page_number: number;
  similarity: number;
}

export type SupportedLanguage = 
  | 'python'
  | 'java'
  | 'c'
  | 'cpp'
  | 'javascript'
  | 'typescript'
  | 'go'
  | 'rust'
  | 'php'
  | 'csharp'
  | 'html'
  | 'css'
  | 'sql';

export interface LanguageOption {
  id: SupportedLanguage;
  name: string;
  extension: string;
  monacoLang: string;
  sampleSnippet: string;
}

export interface ReviewHistoryItem {
  id: string;
  projectName: string;
  fileName: string;
  language: string;
  reviewDate: string;
  issueCount: number;
  status: 'Completed' | 'Pending' | 'Needs Attention';
  criticalIssues: number;
}

export interface ChatSourceItem {
  chunk_id?: string;
  chunk_identifier?: string;
  document_id: string;
  document_name: string;
  page_number?: number | null;
  chunk_index: number;
  similarity: number;
  snippet: string;
  relevant_excerpt?: string;
  source_id?: string;
  has_page_number?: boolean;
  file_type?: string;
}

export interface ChatRetrievalMetadata {
  total_chunks_retrieved: number;
  match_threshold: number;
  embedding_model: string;
  generation_model: string;
  grounded: boolean;
  is_insufficient_context: boolean;
  tokens_used?: number;
}

export interface ChatAskRequest {
  question: string;
  document_ids?: string[];
  conversation_id?: string;
  match_threshold?: number;
  limit?: number;
  temperature?: number;
  max_output_tokens?: number;
}

export interface ChatAskResponse {
  conversation_id?: string;
  question: string;
  answer: string;
  sources: ChatSourceItem[];
  document_names: string[];
  page_numbers: number[];
  relevant_chunk_ids: string[];
  retrieval_metadata: ChatRetrievalMetadata;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  sources?: ChatSourceItem[];
  document_names?: string[];
  page_numbers?: number[];
  grounded?: boolean;
  is_insufficient_context?: boolean;
}

export interface ConversationRecord {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface ChatMessageDbRecord {
  id: string;
  conversation_id: string;
  user_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources: ChatSourceItem[];
  created_at: string;
}

export interface ConversationDetailResponse {
  conversation: ConversationRecord;
  messages: ChatMessageDbRecord[];
}

// ==========================================
// Quiz Generation & Assessment Types
// ==========================================

export type QuizDifficulty = 'easy' | 'medium' | 'hard';
export type QuestionType = 'multiple_choice' | 'true_false' | 'short_answer';

export interface QuizQuestionStudentView {
  id: string;
  quiz_id: string;
  question_index: number;
  question_type: QuestionType;
  question_text: string;
  options: string[];
  page_number?: number | null;
}

export interface QuizSummary {
  id: string;
  document_id: string;
  document_name: string;
  user_id: string;
  title: string;
  difficulty: QuizDifficulty;
  total_questions: number;
  question_types: QuestionType[];
  created_at: string;
}

export interface QuizResponse {
  quiz: QuizSummary;
  questions: QuizQuestionStudentView[];
}

export interface GenerateQuizRequest {
  difficulty?: QuizDifficulty;
  num_questions?: number;
  question_types?: QuestionType[];
  title?: string;
}

export interface QuizSubmitRequest {
  answers: Record<string, string>;
}

export interface QuestionReviewItem {
  question_id: string;
  question_index: number;
  question_type: QuestionType;
  question_text: string;
  options: string[];
  user_answer: string;
  correct_answer: string;
  is_correct: boolean;
  explanation: string;
  source_excerpt: string;
  page_number?: number | null;
}

export interface QuizSubmitResponse {
  attempt_id: string;
  quiz_id: string;
  score: number;
  total_points: number;
  percentage: number;
  passed: boolean;
  completed_at: string;
  questions: QuestionReviewItem[];
}

export interface QuizAttemptHistoryItem {
  id: string;
  quiz_id: string;
  score: number;
  total_points: number;
  percentage: number;
  completed_at: string;
  review_details: QuestionReviewItem[];
}

