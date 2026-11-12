export type NavPage = 
  | 'dashboard' 
  | 'documents' 
  | 'assistant' 
  | 'search' 
  | 'settings'
  | 'new-review' 
  | 'history';

export type DocumentProcessingStatus = 'uploaded' | 'processing' | 'ready' | 'failed';

export interface DocumentRecord {
  id: string;
  user_id: string;
  file_name: string;
  file_type: 'pdf' | 'docx' | 'txt' | string;
  file_size: number;
  storage_path: string;
  status: DocumentProcessingStatus;
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
