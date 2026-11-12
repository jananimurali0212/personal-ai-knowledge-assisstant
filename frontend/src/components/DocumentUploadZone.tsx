import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  CheckCircle2, 
  AlertCircle, 
  X, 
  Loader2, 
  FileUp
} from 'lucide-react';
import { 
  isSupportedFile, 
  uploadUserDocument, 
  formatBytes,
  getFileExtension 
} from '../services/documents';
import { DocumentRecord } from '../types';

interface DocumentUploadZoneProps {
  userId: string;
  onUploadSuccess: (newDoc: DocumentRecord) => void;
  compact?: boolean;
}

export const DocumentUploadZone: React.FC<DocumentUploadZoneProps> = ({
  userId,
  onUploadSuccess,
  compact = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const resetState = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    setIsUploading(false);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileSelect = (file: File) => {
    setErrorMessage(null);
    setSuccessMessage(null);

    const validation = isSupportedFile(file);
    if (!validation.valid) {
      setErrorMessage(validation.reason || 'Invalid file format.');
      return;
    }

    setSelectedFile(file);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMessage(null);
    setSuccessMessage(null);
    setUploadProgress(20);

    // Simulate natural progress step before cloud completion
    const timer = setInterval(() => {
      setUploadProgress((prev) => (prev < 85 ? prev + 15 : prev));
    }, 150);

    try {
      const { data, error } = await uploadUserDocument(selectedFile, userId);
      clearInterval(timer);

      if (error) {
        setErrorMessage(error);
        setIsUploading(false);
        setUploadProgress(0);
      } else if (data) {
        setUploadProgress(100);
        setSuccessMessage(`"${data.file_name}" uploaded successfully!`);
        onUploadSuccess(data);
        setTimeout(() => {
          resetState();
        }, 1800);
      }
    } catch (err: any) {
      clearInterval(timer);
      setErrorMessage(err.message || 'An error occurred during upload.');
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const getFormatIcon = (filename: string) => {
    const ext = getFileExtension(filename);
    if (ext === 'pdf') {
      return <div className="h-8 w-8 rounded-lg bg-rose-500/10 text-rose-400 flex items-center justify-center font-bold text-xs uppercase">PDF</div>;
    }
    if (ext === 'docx') {
      return <div className="h-8 w-8 rounded-lg bg-blue-500/10 text-blue-400 flex items-center justify-center font-bold text-xs uppercase">DOC</div>;
    }
    return <div className="h-8 w-8 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center font-bold text-xs uppercase">TXT</div>;
  };

  return (
    <div className="w-full">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,text/plain"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
          }
        }}
      />

      {/* Drop Zone Box */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all duration-200 ${
          compact ? 'p-5' : 'p-8'
        } ${
          isDragging
            ? 'border-purple-500 bg-purple-500/10 scale-[1.01]'
            : 'border-slate-700/80 bg-[#0B0F19]/60 hover:border-purple-500/50 hover:bg-[#0F172A]/80'
        }`}
      >
        <div className="flex flex-col items-center text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-tr from-purple-600/20 to-indigo-600/20 text-purple-400 border border-purple-500/30 group-hover:scale-110 transition-transform shadow-inner">
            <UploadCloud className="h-6 w-6" />
          </div>

          <h3 className="mt-3 text-sm font-semibold text-white">
            {compact ? 'Quick Upload Document' : 'Upload knowledge document'}
          </h3>
          <p className="mt-1 text-xs text-slate-400 max-w-sm">
            Drag & drop your files here, or <span className="font-semibold text-purple-400 underline decoration-purple-400/40">browse files</span> from your computer
          </p>

          <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
            <span className="rounded-md border border-rose-500/20 bg-rose-500/10 px-2 py-0.5 text-[10px] font-semibold text-rose-300">
              PDF
            </span>
            <span className="rounded-md border border-blue-500/20 bg-blue-500/10 px-2 py-0.5 text-[10px] font-semibold text-blue-300">
              DOCX
            </span>
            <span className="rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
              TXT
            </span>
            <span className="text-[10px] text-slate-400">· Max 25 MB</span>
          </div>
        </div>
      </div>

      {/* Selected File Stage & Progress */}
      {selectedFile && (
        <div className="mt-3 rounded-xl border border-slate-700/90 bg-[#0F172A] p-4 shadow-lg animate-fadeIn">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center space-x-3 min-w-0">
              {getFormatIcon(selectedFile.name)}
              <div className="min-w-0">
                <p className="truncate text-xs font-semibold text-white">
                  {selectedFile.name}
                </p>
                <p className="text-[11px] text-slate-400">
                  {formatBytes(selectedFile.size)} · Ready to upload
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2 shrink-0">
              {!isUploading && !successMessage && (
                <button
                  type="button"
                  onClick={resetState}
                  className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
                  aria-label="Remove selected file"
                >
                  <X className="h-4 w-4" />
                </button>
              )}

              <button
                type="button"
                onClick={handleUpload}
                disabled={isUploading}
                className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 px-4 py-1.5 text-xs font-semibold text-white shadow-md shadow-purple-600/20 hover:opacity-95 active:scale-95 disabled:opacity-50"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <FileUp className="h-3.5 w-3.5" />
                    <span>Upload to Cloud</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Progress Bar */}
          {isUploading && (
            <div className="mt-3 space-y-1">
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>Sending to private Supabase Storage...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 transition-all duration-200"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Success Notification */}
      {successMessage && (
        <div className="mt-3 flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-300 animate-fadeIn">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>{successMessage}</span>
        </div>
      )}

      {/* Error Notification */}
      {errorMessage && (
        <div className="mt-3 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300 animate-shake">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
          <div className="flex-1 leading-relaxed">
            <span className="font-semibold text-rose-200">Upload Failed: </span>
            {errorMessage}
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-rose-400 hover:text-rose-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
};
