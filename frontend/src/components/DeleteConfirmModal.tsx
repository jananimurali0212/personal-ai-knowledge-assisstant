import React from 'react';
import { Trash2, AlertTriangle, Loader2 } from 'lucide-react';
import { DocumentRecord } from '../types';

interface DeleteConfirmModalProps {
  document: DocumentRecord | null;
  isOpen: boolean;
  isLoading: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

export const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({
  document,
  isOpen,
  isLoading,
  onClose,
  onConfirm,
}) => {
  if (!isOpen || !document) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
      <div 
        className="w-full max-w-md rounded-2xl border border-slate-800 bg-[#0F172A] p-6 shadow-2xl text-slate-100"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-dialog-title"
      >
        <div className="flex items-start gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <h3 id="delete-dialog-title" className="text-base font-semibold text-white">
              Delete Document
            </h3>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed">
              Are you sure you want to permanently delete{' '}
              <span className="font-semibold text-slate-200 break-all">"{document.file_name}"</span>?
              This will remove the file from private Supabase Storage and delete its database metadata.
            </p>
          </div>
        </div>

        <div className="mt-6 flex items-center justify-end gap-3 border-t border-slate-800/80 pt-4">
          <button
            type="button"
            onClick={onClose}
            disabled={isLoading}
            className="rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isLoading}
            className="inline-flex items-center gap-2 rounded-xl bg-rose-600 px-4 py-2 text-xs font-semibold text-white hover:bg-rose-500 shadow-md shadow-rose-600/20 transition-all disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Deleting...</span>
              </>
            ) : (
              <>
                <Trash2 className="h-3.5 w-3.5" />
                <span>Delete Permanently</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

