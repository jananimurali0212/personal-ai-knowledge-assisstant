import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const handleCopyCode = async (code: string, index: number) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch {
      // Fallback
    }
  };

  // Helper to parse inline styles (bold, italic, code, citation badges)
  const renderInlineFormatted = (text: string): React.ReactNode => {
    // Regex splits into tokens: code (`...`), bold (**...**), citations ([Source X]), and plain text
    const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[Source\s+\d+\])/g);

    return parts.map((part, i) => {
      if (!part) return null;

      // Inline code
      if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
        return (
          <code
            key={i}
            className="rounded-md bg-indigo-50/80 px-1.5 py-0.5 font-mono text-xs font-semibold text-indigo-700 border border-indigo-100"
          >
            {part.slice(1, -1)}
          </code>
        );
      }

      // Bold text
      if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
        return (
          <strong key={i} className="font-semibold text-slate-900">
            {part.slice(2, -2)}
          </strong>
        );
      }

      // Inline Citation Pill [Source X]
      if (/^\[Source\s+\d+\]$/i.test(part)) {
        return (
          <span
            key={i}
            className="inline-flex items-center mx-1 px-1.5 py-0.5 rounded-md text-[11px] font-bold bg-indigo-100/90 text-indigo-700 border border-indigo-200 shadow-2xs select-none"
            title="Grounding Citation"
          >
            {part}
          </span>
        );
      }

      return <span key={i}>{part}</span>;
    });
  };

  // Split lines into blocks (fenced code, headings, bullet lists, numbered lists, blockquotes, paragraphs)
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: { type: 'ul' | 'ol'; items: string[] } | null = null;
  let inCodeBlock = false;
  let codeBlockLang = '';
  let codeBlockLines: string[] = [];
  let codeBlockCounter = 0;

  const flushList = () => {
    if (!currentList) return;
    const listIndex = elements.length;
    if (currentList.type === 'ul') {
      elements.push(
        <ul key={`ul-${listIndex}`} className="my-2.5 space-y-1.5 pl-1.5">
          {currentList.items.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2 text-xs sm:text-sm text-slate-700 leading-relaxed">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
              <span>{renderInlineFormatted(item)}</span>
            </li>
          ))}
        </ul>
      );
    } else {
      elements.push(
        <ol key={`ol-${listIndex}`} className="my-2.5 space-y-1.5 pl-1.5">
          {currentList.items.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2 text-xs sm:text-sm text-slate-700 leading-relaxed">
              <span className="shrink-0 font-semibold text-indigo-600 font-mono text-xs mt-0.5">
                {idx + 1}.
              </span>
              <span>{renderInlineFormatted(item)}</span>
            </li>
          ))}
        </ol>
      );
    }
    currentList = null;
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();

    // Check for fenced code block toggle
    if (trimmed.startsWith('```')) {
      if (inCodeBlock) {
        // End of code block
        flushList();
        const codeText = codeBlockLines.join('\n');
        const codeIdx = codeBlockCounter++;
        const lang = codeBlockLang || 'code';

        elements.push(
          <div
            key={`code-${index}`}
            className="my-3 overflow-hidden rounded-xl border border-slate-700 bg-slate-900 text-slate-100 shadow-md"
          >
            <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-3.5 py-1.5 text-xs text-slate-400 font-mono">
              <span className="uppercase tracking-wider text-[11px] font-semibold text-indigo-400">
                {lang}
              </span>
              <button
                type="button"
                onClick={() => handleCopyCode(codeText, codeIdx)}
                className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
                title="Copy code"
              >
                {copiedIndex === codeIdx ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="overflow-x-auto p-3 text-xs leading-relaxed font-mono text-slate-200">
              <code>{codeText}</code>
            </pre>
          </div>
        );

        inCodeBlock = false;
        codeBlockLang = '';
        codeBlockLines = [];
        return;
      } else {
        // Start of code block
        flushList();
        inCodeBlock = true;
        codeBlockLang = trimmed.slice(3).trim();
        codeBlockLines = [];
        return;
      }
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      return;
    }

    // Empty line separates blocks
    if (!trimmed) {
      flushList();
      return;
    }

    // Headings
    if (trimmed.startsWith('### ')) {
      flushList();
      elements.push(
        <h4
          key={`h3-${index}`}
          className="mt-3.5 mb-1.5 text-sm sm:text-base font-bold text-slate-900 tracking-tight"
        >
          {renderInlineFormatted(trimmed.slice(4))}
        </h4>
      );
      return;
    }

    if (trimmed.startsWith('## ')) {
      flushList();
      elements.push(
        <h3
          key={`h2-${index}`}
          className="mt-4 mb-2 text-base sm:text-lg font-bold text-slate-900 tracking-tight pb-1 border-b border-indigo-100/60"
        >
          {renderInlineFormatted(trimmed.slice(3))}
        </h3>
      );
      return;
    }

    if (trimmed.startsWith('# ')) {
      flushList();
      elements.push(
        <h2
          key={`h1-${index}`}
          className="mt-4 mb-2 text-lg sm:text-xl font-extrabold text-slate-900 tracking-tight pb-1 border-b border-indigo-200/80"
        >
          {renderInlineFormatted(trimmed.slice(2))}
        </h2>
      );
      return;
    }

    // Blockquote
    if (trimmed.startsWith('> ')) {
      flushList();
      elements.push(
        <div
          key={`quote-${index}`}
          className="my-2.5 rounded-r-xl border-l-4 border-indigo-500 bg-indigo-50/50 p-2.5 text-xs sm:text-sm text-slate-700 italic"
        >
          {renderInlineFormatted(trimmed.slice(2))}
        </div>
      );
      return;
    }

    // Unordered list item
    if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
      const itemText = trimmed.slice(2);
      if (!currentList || currentList.type !== 'ul') {
        flushList();
        currentList = { type: 'ul', items: [itemText] };
      } else {
        currentList.items.push(itemText);
      }
      return;
    }

    // Ordered list item
    const olMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
    if (olMatch) {
      const itemText = olMatch[2];
      if (!currentList || currentList.type !== 'ol') {
        flushList();
        currentList = { type: 'ol', items: [itemText] };
      } else {
        currentList.items.push(itemText);
      }
      return;
    }

    // Standard paragraph line
    flushList();
    elements.push(
      <p key={`p-${index}`} className="my-1.5 text-xs sm:text-sm text-slate-800 leading-relaxed">
        {renderInlineFormatted(line)}
      </p>
    );
  });

  // Flush any trailing list or unclosed code block
  flushList();
  if (inCodeBlock && codeBlockLines.length > 0) {
    elements.push(
      <pre key="trailing-code" className="my-2 rounded-xl bg-slate-900 p-3 text-xs text-slate-100 font-mono">
        <code>{codeBlockLines.join('\n')}</code>
      </pre>
    );
  }

  return <div className={`space-y-1 ${className}`}>{elements}</div>;
};
