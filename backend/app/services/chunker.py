"""Document chunking service with sentence-boundary preservation and token overlap."""
import re
import logging
from typing import List, Optional
from uuid import UUID
from dataclasses import dataclass

from .extractor import ExtractedPage

logger = logging.getLogger(__name__)

# Heuristic: 1 token is approximately 4 characters in English text
CHARS_PER_TOKEN = 4.0

def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a string based on character and word count."""
    if not text:
        return 0
    # Average between character-based (4 chars/token) and word-based (0.75 words/token) heuristics
    char_tokens = len(text) / CHARS_PER_TOKEN
    word_tokens = len(text.split()) / 0.75
    return max(1, int((char_tokens + word_tokens) / 2))

@dataclass
class ProcessedChunk:
    """Represents a chunk of document text ready for vector embedding and storage."""
    document_id: UUID
    chunk_index: int
    content: str
    page_number: int
    char_count: int
    estimated_tokens: int

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences avoiding mid-sentence cuts.
    Respects standard punctuation (. ! ?) followed by whitespace, newlines,
    or quotes.
    """
    if not text:
        return []

    # Regex splits on sentence-ending punctuation followed by whitespace or line break
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
    raw_sentences = re.split(sentence_pattern, text)

    sentences: List[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # If an individual paragraph/sentence is extremely long without punctuation,
        # break it into clause-level or word-level pieces
        if len(s_clean) > 2000:
            words = s_clean.split(" ")
            current_piece: List[str] = []
            current_len = 0
            for w in words:
                current_piece.append(w)
                current_len += len(w) + 1
                if current_len >= 1200:
                    sentences.append(" ".join(current_piece))
                    current_piece = []
                    current_len = 0
            if current_piece:
                sentences.append(" ".join(current_piece))
        else:
            sentences.append(s_clean)

    return sentences

def chunk_document(
    pages: List[ExtractedPage],
    document_id: UUID,
    target_tokens: int = 800,
    overlap_tokens: int = 150
) -> List[ProcessedChunk]:
    """
    Splits multi-page or single-page documents into meaningful chunks.
    - Uses target chunk size of approximately ~800 tokens (~3200 characters).
    - Uses overlap of approximately ~150 tokens (~600 characters).
    - Preserves sentence boundaries whenever possible.
    - Preserves document_id, chunk_index, and page_number.
    - Handles small documents (< 800 tokens) as a single complete chunk.
    - Gracefully handles empty inputs.
    """
    if not pages:
        return []

    # Filter out empty pages
    valid_pages = [p for p in pages if p.text and p.text.strip()]
    if not valid_pages:
        return []

    # Flatten pages into page-annotated sentence units: (page_number, sentence_text)
    annotated_sentences: List[tuple[int, str]] = []
    total_doc_text = []

    for page in valid_pages:
        total_doc_text.append(page.text)
        # Split page into paragraphs
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_sentences = split_into_sentences(para)
            for s in para_sentences:
                annotated_sentences.append((page.page_number, s))

    if not annotated_sentences:
        return []

    # Check if total text is smaller than target token size
    full_text = "\n\n".join(total_doc_text)
    total_tokens = estimate_tokens(full_text)
    if total_tokens <= target_tokens:
        # Document with very little text -> single chunk
        return [
            ProcessedChunk(
                document_id=document_id,
                chunk_index=0,
                content=full_text,
                page_number=valid_pages[0].page_number,
                char_count=len(full_text),
                estimated_tokens=total_tokens,
            )
        ]

    chunks: List[ProcessedChunk] = []
    chunk_index = 0
    current_sentences: List[str] = []
    current_page = annotated_sentences[0][0]
    current_tokens = 0
    i = 0

    while i < len(annotated_sentences):
        page_num, sentence = annotated_sentences[i]
        s_tokens = estimate_tokens(sentence)

        # If adding this sentence stays within target or current buffer is empty
        if current_tokens + s_tokens <= target_tokens or not current_sentences:
            if not current_sentences:
                current_page = page_num
            current_sentences.append(sentence)
            current_tokens += s_tokens
            i += 1
        else:
            # Chunk is full, finalize current chunk
            chunk_content = " ".join(current_sentences).strip()
            chunks.append(
                ProcessedChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    page_number=current_page,
                    char_count=len(chunk_content),
                    estimated_tokens=estimate_tokens(chunk_content),
                )
            )
            chunk_index += 1

            # Build overlap buffer from the tail of current_sentences
            overlap_sentences: List[str] = []
            overlap_acc_tokens = 0
            for prev_s in reversed(current_sentences):
                prev_tokens = estimate_tokens(prev_s)
                if overlap_acc_tokens + prev_tokens <= overlap_tokens or not overlap_sentences:
                    overlap_sentences.insert(0, prev_s)
                    overlap_acc_tokens += prev_tokens
                else:
                    break

            # Restart buffer with overlap
            current_sentences = list(overlap_sentences)
            current_tokens = estimate_tokens(" ".join(current_sentences))
            # Current page stays with the next incoming sentence
            current_page = page_num

    # Flush any remaining sentences in the final buffer
    if current_sentences:
        chunk_content = " ".join(current_sentences).strip()
        # Avoid creating a tiny duplicate chunk if it's identical to the previous overlap
        if not chunks or chunk_content != chunks[-1].content:
            chunks.append(
                ProcessedChunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    page_number=current_page,
                    char_count=len(chunk_content),
                    estimated_tokens=estimate_tokens(chunk_content),
                )
            )

    return chunks

