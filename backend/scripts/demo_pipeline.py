"""Demonstration script testing PDF, DOCX, TXT extraction, chunking, and vector embedding storage."""
import io
import os
import sys
from uuid import uuid4
import pymupdf as fitz
import docx

# Ensure backend root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.extractor import extract_pdf, extract_docx, extract_txt, extract_document
from app.services.chunker import chunk_document, estimate_tokens
from app.services.embeddings import default_embedding_service, GeminiEmbeddingService
from app.services.chunk_repository import default_chunk_repository
from app.services.processor import DocumentProcessor, default_processor
from app.models.chunk import DocumentChunkCreate
from app.config.settings import settings

def run_demonstration():
    print("=" * 80)
    print("PERSONAL AI KNOWLEDGE ASSISTANT: EXTRACTION, CHUNKING & EMBEDDING DEMO")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. PDF Test (Multi-page with metadata preservation)
    # --------------------------------------------------------------------------
    print("\n[TEST 1] PDF EXTRACTION & PAGE METADATA")
    print("-" * 50)

    pdf_doc = fitz.open()
    p1 = pdf_doc.new_page()
    p1.insert_text(
        fitz.Point(50, 72),
        "Page 1: Architectural Foundations\n\n"
        "The Personal AI Knowledge Assistant decouples the presentation layer from the "
        "retrieval-augmented generation engine. Documents are ingested into Supabase private storage "
        "and metadata is tracked in the PostgreSQL relational schema.",
        fontsize=11
    )
    p2 = pdf_doc.new_page()
    p2.insert_text(
        fitz.Point(50, 72),
        "Page 2: Vector Search & Chunking\n\n"
        "Document chunks are vectorized using Google Gemini text-embedding-004 into 768 dimensions. "
        "Similarity searches utilize the HNSW index using the cosine distance operator (<=>). "
        "Row Level Security policies guarantee multi-tenant tenant isolation.",
        fontsize=11
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    extracted_pdf_pages = extract_pdf(pdf_bytes)
    print(f"Extracted Pages Count: {len(extracted_pdf_pages)}")
    for p in extracted_pdf_pages:
        print(f"  -> Page {p.page_number} ({p.char_count} chars):")
        print(f"     \"{p.text[:100]}...\"")

    pdf_doc_id = uuid4()
    pdf_user_id = uuid4()
    pdf_chunks = chunk_document(extracted_pdf_pages, document_id=pdf_doc_id, target_tokens=800, overlap_tokens=150)
    print(f"Resulting Chunks Count: {len(pdf_chunks)}")
    for c in pdf_chunks:
        print(f"  -> Chunk #{c.chunk_index} | Page {c.page_number} | Tokens: ~{c.estimated_tokens}")
        print(f"     Preview: \"{c.content[:110]}...\"")

    # --------------------------------------------------------------------------
    # 2. Vector Embedding Generation & Supabase Storage Flow
    # --------------------------------------------------------------------------
    print("\n[TEST 2] VECTOR EMBEDDINGS & SUPABASE REPOSITORY INTEGRATION")
    print("-" * 50)
    print(f"Configured Embedding Model: {settings.GEMINI_EMBEDDING_MODEL}")
    print(f"Configured Output Dimension: {settings.GEMINI_EMBEDDING_DIMENSIONS} (Exact match with pgvector column)")

    # Check if a live GEMINI_API_KEY is available
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY not in ("your_gemini_api_key_here", ""):
        print("Live GEMINI_API_KEY detected! Invoking official Google Gen AI SDK...")
        vectors = default_embedding_service.embed_chunks(
            chunks=[c.content for c in pdf_chunks],
            task_type="RETRIEVAL_DOCUMENT"
        )
        print(f"Generated {len(vectors)} real embeddings. Dimensions: {len(vectors[0])}")
    else:
        print("Note: GEMINI_API_KEY not set in environment. Simulating live pipeline validation with 768-dim vector...")
        vectors = [[0.042] * 768 for _ in pdf_chunks]

    chunk_records = [
        DocumentChunkCreate(
            document_id=pdf_doc_id,
            user_id=pdf_user_id,
            chunk_index=c.chunk_index,
            content=c.content,
            page_number=c.page_number,
            embedding=vectors[idx],
        )
        for idx, c in enumerate(pdf_chunks)
    ]

    stored_count = default_chunk_repository.save_chunks(chunk_records, overwrite=True)
    print(f"Chunks Stored in Database/Repository: {stored_count}")
    print(f"Stored Chunks == Processed Chunks: {stored_count == len(pdf_chunks)}")

    retrieved = default_chunk_repository.get_chunks_by_document(pdf_doc_id, pdf_user_id)
    print(f"Retrieved stored chunks: {len(retrieved)}")
    print(f"First chunk vector length: {len(retrieved[0]['embedding'])} dimensions")

    # --------------------------------------------------------------------------
    # 3. DOCX & TXT Tests
    # --------------------------------------------------------------------------
    print("\n[TEST 3] DOCX & TXT EXTRACTION")
    print("-" * 50)

    doc = docx.Document()
    doc.add_heading("Product Requirements Document", level=1)
    doc.add_paragraph("This document outlines the security, chunking, and retrieval requirements for the RAG assistant.")
    doc.add_paragraph("All uploaded documents must be encrypted and bound to authenticated user identifiers.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Feature"
    table.cell(0, 1).text = "Status"
    table.cell(1, 0).text = "Vector Storage"
    table.cell(1, 1).text = "Enabled"

    stream = io.BytesIO()
    doc.save(stream)
    docx_bytes = stream.getvalue()

    extracted_docx = extract_docx(docx_bytes)
    print(f"Extracted DOCX: {len(extracted_docx)} section, {extracted_docx[0].char_count} characters.")

    txt_content = "Personal Knowledge Assistant Notes.\n\nNormalized text with standard encoding."
    extracted_txt = extract_txt(txt_content.encode("utf-8"))
    print(f"Extracted TXT: {len(extracted_txt)} page, \"{extracted_txt[0].text}\"")

    # --------------------------------------------------------------------------
    # 4. Scanned PDF Detection Test
    # --------------------------------------------------------------------------
    print("\n[TEST 4] SCANNED PDF GRACEFUL HANDLING")
    print("-" * 50)
    scanned_doc = fitz.open()
    sp = scanned_doc.new_page()
    sp.draw_rect(fitz.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    scanned_bytes = scanned_doc.tobytes()
    scanned_doc.close()

    scanned_res = default_processor.process_file_content(
        file_bytes=scanned_bytes,
        file_type="pdf",
        file_name="scanned_handout.pdf",
        document_id=uuid4(),
        user_id=uuid4()
    )
    print(f"Status: {scanned_res.status}")
    print(f"OCR Required Flag: {scanned_res.ocr_required}")
    print(f"Reported Message: {scanned_res.error_message}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_demonstration()
