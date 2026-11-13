"""Comprehensive unit and integration tests for document processing pipeline."""
import io
from uuid import UUID, uuid4
import pytest
import pymupdf as fitz  # PyMuPDF
import docx  # python-docx
from app.services.normalizer import normalize_text
from app.services.extractor import (
    extract_pdf,
    extract_docx,
    extract_txt,
    extract_document,
    ScannedPDFException,
    EmptyDocumentException,
    UnsupportedFormatException,
)
from app.services.chunker import (
    chunk_document,
    estimate_tokens,
    ExtractedPage,
    ProcessedChunk,
)
from app.services.processor import (
    DocumentProcessor,
    get_document_status,
    update_document_status,
)

# ==============================================================================
# Helper Functions to Generate Real In-Memory Test Files
# ==============================================================================

def create_sample_pdf(page_texts: list[str]) -> bytes:
    """Creates a real in-memory multi-page PDF using PyMuPDF."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text(fitz.Point(50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_scanned_pdf(num_pages: int = 2) -> bytes:
    """Creates a PDF with pages that have no extractable text (simulating scanned images)."""
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        # Draw a rectangle / graphic without text to simulate an image/scanned page
        page.draw_rect(fitz.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def create_sample_docx(paragraphs: list[str], table_data: list[list[str]] = None) -> bytes:
    """Creates a real in-memory DOCX file using python-docx."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_data:
        table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
        for r_idx, row in enumerate(table_data):
            for c_idx, val in enumerate(row):
                table.cell(r_idx, c_idx).text = val
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()

# ==============================================================================
# 1. Normalization Tests
# ==============================================================================

def test_normalize_text_removes_excessive_blank_lines():
    raw = "Paragraph 1.\n\n\n\n\nParagraph 2 with extra spaces.    \n\n\nParagraph 3."
    cleaned = normalize_text(raw)
    assert cleaned == "Paragraph 1.\n\nParagraph 2 with extra spaces.\n\nParagraph 3."

def test_normalize_text_handles_special_characters_and_whitespace():
    raw = "\t\tHello   world!  \r\n\r\nSecond line with\u200b zero-width space.\t\n"
    cleaned = normalize_text(raw)
    assert "Hello world!" in cleaned
    assert "Second line with zero-width space." in cleaned
    assert "\r" not in cleaned

# ==============================================================================
# 2. PDF Extraction Tests (Preserves Page Numbers)
# ==============================================================================

def test_pdf_extraction_page_by_page():
    page_1 = "This is the content on page one of the architectural specification."
    page_2 = "This is the content on page two discussing database vector indexes."
    page_3 = "This is the content on page three detailing FastAPI endpoints."

    pdf_bytes = create_sample_pdf([page_1, page_2, page_3])
    pages = extract_pdf(pdf_bytes)

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert "page one" in pages[0].text
    assert pages[1].page_number == 2
    assert "page two" in pages[1].text
    assert pages[2].page_number == 3
    assert "page three" in pages[2].text

def test_scanned_pdf_detection_raises_ocr_required():
    """Verifies that a scanned/image-only PDF raises ScannedPDFException."""
    scanned_bytes = create_scanned_pdf(num_pages=2)
    with pytest.raises(ScannedPDFException) as exc_info:
        extract_pdf(scanned_bytes)
    assert "Scanned PDF detected" in str(exc_info.value)
    assert "requires OCR" in str(exc_info.value)

# ==============================================================================
# 3. DOCX Extraction Tests
# ==============================================================================

def test_docx_extraction_paragraphs_and_tables():
    paragraphs = [
        "First paragraph of the project requirements.",
        "Second paragraph covering security and isolation."
    ]
    table = [
        ["Header A", "Header B"],
        ["Row 1 Col 1", "Row 1 Col 2"]
    ]
    docx_bytes = create_sample_docx(paragraphs, table)
    pages = extract_docx(docx_bytes)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "First paragraph of the project requirements." in pages[0].text
    assert "Header A | Header B" in pages[0].text
    assert "Row 1 Col 1 | Row 1 Col 2" in pages[0].text

def test_docx_empty_file_raises_error():
    empty_docx = create_sample_docx([])
    with pytest.raises(EmptyDocumentException):
        extract_docx(empty_docx)

# ==============================================================================
# 4. TXT Extraction Tests
# ==============================================================================

def test_txt_extraction_utf8():
    content = "Sample knowledge assistant notes in plain text.\n\nNext section."
    pages = extract_txt(content.encode("utf-8"))
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Sample knowledge assistant notes" in pages[0].text

def test_txt_extraction_latin1_encoding():
    content = "Document with accented characters: café, résumé, naïve."
    pages = extract_txt(content.encode("latin-1"))
    assert len(pages) == 1
    assert "café" in pages[0].text or "caf" in pages[0].text

def test_txt_empty_file_raises_error():
    with pytest.raises(EmptyDocumentException):
        extract_txt(b"   \n\n  \t ")

# ==============================================================================
# 5. Chunking Tests (Target ~800 tokens, ~150 overlap, Sentence Preservation)
# ==============================================================================

def test_chunking_small_document_produces_single_chunk():
    doc_id = uuid4()
    pages = [ExtractedPage(page_number=1, text="This is a short document. It fits entirely into one chunk.")]
    chunks = chunk_document(pages, document_id=doc_id, target_tokens=800, overlap_tokens=150)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == doc_id
    assert chunks[0].page_number == 1
    assert "short document" in chunks[0].content

def test_chunking_large_document_with_sentence_preservation_and_overlap():
    doc_id = uuid4()
    # Build multiple paragraphs with distinct sentences
    sentence = "Artificial intelligence systems rely on accurate retrieval mechanisms to avoid hallucinations. "
    para_1 = sentence * 25  # ~500 tokens
    para_2 = sentence * 25  # ~500 tokens
    para_3 = sentence * 25  # ~500 tokens

    pages = [
        ExtractedPage(page_number=1, text=para_1),
        ExtractedPage(page_number=2, text=para_2 + "\n\n" + para_3),
    ]

    chunks = chunk_document(pages, document_id=doc_id, target_tokens=800, overlap_tokens=150)

    assert len(chunks) > 1
    # Check indexes are sequential
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.document_id == doc_id
        assert c.estimated_tokens <= 850  # Around target tokens
        assert c.page_number in (1, 2)
        # Avoid cutting sentences mid-way: chunk should end with punctuation or word
        assert c.content.endswith(".") or c.content.endswith("!") or c.content.endswith("?")

# ==============================================================================
# 6. Document Processor Pipeline & Status Tracking
# ==============================================================================

def test_processor_pipeline_success_flow():
    processor = DocumentProcessor(target_chunk_tokens=800, chunk_overlap_tokens=150)
    user_id = uuid4()
    doc_id = uuid4()

    pdf_bytes = create_sample_pdf([
        "Page 1: Introduction to Personal AI Knowledge Assistant.",
        "Page 2: Architecture of the FastAPI backend and pgvector database."
    ])

    response = processor.process_file_content(
        file_bytes=pdf_bytes,
        file_type="pdf",
        file_name="intro.pdf",
        document_id=doc_id,
        user_id=user_id,
    )

    assert response.status == "processed"
    assert response.total_pages == 2
    assert response.total_chunks >= 1
    assert response.sample_extracted_text is not None
    assert response.error_message is None

    # Check status registry
    status_entry = get_document_status(doc_id, user_id)
    assert status_entry is not None
    assert status_entry["status"] == "processed"

def test_processor_pipeline_scanned_pdf_failure():
    processor = DocumentProcessor()
    user_id = uuid4()
    doc_id = uuid4()

    scanned_bytes = create_scanned_pdf(num_pages=2)
    response = processor.process_file_content(
        file_bytes=scanned_bytes,
        file_type="pdf",
        file_name="scanned_receipt.pdf",
        document_id=doc_id,
        user_id=user_id,
    )

    assert response.status == "failed"
    assert response.ocr_required is True
    assert "OCR" in response.error_message

# ==============================================================================
# 7. User Account Isolation & Ownership Security
# ==============================================================================

def test_user_cannot_access_another_users_document_status():
    user_owner = uuid4()
    user_attacker = uuid4()
    doc_id = uuid4()

    update_document_status(
        document_id=doc_id,
        user_id=user_owner,
        status="processed"
    )

    # Owner can access
    assert get_document_status(doc_id, user_owner) is not None

    # Attacker cannot access (returns None)
    assert get_document_status(doc_id, user_attacker) is None

from fastapi import UploadFile, HTTPException
from app.routes.documents import process_document_file, get_status

# ==============================================================================
# 8. FastAPI Endpoint Direct Tests
# ==============================================================================

def test_api_process_file_endpoint():
    user_id = str(uuid4())
    doc_id = str(uuid4())

    text_content = b"This is a sample document for testing the FastAPI document processing endpoint.\n\nSection 2 details."
    upload_file = UploadFile(file=io.BytesIO(text_content), filename="test_doc.txt")

    response = process_document_file(
        file=upload_file,
        document_id=doc_id,
        x_user_id=user_id,
    )

    assert response.status == "processed"
    assert response.file_type == "txt"
    assert response.total_chunks == 1
    assert response.total_pages == 1

def test_api_status_endpoint_enforces_auth():
    user_id = str(uuid4())
    doc_id = str(uuid4())

    # Direct query without header should raise HTTPException 401
    with pytest.raises(HTTPException) as exc_401:
        get_status(document_id=UUID(doc_id), x_user_id=None)
    assert exc_401.value.status_code == 401

    # Processing document first
    upload_file = UploadFile(file=io.BytesIO(b"Notes text content for status checking."), filename="notes.txt")
    process_document_file(
        file=upload_file,
        document_id=doc_id,
        x_user_id=user_id,
    )

    # Query with matching user ID
    res_auth = get_status(document_id=UUID(doc_id), x_user_id=user_id)
    assert res_auth["status"] == "processed"

    # Query with different user ID (should raise HTTPException 404 / unauthorized)
    other_user = str(uuid4())
    with pytest.raises(HTTPException) as exc_404:
        get_status(document_id=UUID(doc_id), x_user_id=other_user)
    assert exc_404.value.status_code == 404
