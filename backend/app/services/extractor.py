"""Document extraction service for PDF, DOCX, and TXT files using PyMuPDF and python-docx."""
import io
import logging
from typing import List, Optional
from dataclasses import dataclass
import pymupdf as fitz  # PyMuPDF
import docx  # python-docx

from .normalizer import normalize_text

logger = logging.getLogger(__name__)

class DocumentExtractionError(Exception):
    """Base exception for document extraction failures."""
    pass

class ScannedPDFException(DocumentExtractionError):
    """Raised when a PDF contains scanned images or lacks extractable text, requiring OCR."""
    pass

class EmptyDocumentException(DocumentExtractionError):
    """Raised when a document is completely empty or contains no readable text."""
    pass

class UnsupportedFormatException(DocumentExtractionError):
    """Raised when the document format is not supported."""
    pass

@dataclass
class ExtractedPage:
    """Represents text extracted from a single page or logical section of a document."""
    page_number: int
    text: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)

def extract_pdf(file_bytes: bytes, min_chars_per_doc: int = 20) -> List[ExtractedPage]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF (fitz).
    Preserves page numbers (1-indexed).
    Detects scanned/image-only PDFs that contain insufficient extractable text and
    reports that OCR is required.
    """
    if not file_bytes:
        raise EmptyDocumentException("PDF file is empty (0 bytes).")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise DocumentExtractionError(f"Failed to open PDF document: {str(e)}") from e

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise EmptyDocumentException("PDF document has 0 pages.")

        pages: List[ExtractedPage] = []
        total_chars = 0
        has_images = False

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-indexed page number

            # Extract raw text from page
            raw_text = page.get_text("text") or ""
            cleaned = normalize_text(raw_text)

            # Check if page has images
            if page.get_images():
                has_images = True

            total_chars += len(cleaned)
            pages.append(ExtractedPage(page_number=page_num, text=cleaned))

        # Check for scanned PDF condition:
        # If total extracted text is below the minimum threshold across pages
        # (especially when images are present), it is a scanned document needing OCR.
        if total_chars < min_chars_per_doc:
            if has_images or total_chars == 0:
                raise ScannedPDFException(
                    f"Scanned PDF detected ({total_pages} page{'s' if total_pages > 1 else ''}): "
                    "The document contains image-only or non-selectable pages and requires OCR processing to extract text."
                )
            else:
                raise EmptyDocumentException("PDF document does not contain any readable text.")

        return pages

    finally:
        doc.close()

def extract_docx(file_bytes: bytes) -> List[ExtractedPage]:
    """
    Extracts text from a DOCX file using python-docx.
    Extracts all paragraphs and table contents, normalized and structured.
    """
    if not file_bytes:
        raise EmptyDocumentException("DOCX file is empty (0 bytes).")

    try:
        stream = io.BytesIO(file_bytes)
        doc = docx.Document(stream)
    except Exception as e:
        raise DocumentExtractionError(f"Failed to open DOCX file: {str(e)}") from e

    elements: List[str] = []

    # 1. Extract paragraphs
    for para in doc.paragraphs:
        p_text = para.text.strip()
        if p_text:
            elements.append(p_text)

    # 2. Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                elements.append(" | ".join(row_cells))

    combined_text = "\n\n".join(elements)
    cleaned = normalize_text(combined_text)

    if not cleaned:
        raise EmptyDocumentException("DOCX document contains no readable text.")

    # In DOCX without explicit page layout rendering, default to logical page 1
    return [ExtractedPage(page_number=1, text=cleaned)]

def extract_txt(file_bytes: bytes) -> List[ExtractedPage]:
    """
    Reads plain text files using standard library with automatic fallback encoding detection
    (UTF-8, UTF-8-SIG, Latin-1, CP1252) and safe error handling.
    """
    if not file_bytes:
        raise EmptyDocumentException("TXT file is empty (0 bytes).")

    # Try common encodings in order
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    decoded_text: Optional[str] = None

    for enc in encodings_to_try:
        try:
            decoded_text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        # Safe fallback with replacement of corrupted characters
        decoded_text = file_bytes.decode("utf-8", errors="replace")

    cleaned = normalize_text(decoded_text)
    if not cleaned:
        raise EmptyDocumentException("TXT document contains no readable text.")

    return [ExtractedPage(page_number=1, text=cleaned)]

def extract_document(
    file_bytes: bytes, 
    file_type: str, 
    file_name: str = ""
) -> List[ExtractedPage]:
    """
    Unified extraction dispatcher based on file type / extension.
    Supported types: 'pdf', 'docx', 'txt'.
    """
    ft = (file_type or "").lower().strip().replace(".", "")
    if not ft and file_name:
        parts = file_name.split(".")
        if len(parts) > 1:
            ft = parts[-1].lower()

    logger.info(f"Extracting document (type={ft}, size={len(file_bytes)} bytes)")

    if ft == "pdf":
        return extract_pdf(file_bytes)
    elif ft in ("docx", "doc"):
        if ft == "doc":
            raise UnsupportedFormatException(
                "Legacy .doc format is not directly supported. Please convert to modern .docx."
            )
        return extract_docx(file_bytes)
    elif ft in ("txt", "text"):
        return extract_txt(file_bytes)
    else:
        raise UnsupportedFormatException(
            f"Unsupported file format '{ft}'. Supported formats are PDF, DOCX, and TXT."
        )
