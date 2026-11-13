"""Text normalization and sanitization utilities for document processing."""
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

def remove_control_characters(text: str) -> str:
    """Removes non-printable control characters while preserving standard whitespace."""
    # Keep newline (\n), carriage return (\r), and tab (\t)
    return "".join(
        ch for ch in text 
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\r", "\t")
    )

def normalize_text(text: str) -> str:
    """
    Normalizes extracted document text:
    - Safely decodes / handles invalid characters.
    - Normalizes Unicode to NFKC form.
    - Cleans up carriage returns and line endings.
    - Collapses repetitive inline horizontal spaces and tabs.
    - Trims excessive blank lines (more than 2 consecutive newlines) to preserve
      clean paragraph boundaries.
    - Strips trailing/leading whitespace.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization (replaces ligature characters, full-width forms, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Filter unprintable control characters
    text = remove_control_characters(text)

    # 3. Standardize line endings (\r\n and \r -> \n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Remove zero-width characters and unusual spaces
    text = re.sub(r"[\u200B-\u200D\uFEFF\u00A0]", " ", text)

    # 5. Collapse consecutive horizontal whitespace (spaces & tabs) on each line
    text = re.sub(r"[ \t]+", " ", text)

    # 6. Remove whitespace immediately before and after newlines
    text = re.sub(r" ?\n ?", "\n", text)

    # 7. Collapse 3 or more consecutive newlines into 2 (preserving clean paragraph boundaries)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. Strip overall leading and trailing whitespace
    return text.strip()

