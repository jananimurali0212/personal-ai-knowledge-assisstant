"""AI-Powered Document Summarization Service.
Implements hierarchical map-reduce summarization for long documents,
strict grounding to avoid fabricated conclusions, and persistent caching in Supabase.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone
import httpx

from google import genai
from google.genai import types
from google.genai.errors import APIError

from ..config.settings import settings
from ..models.summary import (
    KeyTerminologyItem,
    DocumentSummaryContent,
    DocumentSummaryResponse,
    DocumentSummaryRequest,
)
from .chunk_repository import (
    ChunkRepository,
    default_chunk_repository,
)
from .summary_repository import (
    SummaryRepository,
    default_summary_repository,
)
from .processor import (
    get_document_status,
    _processing_status_registry,
)

logger = logging.getLogger(__name__)

# Max chunks per section for hierarchical summarization
SECTION_CHUNK_SIZE = 3
# Threshold above which hierarchical map-reduce summarization is engaged
HIERARCHICAL_CHUNK_THRESHOLD = 3

SUMMARIZATION_SYSTEM_INSTRUCTION = """You are an expert technical document analyst and summarizer.
Your mission is to generate a comprehensive, structured, and strictly factual summary of the provided document text.

CORE RULES:
1. STRICT GROUNDING: Rely ONLY on facts, definitions, findings, and conclusions directly present in the source text.
2. ZERO FABRICATION: Do NOT extrapolate, speculate, or introduce external knowledge. If an aspect is not in the text, do NOT invent it.
3. STRUCTURED OUTPUT: You MUST return a single, valid JSON object matching the exact schema below, with no surrounding commentary or markdown backticks.

REQUIRED JSON SCHEMA:
{
  "executive_summary": "A cohesive 2 to 4 sentence executive summary encapsulating the document's main thesis, context, and primary findings.",
  "main_topics": [
    "Core topic 1 (title & brief scope)",
    "Core topic 2 (title & brief scope)"
  ],
  "important_points": [
    "Essential factual insight or key takeaway 1",
    "Essential factual insight or key takeaway 2",
    "Essential factual insight or key takeaway 3"
  ],
  "key_terminology": [
    {
      "term": "Term or Identifier",
      "definition": "Exact definition or usage as established in the source content."
    }
  ],
  "important_conclusions": [
    "Primary conclusion or final outcome 1",
    "Primary conclusion or final outcome 2"
  ],
  "action_items": [
    "Concrete action item or implementation takeaway 1",
    "Concrete action item or implementation takeaway 2"
  ]
}
"""

SECTION_MAP_SYSTEM_INSTRUCTION = """You are an expert document section analyst.
Analyze the provided section of a larger document. Extract the section's core topics, essential factual details, specialized terms, and intermediate conclusions.
Return a valid JSON object ONLY:
{
  "section_overview": "Brief summary of this section",
  "key_points": ["Point 1", "Point 2"],
  "terminology": [{"term": "Term", "definition": "Definition"}],
  "conclusions": ["Conclusion 1"]
}
"""


class SummarizationError(Exception):
    """Base exception for summarization failures."""
    pass


class DocumentNotFoundError(SummarizationError):
    """Raised when document is not found or unauthorized."""
    pass


class EmptyDocumentError(SummarizationError):
    """Raised when document contains no extracted text or chunks."""
    pass


class DocumentSummarizerService:
    """
    Service responsible for orchestrating document summarization:
    - Verifies document ownership.
    - Retrieves document chunks from Supabase/in-memory store.
    - Decides between single-pass and hierarchical map-reduce summarization.
    - Invokes Gemini Gen AI SDK with structured JSON prompts.
    - Validates output structure.
    - Caches and persists summaries to prevent redundant LLM invocations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        chunk_repository: Optional[ChunkRepository] = None,
        summary_repository: Optional[SummaryRepository] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = self._clean_model_name(
            model_name or settings.GEMINI_GENERATION_MODEL or "gemini-2.5-flash"
        )
        self.chunk_repository = chunk_repository or default_chunk_repository
        self.summary_repository = summary_repository or default_summary_repository
        self.timeout = timeout
        self._client: Optional[genai.Client] = None

    @staticmethod
    def _clean_model_name(name: str) -> str:
        if name.startswith("models/"):
            return name[len("models/"):]
        return name

    def _get_client(self) -> genai.Client:
        """Initializes or returns cached Google Gen AI Client."""
        if self._client is None:
            if not self.api_key or self.api_key == "your_gemini_api_key_here":
                raise SummarizationError(
                    "GEMINI_API_KEY is not configured. Please provide a valid Gemini API key."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _get_document_metadata(self, document_id: UUID, user_id: UUID) -> Dict[str, Any]:
        """
        Retrieves document metadata and validates user ownership.
        Checks in-memory status registry, and falls back to Supabase documents table.
        """
        key = str(document_id)
        # Check in-memory status registry first
        if key in _processing_status_registry:
            entry = _processing_status_registry[key]
            if str(entry.get("user_id")) != str(user_id):
                logger.warning(
                    f"Unauthorized access attempt to document {document_id} by user {user_id}"
                )
                raise DocumentNotFoundError(
                    f"Unauthorized: Document {document_id} does not belong to user {user_id}"
                )
            return {
                "id": str(document_id),
                "user_id": str(user_id),
                "file_name": entry.get("file_name", "document"),
                "file_type": entry.get("file_type", "txt"),
                "status": entry.get("status", "ready"),
            }

        # Check Supabase documents table if configured
        supabase_url = settings.SUPABASE_URL
        api_key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
        if supabase_url and api_key and api_key not in ("your_supabase_anon_key_here", ""):
            try:
                endpoint = f"{supabase_url.rstrip('/')}/rest/v1/documents"
                headers = {
                    "apikey": api_key,
                    "Authorization": f"Bearer {api_key}",
                }
                params = {
                    "id": f"eq.{document_id}",
                    "user_id": f"eq.{user_id}",
                    "select": "id,user_id,file_name,file_type,status,summary,summary_status",
                }
                with httpx.Client(timeout=10.0) as client:
                    res = client.get(endpoint, headers=headers, params=params)
                    if res.status_code == 200:
                        rows = res.json()
                        if rows and len(rows) > 0:
                            return rows[0]
            except Exception as e:
                logger.warning(f"Error fetching document metadata from Supabase: {e}")

        # If we got here, document is not found or unauthorized
        raise DocumentNotFoundError(
            f"Document {document_id} not found or you do not have permission to access it."
        )

    def _clean_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Strips markdown fences and extracts valid JSON object."""
        cleaned = raw_text.strip()
        # Remove markdown code block if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: extract substring between first { and last }
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                sub = cleaned[first_brace : last_brace + 1]
                try:
                    return json.loads(sub)
                except json.JSONDecodeError:
                    pass

        # If JSON parsing completely fails, return safe fallback structure
        logger.warning(f"Could not parse valid JSON from Gemini output: {raw_text[:200]}...")
        return {
            "executive_summary": raw_text[:500].strip(),
            "main_topics": ["General Document Content"],
            "important_points": [raw_text[:300].strip()],
            "key_terminology": [],
            "important_conclusions": [],
            "action_items": [],
        }

    def _call_gemini_json(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        """Invokes Gemini API and returns parsed JSON response."""
        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=3000,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=int(self.timeout)),
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw_text = response.text or ""
            return self._clean_json_response(raw_text)
        except Exception as e:
            logger.error(f"Gemini API summarization call failed: {e}")
            raise SummarizationError(f"AI summarization service error: {str(e)}") from e

    def _summarize_single_pass(
        self,
        chunks: List[Dict[str, Any]],
        file_name: str,
        file_type: str,
    ) -> DocumentSummaryContent:
        """Executes direct single-pass summarization for short/medium documents."""
        context_parts = []
        for c in chunks:
            idx = c.get("chunk_index", 0)
            page = c.get("page_number")
            page_info = f" (Page {page})" if page else ""
            content = c.get("content", "").strip()
            context_parts.append(f"--- Chunk {idx + 1}{page_info} ---\n{content}")

        full_context = "\n\n".join(context_parts)
        prompt = (
            f"DOCUMENT TITLE: {file_name} (Format: {file_type.upper()})\n\n"
            f"DOCUMENT CONTENT:\n{full_context}\n\n"
            "Produce the complete structured JSON summary following the required schema."
        )

        data = self._call_gemini_json(prompt, SUMMARIZATION_SYSTEM_INSTRUCTION)
        return self._format_summary_content(
            data=data,
            total_sections=1,
            is_hierarchical=False,
        )

    def _summarize_hierarchical(
        self,
        chunks: List[Dict[str, Any]],
        file_name: str,
        file_type: str,
    ) -> DocumentSummaryContent:
        """
        Executes hierarchical map-reduce summarization for long documents:
        1. Map Phase: Split chunks into sections of 2-3 chunks; summarize each section.
        2. Reduce Phase: Aggregate section summaries.
        3. Final Synthesis Phase: Synthesize unified structured document summary.
        """
        total_chunks = len(chunks)
        section_briefs: List[Dict[str, Any]] = []

        # Split into sections
        sections = [
            chunks[i : i + SECTION_CHUNK_SIZE]
            for i in range(0, total_chunks, SECTION_CHUNK_SIZE)
        ]
        logger.info(
            f"Long document '{file_name}' ({total_chunks} chunks): partitioned into "
            f"{len(sections)} sections for hierarchical summarization."
        )

        # Stage 1: Map Phase (summarize each section)
        for s_idx, sec_chunks in enumerate(sections, 1):
            sec_context_parts = []
            pages_in_sec = set()
            for c in sec_chunks:
                c_idx = c.get("chunk_index", 0)
                page = c.get("page_number")
                if page:
                    pages_in_sec.add(page)
                page_str = f" (Page {page})" if page else ""
                sec_context_parts.append(f"Chunk {c_idx + 1}{page_str}:\n{c.get('content', '')}")

            pages_str = f"Pages: {', '.join(str(p) for p in sorted(pages_in_sec))}" if pages_in_sec else ""
            sec_prompt = (
                f"DOCUMENT: {file_name}\n"
                f"SECTION {s_idx} OF {len(sections)} ({pages_str}):\n\n"
                + "\n\n".join(sec_context_parts)
                + "\n\nExtract key points, terminology, and intermediate conclusions from this section."
            )
            brief = self._call_gemini_json(sec_prompt, SECTION_MAP_SYSTEM_INSTRUCTION)
            section_briefs.append(brief)

        # Stage 2 & 3: Reduce and Final Synthesis Phase
        combined_briefs_text = []
        for i, b in enumerate(section_briefs, 1):
            combined_briefs_text.append(
                f"--- SECTION {i} BRIEF ---\n"
                f"Overview: {b.get('section_overview', '')}\n"
                f"Key Points: {json.dumps(b.get('key_points', []))}\n"
                f"Terminology: {json.dumps(b.get('terminology', []))}\n"
                f"Conclusions: {json.dumps(b.get('conclusions', []))}\n"
            )

        reduce_prompt = (
            f"DOCUMENT TITLE: {file_name} (Format: {file_type.upper()})\n"
            f"TOTAL SECTIONS ANALYZED: {len(sections)} ({total_chunks} total chunks)\n\n"
            f"CONSOLIDATED SECTION SUMMARIES:\n"
            + "\n".join(combined_briefs_text)
            + "\n\nSynthesize all sections above into the final unified structured JSON summary conforming to the schema."
        )

        final_data = self._call_gemini_json(reduce_prompt, SUMMARIZATION_SYSTEM_INSTRUCTION)
        return self._format_summary_content(
            data=final_data,
            total_sections=len(sections),
            is_hierarchical=True,
        )

    def _format_summary_content(
        self,
        data: Dict[str, Any],
        total_sections: int,
        is_hierarchical: bool,
    ) -> DocumentSummaryContent:
        """Parses and sanitizes dictionary into a validated DocumentSummaryContent model."""
        exec_summary = str(data.get("executive_summary", "")).strip()
        if not exec_summary:
            exec_summary = "Document summary successfully extracted from verified source content."

        raw_topics = data.get("main_topics", [])
        main_topics = [str(t).strip() for t in raw_topics if t and str(t).strip()]

        raw_points = data.get("important_points", [])
        important_points = [str(p).strip() for p in raw_points if p and str(p).strip()]

        raw_terms = data.get("key_terminology", [])
        key_terminology: List[KeyTerminologyItem] = []
        for item in raw_terms:
            if isinstance(item, dict) and "term" in item:
                term_val = str(item.get("term", "")).strip()
                def_val = str(item.get("definition", "")).strip()
                if term_val:
                    key_terminology.append(
                        KeyTerminologyItem(term=term_val, definition=def_val or "Defined in context.")
                    )
            elif isinstance(item, str) and item.strip():
                # Split on colon if string like "Term: Definition"
                parts = item.split(":", 1)
                if len(parts) == 2:
                    key_terminology.append(
                        KeyTerminologyItem(term=parts[0].strip(), definition=parts[1].strip())
                    )
                else:
                    key_terminology.append(
                        KeyTerminologyItem(term=item.strip(), definition="Referenced in document context.")
                    )

        raw_conclusions = data.get("important_conclusions", [])
        important_conclusions = [str(c).strip() for c in raw_conclusions if c and str(c).strip()]

        raw_actions = data.get("action_items", [])
        action_items = [str(a).strip() for a in raw_actions if a and str(a).strip()]

        return DocumentSummaryContent(
            executive_summary=exec_summary,
            main_topics=main_topics,
            important_points=important_points,
            key_terminology=key_terminology,
            important_conclusions=important_conclusions,
            action_items=action_items,
            total_sections_analyzed=total_sections,
            is_hierarchical=is_hierarchical,
            model_used=self.model,
            generated_at=datetime.now(timezone.utc),
        )

    def summarize_document(
        self,
        document_id: UUID,
        user_id: UUID,
        request: Optional[DocumentSummaryRequest] = None,
    ) -> DocumentSummaryResponse:
        """
        Coordinates the end-to-end summarization for a specific document:
        1. Verifies document ownership for authenticated user.
        2. Checks for cached summary if force_regenerate is False.
        3. Retrieves document chunks from storage.
        4. Validates chunk presence (rejects empty documents).
        5. Selects single-pass or hierarchical map-reduce mode based on document length.
        6. Generates structured summary.
        7. Persists summary to Supabase and updates document status.
        8. Returns formatted DocumentSummaryResponse.
        """
        force_regen = request.force_regenerate if request else False

        # Step 1: Verify Ownership & Retrieve Document Metadata
        doc_meta = self._get_document_metadata(document_id, user_id)
        file_name = doc_meta.get("file_name", "document")
        file_type = doc_meta.get("file_type", "txt")

        # Step 2: Check for Cached Summary (avoid redundant generation)
        if not force_regen:
            existing_summary_record = self.summary_repository.get_summary(document_id, user_id)
            if existing_summary_record:
                logger.info(
                    f"Found valid cached summary for doc={document_id}, user={user_id}. Returning cached result."
                )
                summary_content = self._format_summary_content(
                    data=existing_summary_record,
                    total_sections=existing_summary_record.get("total_sections_analyzed", 1),
                    is_hierarchical=existing_summary_record.get("hierarchical", False),
                )
                return DocumentSummaryResponse(
                    document_id=document_id,
                    user_id=user_id,
                    file_name=file_name,
                    file_type=file_type,
                    status="completed",
                    summary=summary_content,
                    cached=True,
                    created_at=existing_summary_record.get("created_at"),
                )

        # Step 3: Retrieve Chunks
        chunks = self.chunk_repository.get_chunks_by_document(document_id, user_id)
        if not chunks:
            logger.warning(
                f"Document {document_id} has 0 chunks stored. Cannot summarize empty document."
            )
            self.summary_repository.update_summary_status(
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error_message="Document has no extracted text chunks.",
            )
            raise EmptyDocumentError(
                f"Document '{file_name}' contains no extracted text or chunks. Please ensure the document is processed before summarizing."
            )

        logger.info(
            f"Summarizing document '{file_name}' (doc_id={document_id}, chunks={len(chunks)}) "
            f"for user={user_id} using model={self.model}"
        )

        # Update status to 'generating'
        self.summary_repository.update_summary_status(
            document_id=document_id,
            user_id=user_id,
            status="generating",
        )

        try:
            # Step 4: Dispatch Summarization Strategy
            if len(chunks) <= HIERARCHICAL_CHUNK_THRESHOLD:
                summary_content = self._summarize_single_pass(
                    chunks=chunks,
                    file_name=file_name,
                    file_type=file_type,
                )
            else:
                summary_content = self._summarize_hierarchical(
                    chunks=chunks,
                    file_name=file_name,
                    file_type=file_type,
                )

            # Step 5: Persist Summary in Supabase & Update Document Status
            serialized_summary = summary_content.model_dump(mode="json")
            self.summary_repository.save_summary(
                document_id=document_id,
                user_id=user_id,
                summary_data=serialized_summary,
            )

            logger.info(
                f"Successfully generated and stored summary for document {document_id} "
                f"(hierarchical={summary_content.is_hierarchical})"
            )

            return DocumentSummaryResponse(
                document_id=document_id,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                status="completed",
                summary=summary_content,
                cached=False,
                created_at=summary_content.generated_at,
            )

        except Exception as e:
            logger.error(f"Failed to generate summary for document {document_id}: {e}")
            self.summary_repository.update_summary_status(
                document_id=document_id,
                user_id=user_id,
                status="failed",
                error_message=str(e),
            )
            raise


default_summarizer_service = DocumentSummarizerService()
