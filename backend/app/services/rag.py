"""Dedicated RAG Orchestration Service.
Integrates semantic vector retrieval (Supabase pgvector) with Gemini answer generation.
Enforces multi-tenant user isolation, chunk deduplication, threshold filtering, and grounding.
"""
import logging
import json
import re
import inspect
from typing import List, Optional, Set, Tuple, Dict, Any
from uuid import UUID

from .search import (
    SemanticSearchService,
    default_semantic_search_service,
    SemanticSearchError,
)
from .hybrid_search import (
    HybridSearchService,
    default_hybrid_search_service,
    HybridSearchError,
)
from .reranker import (
    ReRankingService,
    default_reranker_service,
)
from .generation import (
    GeminiGenerationService,
    default_generation_service,
    GenerationError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    GeminiAPIKeyMissingError,
)
from .conversation_repository import (
    ConversationRepository,
    default_conversation_repository,
)
from ..models.conversation import generate_conversation_title
from ..models.search import (
    SemanticSearchRequest,
    SearchResultItem,
    HybridSearchRequest,
)
from ..models.generation import ContextChunk
from ..models.chat import (
    ChatAskRequest,
    ChatAskResponse,
    ChatSourceItem,
    ChatRetrievalMetadata,
)

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The provided document context does not contain sufficient information to answer this question."
)


class RAGOrchestrationError(Exception):
    """Base exception for RAG pipeline failures."""
    pass


class RAGOrchestratorService:
    """
    Dedicated RAG Orchestration Service.
    Coordinates the complete lifecycle of a Retrieval-Augmented Generation query:
    1. Validates authenticated user ownership.
    2. Persists conversations and chat messages in Supabase with RLS.
    3. Handles contextual follow-up questions using limited conversation history.
    4. Embeds the query via Gemini embeddings (task_type=RETRIEVAL_QUERY).
    5. Retrieves top-k relevant chunks from pgvector with cosine similarity and threshold cutoff.
    6. Enforces strict multi-tenant user isolation (users can NEVER retrieve another user's chunks).
    7. Deduplicates retrieved chunks and formats structured context.
    8. Prompts Gemini with strict grounding, citations, and zero-hallucination rules.
    9. Assembles complete answer with verified citations, chunk IDs, and retrieval metadata.
    """

    def __init__(
        self,
        search_service: Optional[SemanticSearchService] = None,
        generation_service: Optional[GeminiGenerationService] = None,
        conversation_repository: Optional[ConversationRepository] = None,
        hybrid_search_service: Optional[Any] = None,
        use_hybrid_search: bool = True,
        reranker_service: Optional[Any] = None,
        use_reranker: bool = True,
    ):
        self.search_service = search_service or default_semantic_search_service
        self.generation_service = generation_service or default_generation_service
        self.conversation_repository = conversation_repository or default_conversation_repository
        if search_service is not None and hybrid_search_service is None:
            self.use_hybrid_search = False
            self.hybrid_search_service = None
        else:
            self.use_hybrid_search = use_hybrid_search
            self.hybrid_search_service = hybrid_search_service or default_hybrid_search_service
        self.use_reranker = use_reranker
        self.reranker_service = reranker_service or default_reranker_service

    def ask(self, request: ChatAskRequest, user_id: UUID) -> ChatAskResponse:
        """
        Executes end-to-end RAG question answering for the authenticated user.
        
        Parameters:
        - request: ChatAskRequest containing question, optional document_ids, conversation_id, threshold, limit.
        - user_id: Verified UUID of the authenticated user (derived securely from auth credentials).
        
        Returns:
        - ChatAskResponse with answer, citations, document names, page numbers, and telemetry.
        """
        clean_question = request.question.strip()
        logger.info(
            f"Executing RAG pipeline for user={user_id}: question='{clean_question[:60]}...' "
            f"(threshold={request.match_threshold}, limit={request.limit}, "
            f"doc_filters={len(request.document_ids) if request.document_ids else 'all'})"
        )

        # ----------------------------------------------------------------------
        # Step 1: Manage Persistent Conversation & Contextual History
        # ----------------------------------------------------------------------
        conv_id: Optional[UUID] = None
        if request.conversation_id:
            try:
                conv_id = UUID(str(request.conversation_id))
            except (ValueError, TypeError):
                conv_id = None

        existing_conv = None
        recent_history: List[Dict[str, str]] = []

        if conv_id:
            existing_conv = self.conversation_repository.get_conversation(conv_id, user_id)
            if existing_conv:
                recent_history = self.conversation_repository.get_recent_history(
                    conv_id, user_id, limit=6
                )

        if not existing_conv:
            title = generate_conversation_title(clean_question)
            new_conv = self.conversation_repository.create_conversation(
                user_id=user_id,
                title=title,
                conversation_id=conv_id,
            )
            conv_id = new_conv.id
            existing_conv = new_conv

        # Save user message to persistent conversation store
        self.conversation_repository.add_message(
            conversation_id=conv_id,
            user_id=user_id,
            role="user",
            content=clean_question,
        )

        # Contextual Query Expansion for Follow-up questions
        retrieval_query = clean_question
        if recent_history:
            last_user_turns = [h["content"] for h in recent_history if h.get("role") == "user"]
            if last_user_turns:
                last_user_query = last_user_turns[-1]
                followup_triggers = [
                    "that", "it", "its", "example", "simpler", "previous",
                    "explain", "compare", "advantage", "advantages", "benefit",
                    "benefits", "more", "why", "how", "what else", "tell me more"
                ]
                is_followup = (
                    any(re.search(rf"\b{re.escape(w)}\b", clean_question, re.IGNORECASE) for w in followup_triggers)
                    or len(clean_question.split()) <= 4
                )
                if is_followup:
                    retrieval_query = f"{last_user_query} {clean_question}"
                    logger.info(f"Contextual follow-up detected. Expanded retrieval query to: '{retrieval_query[:80]}...'")

        # ----------------------------------------------------------------------
        # Step 2: Document Retrieval via Hybrid or Semantic Search Service
        # ----------------------------------------------------------------------
        is_rerank_active = (
            self.use_reranker
            and self.reranker_service is not None
            and getattr(self.reranker_service, "enabled", True)
        )
        fetch_limit = (
            min(max(getattr(self.reranker_service, "candidate_limit", 20), request.limit * 3), 50)
            if is_rerank_active
            else request.limit
        )

        try:
            if self.use_hybrid_search and self.hybrid_search_service is not None:
                hybrid_req = HybridSearchRequest(
                    query=retrieval_query,
                    document_ids=request.document_ids,
                    match_threshold=request.match_threshold,
                    limit=fetch_limit,
                    rrf_k=60,
                    search_mode="hybrid",
                )
                search_response = self.hybrid_search_service.search_hybrid(
                    request=hybrid_req,
                    user_id=user_id,
                )
            else:
                search_request = SemanticSearchRequest(
                    query=retrieval_query,
                    document_ids=request.document_ids,
                    match_threshold=request.match_threshold,
                    limit=fetch_limit,
                )
                search_response = self.search_service.search(
                    request=search_request,
                    user_id=user_id,
                )
        except (SemanticSearchError, HybridSearchError) as sse:
            logger.error(f"Search failed during RAG execution: {sse}")
            raise RAGOrchestrationError(f"Context retrieval failed: {str(sse)}") from sse
        except Exception as e:
            logger.exception(f"Unexpected error during RAG retrieval: {e}")
            raise RAGOrchestrationError(f"Retrieval error: {str(e)}") from e

        retrieved_items: List[SearchResultItem] = search_response.results

        # ----------------------------------------------------------------------
        # Step 2.5: Second-Stage Cross-Encoder Re-Ranking
        # ----------------------------------------------------------------------
        if is_rerank_active and retrieved_items:
            try:
                rerank_response = self.reranker_service.rerank(
                    query=retrieval_query,
                    candidates=retrieved_items,
                    top_k=request.limit,
                )
                retrieved_items = rerank_response.results
            except Exception as e:
                logger.warning(f"Re-ranking failed ({e}). Proceeding with first-stage candidate order.")
                retrieved_items = retrieved_items[:request.limit]

        emb_model = getattr(getattr(self.search_service, "embedding_service", None), "model", None)
        embedding_model_str = emb_model if isinstance(emb_model, str) else "text-embedding-004"
        gen_model = getattr(self.generation_service, "model", None)
        generation_model_str = gen_model if isinstance(gen_model, str) else "gemini-2.5-flash"

        # ----------------------------------------------------------------------
        # Step 3: Handle Empty / Insufficient Retrieval
        # ----------------------------------------------------------------------
        if not retrieved_items:
            logger.info(
                f"No relevant document chunks found above threshold ({request.match_threshold}) "
                f"for query='{clean_question[:50]}...'. Returning insufficient context response."
            )
            # Save assistant insufficient context message to persistent store
            self.conversation_repository.add_message(
                conversation_id=conv_id,
                user_id=user_id,
                role="assistant",
                content=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
            )

            return ChatAskResponse(
                conversation_id=conv_id,
                question=clean_question,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                document_names=[],
                page_numbers=[],
                relevant_chunk_ids=[],
                retrieval_metadata=ChatRetrievalMetadata(
                    total_chunks_retrieved=0,
                    match_threshold=request.match_threshold,
                    embedding_model=embedding_model_str,
                    generation_model=generation_model_str,
                    grounded=False,
                    is_insufficient_context=True,
                    tokens_used=0,
                ),
            )

        # ----------------------------------------------------------------------
        # Step 4: Deduplicate Chunks and Construct Context
        # ----------------------------------------------------------------------
        unique_chunks: List[SearchResultItem] = []
        seen_keys: Set[Tuple[str, int]] = set()

        for item in retrieved_items:
            chunk_key = (str(item.document_id), item.chunk_index)
            if chunk_key not in seen_keys:
                seen_keys.add(chunk_key)
                unique_chunks.append(item)

        # Convert to ContextChunk objects with sequential source identifiers
        context_chunks: List[ContextChunk] = []
        for idx, item in enumerate(unique_chunks, 1):
            is_pdf = item.document_name.lower().endswith(".pdf")
            effective_page = item.page_number if is_pdf else None
            ext = "pdf" if is_pdf else (item.document_name.split(".")[-1].lower() if "." in item.document_name else "docx")
            source_id = f"Source {idx}"

            context_chunks.append(
                ContextChunk(
                    content=item.content,
                    document_name=item.document_name,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    page_number=effective_page,
                    chunk_index=item.chunk_index,
                    similarity=item.similarity,
                    source_id=source_id,
                    file_type=ext,
                )
            )

        logger.info(
            f"Retrieved {len(retrieved_items)} chunks, deduplicated to {len(context_chunks)} unique chunks "
            f"across {len(set(c.document_name for c in context_chunks))} document(s)."
        )

        # ----------------------------------------------------------------------
        # Step 5: Grounded Answer Generation via Gemini with History
        # ----------------------------------------------------------------------
        try:
            gen_response = self.generation_service.generate_answer(
                question=clean_question,
                context=context_chunks,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                history=recent_history,
            )
        except (GeminiRateLimitError, GeminiTimeoutError, GeminiAPIKeyMissingError):
            raise
        except GenerationError as ge:
            logger.error(f"Gemini generation error in RAG pipeline: {ge}")
            raise RAGOrchestrationError(f"Answer generation failed: {str(ge)}") from ge
        except Exception as e:
            logger.exception(f"Unexpected generation error in RAG pipeline: {e}")
            raise RAGOrchestrationError(f"Generation error: {str(e)}") from e

        # ----------------------------------------------------------------------
        # Step 6: Format Source Attributions & Metadata
        # ----------------------------------------------------------------------
        sources: List[ChatSourceItem] = []
        document_names_set = set()
        page_numbers_set = set()
        relevant_chunk_ids: List[UUID] = []

        for idx, item in enumerate(unique_chunks, 1):
            snippet = item.content[:140].replace("\n", " ").strip()
            if len(item.content) > 140:
                snippet += "..."

            is_pdf = item.document_name.lower().endswith(".pdf")
            effective_page = item.page_number if is_pdf else None
            has_page = is_pdf and (item.page_number is not None)
            ext = "pdf" if is_pdf else (item.document_name.split(".")[-1].lower() if "." in item.document_name else "docx")
            source_id = f"Source {idx}"

            sources.append(
                ChatSourceItem(
                    chunk_id=item.chunk_id,
                    chunk_identifier=str(item.chunk_id) if item.chunk_id else f"{item.document_id}:{item.chunk_index}",
                    document_id=item.document_id,
                    document_name=item.document_name,
                    page_number=effective_page,
                    chunk_index=item.chunk_index,
                    similarity=item.similarity,
                    snippet=snippet,
                    relevant_excerpt=snippet,
                    source_id=source_id,
                    has_page_number=has_page,
                    file_type=ext,
                )
            )
            document_names_set.add(item.document_name)
            if effective_page is not None:
                page_numbers_set.add(effective_page)
            if item.chunk_id:
                relevant_chunk_ids.append(item.chunk_id)

        distinct_doc_names = sorted(list(document_names_set))
        distinct_pages = sorted(list(page_numbers_set))

        # ----------------------------------------------------------------------
        # Step 7: Persist Assistant Response in Supabase
        # ----------------------------------------------------------------------
        serialized_sources = [s.model_dump(mode="json") for s in sources]
        self.conversation_repository.add_message(
            conversation_id=conv_id,
            user_id=user_id,
            role="assistant",
            content=gen_response.answer,
            sources=serialized_sources,
        )

        retrieval_meta = ChatRetrievalMetadata(
            total_chunks_retrieved=len(unique_chunks),
            match_threshold=request.match_threshold,
            embedding_model=embedding_model_str,
            generation_model=generation_model_str,
            grounded=gen_response.grounded,
            is_insufficient_context=gen_response.is_insufficient_context,
            tokens_used=gen_response.tokens_used,
        )

        return ChatAskResponse(
            conversation_id=conv_id,
            question=clean_question,
            answer=gen_response.answer,
            sources=sources,
            document_names=distinct_doc_names,
            page_numbers=distinct_pages,
            relevant_chunk_ids=relevant_chunk_ids,
            retrieval_metadata=retrieval_meta,
        )

    def ask_stream(
        self,
        request: ChatAskRequest,
        user_id: UUID,
        raw_request: Optional[Any] = None,
    ):
        """
        Real-time Server-Sent Events (SSE) streaming RAG generator.
        Supports structured events: start, token, sources, complete, error.
        Streams genuine Gemini answer tokens, handles client disconnects,
        and saves message history upon completion.
        """
        clean_question = request.question.strip()
        conv_id: Optional[UUID] = None
        if request.conversation_id:
            try:
                conv_id = UUID(str(request.conversation_id))
            except (ValueError, TypeError):
                conv_id = None

        existing_conv = None
        recent_history: List[Dict[str, str]] = []

        if conv_id:
            existing_conv = self.conversation_repository.get_conversation(conv_id, user_id)
            if existing_conv:
                recent_history = self.conversation_repository.get_recent_history(
                    conv_id, user_id, limit=6
                )

        if not existing_conv:
            title = generate_conversation_title(clean_question)
            new_conv = self.conversation_repository.create_conversation(
                user_id=user_id,
                title=title,
                conversation_id=conv_id,
            )
            conv_id = new_conv.id

        # 1. Yield 'start' event
        yield format_sse("start", {
            "conversation_id": str(conv_id),
            "status": "started",
        })

        # Save user message to persistent conversation store
        self.conversation_repository.add_message(
            conversation_id=conv_id,
            user_id=user_id,
            role="user",
            content=clean_question,
        )

        try:
            # Contextual Query Expansion for Follow-up questions
            retrieval_query = clean_question
            if recent_history:
                last_user_turns = [h["content"] for h in recent_history if h.get("role") == "user"]
                if last_user_turns:
                    last_user_query = last_user_turns[-1]
                    followup_triggers = [
                        "that", "it", "its", "example", "simpler", "previous",
                        "explain", "compare", "advantage", "advantages", "benefit",
                        "benefits", "more", "why", "how", "what else", "tell me more"
                    ]
                    is_followup = (
                        any(re.search(rf"\b{re.escape(w)}\b", clean_question, re.IGNORECASE) for w in followup_triggers)
                        or len(clean_question.split()) <= 4
                    )
                    if is_followup:
                        retrieval_query = f"{last_user_query} {clean_question}"

            # Step 2: Document Retrieval via Hybrid or Semantic Search Service
            is_rerank_active = (
                self.use_reranker
                and self.reranker_service is not None
                and getattr(self.reranker_service, "enabled", True)
            )
            fetch_limit = (
                min(max(getattr(self.reranker_service, "candidate_limit", 20), request.limit * 3), 50)
                if is_rerank_active
                else request.limit
            )

            if self.use_hybrid_search and self.hybrid_search_service is not None:
                hybrid_req = HybridSearchRequest(
                    query=retrieval_query,
                    document_ids=request.document_ids,
                    match_threshold=request.match_threshold,
                    limit=fetch_limit,
                    rrf_k=60,
                    search_mode="hybrid",
                )
                search_response = self.hybrid_search_service.search_hybrid(
                    request=hybrid_req,
                    user_id=user_id,
                )
            else:
                search_request = SemanticSearchRequest(
                    query=retrieval_query,
                    document_ids=request.document_ids,
                    match_threshold=request.match_threshold,
                    limit=fetch_limit,
                )
                search_response = self.search_service.search(
                    request=search_request,
                    user_id=user_id,
                )
            retrieved_items: List[SearchResultItem] = search_response.results

            # Step 2.5: Second-Stage Cross-Encoder Re-Ranking
            if is_rerank_active and retrieved_items:
                try:
                    rerank_response = self.reranker_service.rerank(
                        query=retrieval_query,
                        candidates=retrieved_items,
                        top_k=request.limit,
                    )
                    retrieved_items = rerank_response.results
                except Exception as e:
                    logger.warning(f"Streaming re-ranking failed ({e}). Proceeding with first-stage candidate order.")
                    retrieved_items = retrieved_items[:request.limit]

            # Step 3: Handle Empty Retrieval
            if not retrieved_items:
                self.conversation_repository.add_message(
                    conversation_id=conv_id,
                    user_id=user_id,
                    role="assistant",
                    content=INSUFFICIENT_CONTEXT_MESSAGE,
                    sources=[],
                )
                yield format_sse("token", {"token": INSUFFICIENT_CONTEXT_MESSAGE})
                yield format_sse("sources", {
                    "sources": [],
                    "document_names": [],
                    "page_numbers": [],
                })
                yield format_sse("complete", {
                    "conversation_id": str(conv_id),
                    "grounded": False,
                    "is_insufficient_context": True,
                    "finish_reason": "INSUFFICIENT_CONTEXT",
                    "total_tokens": 0,
                })
                return

            # Step 4: Deduplicate Chunks and Construct Context
            unique_chunks: List[SearchResultItem] = []
            seen_keys: Set[Tuple[str, int]] = set()

            for item in retrieved_items:
                chunk_key = (str(item.document_id), item.chunk_index)
                if chunk_key not in seen_keys:
                    seen_keys.add(chunk_key)
                    unique_chunks.append(item)

            context_chunks: List[ContextChunk] = []
            sources: List[ChatSourceItem] = []
            document_names_set = set()
            page_numbers_set = set()

            for idx, item in enumerate(unique_chunks, 1):
                is_pdf = item.document_name.lower().endswith(".pdf")
                effective_page = item.page_number if is_pdf else None
                ext = "pdf" if is_pdf else (item.document_name.split(".")[-1].lower() if "." in item.document_name else "docx")
                source_id = f"Source {idx}"

                context_chunks.append(
                    ContextChunk(
                        content=item.content,
                        document_name=item.document_name,
                        document_id=item.document_id,
                        chunk_id=item.chunk_id,
                        page_number=effective_page,
                        chunk_index=item.chunk_index,
                        similarity=item.similarity,
                        source_id=source_id,
                        file_type=ext,
                    )
                )

                snippet = item.content[:140].replace("\n", " ").strip()
                if len(item.content) > 140:
                    snippet += "..."

                sources.append(
                    ChatSourceItem(
                        chunk_id=item.chunk_id,
                        chunk_identifier=str(item.chunk_id) if item.chunk_id else f"{item.document_id}:{item.chunk_index}",
                        document_id=item.document_id,
                        document_name=item.document_name,
                        page_number=effective_page,
                        chunk_index=item.chunk_index,
                        similarity=item.similarity,
                        snippet=snippet,
                        relevant_excerpt=snippet,
                        source_id=source_id,
                        has_page_number=is_pdf and (item.page_number is not None),
                        file_type=ext,
                    )
                )
                document_names_set.add(item.document_name)
                if effective_page is not None:
                    page_numbers_set.add(effective_page)

            distinct_doc_names = sorted(list(document_names_set))
            distinct_pages = sorted(list(page_numbers_set))

            # Step 5: Stream Tokens from Gemini
            full_answer = ""
            tokens_stream = self.generation_service.generate_answer_stream(
                question=clean_question,
                context=context_chunks,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                history=recent_history,
            )

            for token_chunk in tokens_stream:
                # Check for client cancellation
                if raw_request is not None and hasattr(raw_request, "is_disconnected"):
                    disc = raw_request.is_disconnected
                    is_disc = False
                    if callable(disc):
                        try:
                            res = disc()
                            if inspect.iscoroutine(res):
                                res.close()
                                is_disc = False
                            else:
                                is_disc = bool(res)
                        except Exception:
                            is_disc = False
                    else:
                        is_disc = bool(disc)

                    if is_disc:
                        logger.info(f"Client disconnected during streaming for user={user_id}. Terminating generation.")
                        return

                full_answer += token_chunk
                yield format_sse("token", {"token": token_chunk})

            # Step 6: Validate/Sanitize Citations & Yield Sources
            valid_source_tags = {c.source_id for c in context_chunks if c.source_id}
            sanitize_res = self.generation_service.validate_and_sanitize_citations(full_answer, valid_source_tags)
            if isinstance(sanitize_res, tuple) and len(sanitize_res) == 2:
                sanitized_answer, _ = sanitize_res
            else:
                sanitized_answer = full_answer

            serialized_sources = [s.model_dump(mode="json") for s in sources]
            yield format_sse("sources", {
                "sources": serialized_sources,
                "document_names": distinct_doc_names,
                "page_numbers": distinct_pages,
            })

            # Persist assistant response
            self.conversation_repository.add_message(
                conversation_id=conv_id,
                user_id=user_id,
                role="assistant",
                content=sanitized_answer or full_answer,
                sources=serialized_sources,
            )

            # Step 7: Yield Complete Event
            yield format_sse("complete", {
                "conversation_id": str(conv_id),
                "grounded": True,
                "is_insufficient_context": False,
                "finish_reason": "STOP",
            })

        except (GeminiRateLimitError, GeminiTimeoutError, GeminiAPIKeyMissingError, GenerationError, Exception) as e:
            logger.error(f"Error during RAG streaming: {e}")
            yield format_sse("error", {"error": str(e)})


def format_sse(event: str, data: Dict[str, Any]) -> str:
    """Formats payload into a standard Server-Sent Event (SSE) data frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


default_rag_service = RAGOrchestratorService()
