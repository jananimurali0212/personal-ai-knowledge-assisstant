"""Unit and integration tests for Gemini LLM Answer Generation Service and API routes."""
import os
from uuid import uuid4
from unittest.mock import MagicMock, patch
import pytest
from app.config.settings import settings
from app.routes.generation import (
    generate_answer_endpoint,
    rag_query_endpoint,
)
from app.services.generation import default_generation_service
from app.models.generation import (
    ContextChunk,
    SourceCitation,
    GenerateAnswerRequest,
    GenerateAnswerResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.services.generation import (
    GeminiGenerationService,
    GenerationError,
    GeminiAPIKeyMissingError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    SYSTEM_INSTRUCTION,
)
from app.services.search import (
    SemanticSearchService,
    default_semantic_search_service,
)
from app.services.chunk_repository import _in_memory_chunks_store

# Sample test context
SAMPLE_CHUNKS = [
    ContextChunk(
        content="Inheritance in Java allows a subclass to acquire methods and fields from a superclass using the 'extends' keyword.",
        document_name="Java_OOP_Guide.pdf",
        document_id=uuid4(),
        page_number=1,
        chunk_index=0,
        similarity=0.92,
    ),
    ContextChunk(
        content="Method overriding allows a subclass to provide a specific implementation of a method defined in its superclass using @Override.",
        document_name="Java_OOP_Guide.pdf",
        document_id=uuid4(),
        page_number=2,
        chunk_index=1,
        similarity=0.88,
    ),
]


class TestGeminiGenerationServiceConfiguration:
    """Verifies environment-based configuration and initialization."""

    def test_generation_model_from_env(self):
        with patch.dict(os.environ, {"GEMINI_GENERATION_MODEL": "gemini-2.5-flash"}):
            svc = GeminiGenerationService(api_key="test_key", model_name="gemini-2.5-flash")
            assert svc.model == "gemini-2.5-flash"

    def test_model_name_prefix_stripping(self):
        svc = GeminiGenerationService(api_key="test_key", model_name="models/gemini-2.5-flash")
        assert svc.model == "gemini-2.5-flash"

    def test_default_generation_parameters(self):
        svc = GeminiGenerationService(api_key="test_key")
        assert svc.temperature == settings.GEMINI_GENERATION_TEMPERATURE
        assert svc.max_output_tokens == settings.GEMINI_GENERATION_MAX_TOKENS
        assert svc.timeout == settings.GEMINI_GENERATION_TIMEOUT
        assert svc.max_retries == 3


class TestRequestValidation:
    """Tests input validation for questions and context."""

    def test_empty_question_raises_value_error(self):
        svc = GeminiGenerationService(api_key="test_key")
        with pytest.raises(ValueError, match="cannot be empty"):
            svc.generate_answer(question="", context=SAMPLE_CHUNKS)

    def test_whitespace_question_raises_value_error(self):
        svc = GeminiGenerationService(api_key="test_key")
        with pytest.raises(ValueError, match="cannot be empty"):
            svc.generate_answer(question="   \n  \t ", context=SAMPLE_CHUNKS)

    def test_too_short_question_raises_value_error(self):
        svc = GeminiGenerationService(api_key="test_key")
        with pytest.raises(ValueError, match="at least 2 characters"):
            svc.generate_answer(question="?", context=SAMPLE_CHUNKS)

    def test_too_long_question_raises_value_error(self):
        svc = GeminiGenerationService(api_key="test_key")
        with pytest.raises(ValueError, match="cannot exceed 2000 characters"):
            svc.generate_answer(question="A" * 2001, context=SAMPLE_CHUNKS)

    def test_pydantic_request_model_validation(self):
        with pytest.raises(ValueError):
            GenerateAnswerRequest(question=" ", context=["some text"])


class TestContextNormalizationAndPromptConstruction:
    """Tests normalization of various context input formats and prompt structure."""

    def test_normalize_dict_context(self):
        svc = GeminiGenerationService(api_key="test_key")
        doc_id = uuid4()
        dict_context = [
            {
                "content": "Polymorphism enables one interface to be used for a general class of actions.",
                "document_name": "OOP_Principles.pdf",
                "document_id": str(doc_id),
                "page_number": 3,
                "chunk_index": 2,
                "similarity": 0.85,
            }
        ]
        normalized = svc.normalize_context(dict_context)
        assert len(normalized) == 1
        assert isinstance(normalized[0], ContextChunk)
        assert normalized[0].document_name == "OOP_Principles.pdf"
        assert normalized[0].page_number == 3
        assert normalized[0].chunk_index == 2
        assert normalized[0].similarity == 0.85

    def test_normalize_string_list_context_with_metadata(self):
        svc = GeminiGenerationService(api_key="test_key")
        str_context = ["First chunk content", "Second chunk content"]
        metadata = [
            {"document_name": "Doc1.pdf", "page_number": 1},
            {"document_name": "Doc2.pdf", "page_number": 5},
        ]
        normalized = svc.normalize_context(str_context, source_metadata=metadata)
        assert len(normalized) == 2
        assert normalized[0].document_name == "Doc1.pdf"
        assert normalized[1].document_name == "Doc2.pdf"
        assert normalized[1].page_number == 5

    def test_normalize_single_string_context(self):
        svc = GeminiGenerationService(api_key="test_key")
        normalized = svc.normalize_context("Single raw context block")
        assert len(normalized) == 1
        assert normalized[0].content == "Single raw context block"
        assert normalized[0].document_name == "Provided Context"

    def test_build_structured_prompt_format(self):
        svc = GeminiGenerationService(api_key="test_key")
        prompt = svc.build_structured_prompt("What is inheritance?", SAMPLE_CHUNKS)

        assert "### RETRIEVED DOCUMENT CONTEXT" in prompt
        assert "[Source 1] Document: \"Java_OOP_Guide.pdf\" | Page 1" in prompt
        assert "[Source 2] Document: \"Java_OOP_Guide.pdf\" | Page 2" in prompt
        assert "### USER QUESTION:" in prompt
        assert "What is inheritance?" in prompt
        assert "### INSTRUCTIONS:" in prompt
        assert "Cite supporting sources using their exact identifiers" in prompt

    def test_validate_and_sanitize_fabricated_citations(self):
        svc = GeminiGenerationService(api_key="test_key")
        valid_sources = {"Source 1", "Source 2"}
        # Generated text with valid [Source 1] and fabricated [Source 99]
        text_with_fabricated = (
            "Inheritance allows code reuse [Source 1]. "
            "Fabricated claim from imaginary source [Source 99]."
        )
        cleaned_text, cited_tags = svc.validate_and_sanitize_citations(text_with_fabricated, valid_sources)
        assert "[Source 1]" in cleaned_text
        assert "[Source 99]" not in cleaned_text
        assert cited_tags == ["Source 1"]


class TestAnswerGenerationExecution:
    """Tests Gemini generation calls, citations, and grounding rules."""

    def test_empty_context_early_exit_insufficient(self):
        svc = GeminiGenerationService(api_key="test_key")
        res = svc.generate_answer(question="What is Java?", context=[])
        assert res.is_insufficient_context is True
        assert res.grounded is False
        assert "does not contain sufficient information" in res.answer
        assert len(res.sources) == 0

    def test_successful_grounded_answer_generation(self):
        svc = GeminiGenerationService(api_key="test_key", model_name="gemini-2.5-flash")

        mock_response = MagicMock()
        mock_response.text = (
            "Inheritance in Java allows a subclass to acquire methods and fields from a superclass using the `extends` keyword. "
            "According to Java_OOP_Guide.pdf (Page 1), it supports code reuse."
        )
        mock_response.usage_metadata.total_token_count = 145
        mock_cand = MagicMock()
        mock_cand.finish_reason = "STOP"
        mock_response.candidates = [mock_cand]

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            svc._client = mock_client

            result = svc.generate_answer(
                question="What is inheritance in Java?",
                context=SAMPLE_CHUNKS,
                temperature=0.1,
            )

            assert mock_client.models.generate_content.called
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            assert call_kwargs["model"] == "gemini-2.5-flash"
            assert call_kwargs["config"].temperature == 0.1
            assert call_kwargs["config"].system_instruction == SYSTEM_INSTRUCTION

            assert result.grounded is True
            assert result.is_insufficient_context is False
            assert "Inheritance in Java" in result.answer
            assert len(result.sources) == 2
            assert result.sources[0].document_name == "Java_OOP_Guide.pdf"
            assert result.sources[0].page_number == 1
            assert result.tokens_used == 145

    def test_insufficient_context_detected_from_llm(self):
        svc = GeminiGenerationService(api_key="test_key")

        mock_response = MagicMock()
        mock_response.text = "The provided document context does not contain sufficient information to answer this question."
        mock_response.usage_metadata.total_token_count = 42
        mock_response.candidates = []

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            svc._client = mock_client

            result = svc.generate_answer(
                question="How do I bake cookies?",
                context=SAMPLE_CHUNKS,
            )

            assert result.grounded is False
            assert result.is_insufficient_context is True
            assert "does not contain sufficient information" in result.answer


class TestErrorHandlingAndResilience:
    """Tests API key verification, rate limits, timeouts, and backoff retries."""

    def test_missing_api_key_raises_error(self):
        svc = GeminiGenerationService(api_key="")
        with patch.object(settings, "GEMINI_API_KEY", ""):
            svc.api_key = ""
            with pytest.raises(GeminiAPIKeyMissingError):
                svc.generate_answer(question="What is Java?", context=SAMPLE_CHUNKS)

    def test_rate_limit_retry_and_success(self):
        svc = GeminiGenerationService(api_key="test_key", max_retries=3, backoff_factor=0.01)

        mock_response = MagicMock()
        mock_response.text = "Success after rate limit backoff."
        mock_response.usage_metadata.total_token_count = 50
        mock_response.candidates = []

        # Attempt 1: 429 Resource Exhausted; Attempt 2: Success
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = [
                Exception("429 Resource exhausted: quota exceeded"),
                mock_response,
            ]
            svc._client = mock_client

            result = svc.generate_answer(question="What is Java?", context=SAMPLE_CHUNKS)
            assert result.answer == "Success after rate limit backoff."
            assert mock_client.models.generate_content.call_count == 2

    def test_rate_limit_exhausted_raises_gemini_rate_limit_error(self):
        svc = GeminiGenerationService(api_key="test_key", max_retries=2, backoff_factor=0.01)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("429 Resource exhausted")
            svc._client = mock_client

            with pytest.raises(GeminiRateLimitError, match="rate limit exceeded"):
                svc.generate_answer(question="What is Java?", context=SAMPLE_CHUNKS)

    def test_authentication_error_fails_immediately_without_retry(self):
        svc = GeminiGenerationService(api_key="test_key", max_retries=3)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("API_KEY_INVALID: permission denied")
            svc._client = mock_client

            with pytest.raises(GenerationError, match="Authentication Error"):
                svc.generate_answer(question="What is Java?", context=SAMPLE_CHUNKS)

            # Should fail immediately on attempt 1 without retry
            assert mock_client.models.generate_content.call_count == 1


class TestGenerationAPIRoutes:
    """Tests the FastAPI generation endpoints."""

    def test_direct_answer_endpoint(self):
        user_id = uuid4()
        mock_response = MagicMock()
        mock_response.text = "Inheritance allows subclasses to inherit methods."
        mock_response.usage_metadata.total_token_count = 80
        mock_response.candidates = []

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response

            with patch.object(default_generation_service, "_client", mock_client):
                with patch.object(default_generation_service, "api_key", "test_key"):
                    req = GenerateAnswerRequest(
                        question="What is inheritance?",
                        context=[
                            ContextChunk(
                                content="Inheritance allows subclasses to inherit methods.",
                                document_name="OOP.pdf",
                                page_number=1,
                            )
                        ],
                    )

                    res = generate_answer_endpoint(request=req, current_user=user_id)
                    assert res.question == "What is inheritance?"
                    assert "Inheritance allows subclasses" in res.answer
                    assert res.grounded is True
                    assert len(res.sources) == 1
                    assert res.sources[0].document_name == "OOP.pdf"

    def test_rag_query_endpoint(self):
        user_id = uuid4()
        doc_id = uuid4()

        # Seed in-memory store
        doc_key = f"{user_id}:{doc_id}"
        _in_memory_chunks_store[doc_key] = [
            {
                "id": str(uuid4()),
                "document_id": str(doc_id),
                "document_name": "Java_Inheritance.pdf",
                "user_id": str(user_id),
                "chunk_index": 0,
                "page_number": 1,
                "content": "Inheritance provides code reusability in Java.",
                "embedding": [0.0] * 768,
            }
        ]

        mock_emb = MagicMock()
        mock_emb.model = "text-embedding-004"
        mock_emb.dimension = 768
        mock_emb.embed_text.return_value = [0.0] * 768

        mock_llm_response = MagicMock()
        mock_llm_response.text = "According to Java_Inheritance.pdf, inheritance provides code reusability."
        mock_llm_response.usage_metadata.total_token_count = 95
        mock_llm_response.candidates = []

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_llm_response

            with patch.object(default_generation_service, "_client", mock_client):
                with patch.object(default_generation_service, "api_key", "test_key"):
                    with patch.object(default_semantic_search_service, "embedding_service", mock_emb):
                        with patch.object(default_semantic_search_service.chunk_repository, "is_configured", return_value=False):
                            req = RAGQueryRequest(
                                question="What does inheritance provide in Java?",
                                match_threshold=0.0,
                                limit=3,
                            )

                            res = rag_query_endpoint(request=req, current_user=user_id)
                            assert "code reusability" in res.answer
                            assert res.retrieved_chunks_count == 1
                            assert res.grounded is True
                            assert len(res.sources) == 1
                            assert res.sources[0].document_name == "Java_Inheritance.pdf"
