"""Gemini Vector Embedding Service using the official Google Gen AI Python SDK."""
import time
import logging
from typing import List, Optional, Union
from google import genai
from google.genai import types
from google.genai.errors import APIError

from ..config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""
    pass

class GeminiAPIKeyMissingError(EmbeddingError):
    """Raised when the Gemini API key is not configured."""
    pass

class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when the generated embedding vector does not match the configured dimension."""
    pass

class GeminiEmbeddingService:
    """
    Manages vector embedding generation using Google Gemini text-embedding-004.
    Guarantees consistent 768-dimensional outputs for Supabase pgvector compatibility.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        # Normalize model name: e.g. "models/text-embedding-004" -> "text-embedding-004"
        raw_model = model_name or settings.GEMINI_EMBEDDING_MODEL
        self.model = raw_model.replace("models/", "") if raw_model else "text-embedding-004"
        self.dimension = dimension or settings.GEMINI_EMBEDDING_DIMENSIONS
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        """Initializes and returns the Google Gen AI client with validation."""
        api_key = self.api_key or settings.GEMINI_API_KEY
        if not api_key or api_key in ("your_gemini_api_key_here", ""):
            raise GeminiAPIKeyMissingError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your "
                "environment or backend/.env file."
            )
        if self._client is None or self.api_key != api_key:
            self.api_key = api_key
            self._client = genai.Client(api_key=api_key)
        return self._client

    def embed_text(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
    ) -> List[float]:
        """
        Generates a vector embedding for a single text input.
        - task_type: "RETRIEVAL_DOCUMENT" for indexed chunks, "RETRIEVAL_QUERY" for search queries.
        - Validates input and dimension (768).
        - Retries on transient API errors with exponential backoff.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        cleaned_text = text.strip()
        client = self._get_client()

        config_dict = {
            "output_dimensionality": self.dimension,
            "task_type": task_type,
        }
        if title and task_type == "RETRIEVAL_DOCUMENT":
            config_dict["title"] = title

        config = types.EmbedContentConfig(**config_dict)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Calling Gemini embed_content (attempt {attempt}/{self.max_retries}, model={self.model})")
                response = client.models.embed_content(
                    model=self.model,
                    contents=cleaned_text,
                    config=config,
                )

                if not response.embeddings or not response.embeddings[0].values:
                    raise EmbeddingError("Gemini API returned an empty embedding response.")

                vector = list(response.embeddings[0].values)
                actual_dim = len(vector)

                if actual_dim != self.dimension:
                    raise EmbeddingDimensionMismatchError(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {actual_dim} from Gemini API."
                    )

                return vector

            except (EmbeddingDimensionMismatchError, GeminiAPIKeyMissingError):
                raise
            except (APIError, Exception) as e:
                # If error is a configuration / auth error, fail immediately without retry
                err_str = str(e).lower()
                if "api_key" in err_str or "unregistered" in err_str or "permission" in err_str:
                    logger.error(f"Gemini API authentication failed: {e}")
                    raise EmbeddingError(f"Gemini API Authentication Error: {str(e)}") from e

                if attempt == self.max_retries:
                    logger.error(f"Gemini embedding failed after {self.max_retries} attempts: {e}")
                    raise EmbeddingError(f"Gemini embedding generation failed: {str(e)}") from e

                sleep_time = self.backoff_factor ** attempt
                logger.warning(f"Transient error generating embedding ({e}). Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        raise EmbeddingError("Failed to generate embedding: Maximum retries exceeded.")

    def embed_chunks(
        self,
        chunks: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 50,
    ) -> List[List[float]]:
        """
        Generates vector embeddings for a list of document chunks efficiently.
        Processes chunks in batches to respect rate limits and API batch size limits.
        """
        if not chunks:
            return []

        # Validate chunks
        cleaned_chunks: List[str] = []
        for idx, c in enumerate(chunks):
            if not c or not c.strip():
                raise ValueError(f"Chunk at index {idx} is empty or whitespace.")
            cleaned_chunks.append(c.strip())

        client = self._get_client()
        all_embeddings: List[List[float]] = []

        config = types.EmbedContentConfig(
            output_dimensionality=self.dimension,
            task_type=task_type,
        )

        # Process in batches
        for start_idx in range(0, len(cleaned_chunks), batch_size):
            batch = cleaned_chunks[start_idx : start_idx + batch_size]
            logger.info(
                f"Generating embeddings for batch {start_idx // batch_size + 1} "
                f"({len(batch)} chunks, model={self.model})"
            )

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=config,
                    )

                    if not response.embeddings or len(response.embeddings) != len(batch):
                        raise EmbeddingError(
                            f"Mismatch in returned embeddings count: sent {len(batch)}, "
                            f"received {len(response.embeddings) if response.embeddings else 0}."
                        )

                    for emb in response.embeddings:
                        vector = list(emb.values)
                        if len(vector) != self.dimension:
                            raise EmbeddingDimensionMismatchError(
                                f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}."
                            )
                        all_embeddings.append(vector)
                    break

                except (EmbeddingDimensionMismatchError, GeminiAPIKeyMissingError):
                    raise
                except (APIError, Exception) as e:
                    err_str = str(e).lower()
                    if "api_key" in err_str or "permission" in err_str:
                        logger.error(f"Gemini API authentication failed: {e}")
                        raise EmbeddingError(f"Gemini API Authentication Error: {str(e)}") from e

                    if attempt == self.max_retries:
                        logger.error(f"Batch embedding failed after {self.max_retries} attempts: {e}")
                        raise EmbeddingError(f"Failed to generate embeddings for batch: {str(e)}") from e

                    sleep_time = self.backoff_factor ** attempt
                    logger.warning(f"Transient error in batch embedding ({e}). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)

        return all_embeddings

default_embedding_service = GeminiEmbeddingService()
