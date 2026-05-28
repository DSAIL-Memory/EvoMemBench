import os
import time

from openai import AzureOpenAI as AzureClient
from openai import OpenAI as OpenAIClient

from memos.configs.embedder import UniversalAPIEmbedderConfig
from memos.embedders.base import BaseEmbedder
from memos.log import get_logger
from memos.utils import timed_with_status


logger = get_logger(__name__)


def _sanitize_unicode(text: str) -> str:
    """
    Remove Unicode surrogates and other problematic characters.
    Surrogates (U+D800-U+DFFF) cause UnicodeEncodeError with some APIs.
    """
    try:
        # Encode with 'surrogatepass' then decode, replacing invalid chars
        cleaned = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        # Replace replacement char with empty string for cleaner output
        return cleaned.replace("\ufffd", "")
    except Exception:
        # Fallback: remove all non-BMP characters
        return "".join(c for c in text if ord(c) < 0x10000)


class UniversalAPIEmbedder(BaseEmbedder):
    def __init__(self, config: UniversalAPIEmbedderConfig):
        self.provider = config.provider
        self.config = config

        if self.provider == "openai":
            self.client = OpenAIClient(
                api_key=config.api_key,
                base_url=config.base_url,
                default_headers=config.headers_extra if config.headers_extra else None,
            )
        elif self.provider == "azure":
            self.client = AzureClient(
                azure_endpoint=config.base_url,
                api_version="2024-03-01-preview",
                api_key=config.api_key,
            )
        else:
            raise ValueError(f"Embeddings unsupported provider: {self.provider}")
        self.use_backup_client = config.backup_client
        if self.use_backup_client:
            self.backup_client = OpenAIClient(
                api_key=config.backup_api_key,
                base_url=config.backup_base_url,
                default_headers=config.backup_headers_extra
                if config.backup_headers_extra
                else None,
            )

    # DashScope text-embedding-v4 accepts at most 10 texts per request.
    # Use a conservative default that also works for other providers.
    _BATCH_SIZE = int(os.getenv("MOS_EMBED_BATCH_SIZE", 10))

    def _embed_batch(self, texts: list[str], client, model: str) -> list[list[float]]:
        """Call the embedding API for a single batch (no batching logic here)."""
        init_time = time.time()
        response = client.embeddings.create(model=model, input=texts)
        logger.info(f"Embeddings batch ({len(texts)} texts) succeeded in {time.time() - init_time:.2f}s")
        return [r.embedding for r in response.data]

    @timed_with_status(
        log_prefix="model_timed_embedding",
        log_extra_args=lambda self, texts: {
            "model_name_or_path": "text-embedding-3-large",
            "text_len": len(texts),
            "text_content": texts,
        },
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # Sanitize Unicode to prevent encoding errors with emoji/surrogates
        texts = [_sanitize_unicode(t) for t in texts]
        # Truncate texts if max_tokens is configured
        texts = self._truncate_texts(texts)
        logger.info(f"Embeddings request: {len(texts)} texts")
        if self.provider not in ("openai", "azure"):
            raise ValueError(f"Embeddings unsupported provider: {self.provider}")

        model = getattr(self.config, "model_name_or_path", "text-embedding-3-large")
        try:
            # Split into batches to respect per-request limits (e.g. DashScope: max 10)
            results: list[list[float]] = []
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                results.extend(self._embed_batch(batch, self.client, model))
            return results
        except Exception as e:
            if self.use_backup_client:
                logger.warning(
                    f"Embeddings request ended with {type(e).__name__} error: {e}, try backup client"
                )
                try:
                    backup_model = getattr(self.config, "backup_model_name_or_path", model)
                    results = []
                    for i in range(0, len(texts), self._BATCH_SIZE):
                        batch = texts[i : i + self._BATCH_SIZE]
                        results.extend(self._embed_batch(batch, self.backup_client, backup_model))
                    return results
                except Exception as e2:
                    raise ValueError(f"Backup embeddings request ended with error: {e2}") from e2
            else:
                raise ValueError(f"Embeddings request ended with error: {e}") from e
