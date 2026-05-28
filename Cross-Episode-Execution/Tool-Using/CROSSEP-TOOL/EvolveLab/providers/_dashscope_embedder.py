"""DashScope text-embedding wrapper with sentence-transformers-compatible interface."""
import os
import numpy as np
from openai import OpenAI


class DashScopeEmbedder:
    DEFAULT_MODEL = "text-embedding-v3"
    MAX_BATCH = 25

    def __init__(self, model: str = None):
        self.model = model or os.getenv("DASHSCOPE_EMBED_MODEL", self.DEFAULT_MODEL)
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _embed_batch(self, texts: list) -> list:
        resp = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
        )
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

    def encode(self, text_or_texts, **kwargs) -> np.ndarray:
        """Mimics sentence-transformers SentenceTransformer.encode() interface."""
        single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if single else list(text_or_texts)
        texts = [t.replace("\n", " ") for t in texts]

        all_embeddings = []
        batch_size = min(kwargs.get("batch_size", self.MAX_BATCH), self.MAX_BATCH)
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            all_embeddings.extend(self._embed_batch(chunk))

        result = np.array(all_embeddings, dtype=np.float32)
        return result[0] if single else result
