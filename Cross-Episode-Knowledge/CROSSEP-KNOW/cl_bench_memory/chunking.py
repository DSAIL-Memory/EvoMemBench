"""Sentence-boundary text chunking utility shared by BM25 and embedding backends.

Mirrors MAB's utils/eval_other_utils.py::chunk_text_into_sentences.
"""

import nltk
import tiktoken


def chunk_text(text: str, chunk_size: int = 1024) -> list:
    """Split text into chunks of <= chunk_size tokens at sentence boundaries.

    Uses tiktoken (gpt-4o-mini encoding) for token counting and NLTK
    punkt for sentence segmentation — same approach as MAB.
    """
    nltk.download("punkt_tab", quiet=True)
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        token_count = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current_token_count + token_count > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_token_count = token_count
        else:
            current_sentences.append(sentence)
            current_token_count += token_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks if chunks else [text]
