"""gemini-3-flash handler — same logic as gpt5mini (same OpenAI-compatible endpoint)."""
from bfcl_eval.model_handler.api_inference.gpt5mini import (
    Gpt5MiniFCHandler as Gemini3FlashFCHandler,
    Gpt5MiniPromptingHandler as Gemini3FlashPromptingHandler,
)

__all__ = ["Gemini3FlashFCHandler", "Gemini3FlashPromptingHandler"]
