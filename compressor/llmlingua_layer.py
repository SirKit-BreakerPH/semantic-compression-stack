"""
Layer 2 — Semantic Token Pruning (LLMLingua)
--------------------------------------------
Uses Microsoft's LLMLingua to remove low-information tokens while
preserving semantic meaning. Supports both LLMLingua and LLMLingua-2.

Typical compression: 2–5× with <5% semantic loss.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal

_COMPRESSOR = None  # lazy-loaded singleton


@dataclass
class LinguaResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    rate: float  # target rate used


def _get_compressor(model_name: str):
    global _COMPRESSOR
    if _COMPRESSOR is None:
        try:
            from llmlingua import PromptCompressor
            _COMPRESSOR = PromptCompressor(
                model_name=model_name,
                use_llmlingua2=("llmlingua-2" in model_name.lower()),
                device_map="cpu",
            )
        except ImportError:
            raise ImportError(
                "llmlingua not installed. Run: pip install llmlingua"
            )
    return _COMPRESSOR


def compress(
    text: str,
    target_token_rate: float = 0.5,
    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    instruction: str = "",
    question: str = "",
    mode: Literal["auto", "aggressive", "conservative"] = "auto",
) -> LinguaResult:
    """
    Apply LLMLingua semantic compression.

    Args:
        text:              Input text.
        target_token_rate: Fraction of tokens to retain (0.1–0.9).
                           Lower = more aggressive.
        model_name:        HuggingFace model for perplexity scoring.
        instruction:       Optional task context (improves preservation).
        question:          Optional query (preserves relevant tokens).
        mode:              "auto" uses target_token_rate; "aggressive" = 0.3;
                           "conservative" = 0.7.

    Returns:
        LinguaResult with compressed text and stats.
    """
    if mode == "aggressive":
        target_token_rate = 0.3
    elif mode == "conservative":
        target_token_rate = 0.7

    compressor = _get_compressor(model_name)

    kwargs: dict = dict(
        target_token=target_token_rate,
        force_tokens=["\n", ".", "?", "!"],      # preserve sentence structure
        drop_consecutive=True,
    )
    if instruction:
        kwargs["instruction"] = instruction
    if question:
        kwargs["question"] = question

    result = compressor.compress_prompt(text, **kwargs)

    compressed = result["compressed_prompt"]
    origin_tokens = result.get("origin_tokens", len(text.split()))
    compressed_tokens = result.get("compressed_tokens", len(compressed.split()))

    return LinguaResult(
        compressed_text=compressed,
        original_tokens=origin_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(origin_tokens / max(compressed_tokens, 1), 2),
        rate=target_token_rate,
    )


def decompress_hint(compressed: str) -> str:
    """
    LLMLingua compression is not directly reversible.
    Returns a hint string for the LLM reconstruction layer.
    """
    return (
        "The following text was semantically compressed by LLMLingua. "
        "Reconstruct the full, natural-language version, inferring implied "
        "words and grammatical connectors:\n\n" + compressed
    )
