from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal, Optional

_COMPRESSOR = None

@dataclass
class LinguaResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    rate: float

def _count(text: str) -> int:
    return len(text.split())

def _preprocess(text: str) -> str:
    """Remove separator lines and excessive blank lines."""
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*[\.\-\*]\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def _get_compressor(model_name: str):
    global _COMPRESSOR
    if _COMPRESSOR is None:
        from llmlingua import PromptCompressor
        _COMPRESSOR = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=("llmlingua-2" in model_name.lower()),
            device_map="cpu",
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
    if mode == "aggressive":
        target_token_rate = 0.3
    elif mode == "conservative":
        target_token_rate = 0.7

    # Never compress below 15% — prevents near-empty output
    target_token_rate = max(target_token_rate, 0.15)

    clean_text = _preprocess(text)
    original_tokens = _count(clean_text)

    # If text is too short after cleaning, return as-is
    if original_tokens < 20:
        return LinguaResult(
            compressed_text=clean_text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            rate=target_token_rate,
        )

    compressor = _get_compressor(model_name)
    kwargs = dict(
        target_token=target_token_rate,
        force_tokens=["\n", ".", "?", "!"],
        drop_consecutive=True,
    )
    if instruction:
        kwargs["instruction"] = instruction
    if question:
        kwargs["question"] = question

    result = compressor.compress_prompt(clean_text, **kwargs)
    compressed = result["compressed_prompt"]

    # Safety net: if output is mostly punctuation/dots, return cleaned original
    real_words = [w for w in compressed.split() if re.search(r'[a-zA-Z]{2,}', w)]
    if len(real_words) < 5:
        compressed = clean_text

    origin_tokens     = result.get("origin_tokens", original_tokens)
    compressed_tokens = result.get("compressed_tokens", _count(compressed))

    return LinguaResult(
        compressed_text=compressed,
        original_tokens=origin_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(origin_tokens / max(compressed_tokens, 1), 2),
        rate=target_token_rate,
    )

def decompress_hint(compressed: str) -> str:
    return (
        "The following text was semantically compressed by LLMLingua. "
        "Reconstruct the full natural-language version:\n\n" + compressed
    )
