from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal

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
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*[.\-*]\s*$', '', text, flags=re.MULTILINE)
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

    target_token_rate = max(target_token_rate, 0.15)
    clean_text = _preprocess(text)
    original_tokens = _count(clean_text)

    if original_tokens < 20:
        return LinguaResult(clean_text, original_tokens, original_tokens, 1.0, target_token_rate)

    compressor = _get_compressor(model_name)

    # Split into paragraphs and pass as context list — correct LLMLingua API format
    paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip() and len(p.split()) > 3]
    if not paragraphs:
        paragraphs = [clean_text]

    try:
        result = compressor.compress_prompt(
            context=paragraphs,
            instruction=instruction or "",
            question=question or "What is the main content?",
            target_token=int(original_tokens * target_token_rate),
            condition_compare=True,
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
        )
        compressed = result.get("compressed_prompt", "").strip()
    except Exception:
        # Fallback to simple call if advanced params not supported
        try:
            result = compressor.compress_prompt(
                context=paragraphs,
                target_token=int(original_tokens * target_token_rate),
            )
            compressed = result.get("compressed_prompt", "").strip()
        except Exception:
            compressed = ""

    # Safety net: if output has fewer than 10 real words, return cleaned original
    real_words = [w for w in compressed.split() if re.search(r'[a-zA-Z]{2,}', w)]
    if len(real_words) < 10:
        compressed = clean_text

    compressed_tokens = _count(compressed)
    return LinguaResult(
        compressed_text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(original_tokens / max(compressed_tokens, 1), 2),
        rate=target_token_rate,
    )

def decompress_hint(compressed: str) -> str:
    return (
        "The following text was semantically compressed by LLMLingua. "
        "Reconstruct the full natural-language version:\n\n" + compressed
    )
