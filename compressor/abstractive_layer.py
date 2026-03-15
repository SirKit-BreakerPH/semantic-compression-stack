"""
Layer 5 — Abstractive LLM Compression & Reconstruction (Cmprsr-style)
----------------------------------------------------------------------
Uses a free LLM API (Groq or Gemini) or local BART fallback to:
  - COMPRESS: abstractively summarize while preserving key facts/structure
  - RECONSTRUCT: expand compressed text back to near-original fidelity

FREE API options (no credit card required):
  - Groq  : https://console.groq.com  → set GROQ_API_KEY
  - Gemini: https://aistudio.google.com → set GEMINI_API_KEY

Local fallback (no key needed):
  - facebook/bart-large-cnn (summarization)
  - t5-base (reconstruction)

Priority order: groq → gemini → local BART
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Literal, Optional

_HF_PIPELINE = None  # lazy HuggingFace pipeline

# ── Free model choices ─────────────────────────────────────────────────────
GROQ_DEFAULT_MODEL  = "llama-3.1-8b-instant"   # fast, free tier
GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"       # free tier, generous limits

COMPRESS_SYSTEM = """You are a semantic compression engine. Compress the following text to approximately {ratio}% of its original length while:
1. Preserving ALL key facts, numbers, names, and code identifiers exactly
2. Preserving logical structure and cause-effect relationships
3. Using telegraphic style — drop filler words, reduce adjectives
4. Keeping all technical terms, variable names, and URLs intact
Output ONLY the compressed text, no preamble or explanation."""

RECONSTRUCT_SYSTEM = """You are a semantic reconstruction engine. The following text was compressed — tokens removed, prose shortened. Reconstruct the full natural-language version:
1. Expand telegraphic phrases into complete sentences
2. Re-add appropriate articles, connectors, and transitions
3. Preserve ALL technical terms, names, numbers, and identifiers exactly
4. Match the original document's likely tone and structure
5. Do NOT invent information not implied by the compressed text
Output ONLY the reconstructed text, no preamble."""


@dataclass
class AbstractiveResult:
    output_text: str
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    model_used: str
    backend: str
    mode: str  # "compress" | "reconstruct"


def _token_count(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


# ── Groq (OpenAI-compatible, free tier) ──────────────────────────────────
def _groq_call(prompt: str, system: str, model: str, api_key: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ── Google Gemini (free tier via REST) ────────────────────────────────────
def _gemini_call(prompt: str, system: str, model: str, api_key: str) -> str:
    full_prompt = f"{system}\n\n{prompt}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
    }).encode()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ── Local HuggingFace BART (zero API key) ─────────────────────────────────
def _hf_pipeline_get():
    global _HF_PIPELINE
    if _HF_PIPELINE is None:
        from transformers import pipeline
        _HF_PIPELINE = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=-1
        )
    return _HF_PIPELINE


def _hf_compress(text: str, target_ratio: float) -> str:
    pipe = _hf_pipeline_get()
    word_count = len(text.split())
    max_len = max(30, int(word_count * target_ratio))
    min_len = max(10, int(max_len * 0.5))
    if word_count > 800:
        chunks = [text[i : i + 3000] for i in range(0, len(text), 3000)]
        chunk_max = max(20, max_len // len(chunks))
        parts = [
            pipe(c, max_length=chunk_max, min_length=5, do_sample=False)[0]["summary_text"]
            for c in chunks
        ]
        return " ".join(parts)
    return pipe(text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]


def _hf_reconstruct(compressed: str) -> str:
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        ids = tokenizer(
            "expand and reconstruct: " + compressed,
            return_tensors="pt", max_length=512, truncation=True,
        ).input_ids
        out = model.generate(ids, max_length=1024, num_beams=4, early_stopping=True)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        return compressed


# ── Auto-detect available backend ─────────────────────────────────────────
def _resolve_backend(
    preferred: str,
    groq_model: str,
    gemini_model: str,
) -> tuple[str, str, str]:
    """
    Returns (backend_name, model, api_key).
    Priority: preferred → groq → gemini → hf
    """
    groq_key   = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if preferred == "groq" and groq_key:
        return "groq", groq_model, groq_key
    if preferred == "gemini" and gemini_key:
        return "gemini", gemini_model, gemini_key
    if preferred == "hf":
        return "hf", "bart-large-cnn", ""

    # auto
    if groq_key:
        return "groq", groq_model, groq_key
    if gemini_key:
        return "gemini", gemini_model, gemini_key
    return "hf", "bart-large-cnn", ""


# ── Public API ────────────────────────────────────────────────────────────
def compress(
    text: str,
    target_ratio: float = 0.3,
    backend: Literal["auto", "groq", "gemini", "hf"] = "auto",
    groq_model: str = GROQ_DEFAULT_MODEL,
    gemini_model: str = GEMINI_DEFAULT_MODEL,
) -> AbstractiveResult:
    """
    Abstractively compress text.

    Args:
        text:          Input text.
        target_ratio:  Target length as fraction of original (0.1–0.5).
        backend:       "auto" | "groq" | "gemini" | "hf"
                       "auto" uses whichever API key is set, falls back to local.
        groq_model:    Groq model (default: llama-3.1-8b-instant — free).
        gemini_model:  Gemini model (default: gemini-1.5-flash — free).
    """
    bname, model, api_key = _resolve_backend(backend, groq_model, gemini_model)
    system = COMPRESS_SYSTEM.format(ratio=int(target_ratio * 100))

    if bname == "groq":
        output = _groq_call(text, system, model, api_key)
    elif bname == "gemini":
        output = _gemini_call(text, system, model, api_key)
    else:
        output = _hf_compress(text, target_ratio)

    input_tok  = _token_count(text)
    output_tok = _token_count(output)
    return AbstractiveResult(
        output_text=output,
        input_tokens=input_tok,
        output_tokens=output_tok,
        compression_ratio=round(input_tok / max(output_tok, 1), 2),
        model_used=model,
        backend=bname,
        mode="compress",
    )


def reconstruct(
    compressed: str,
    reconstruction_hints: Optional[list[str]] = None,
    backend: Literal["auto", "groq", "gemini", "hf"] = "auto",
    groq_model: str = GROQ_DEFAULT_MODEL,
    gemini_model: str = GEMINI_DEFAULT_MODEL,
) -> AbstractiveResult:
    """
    Reconstruct compressed text to near-original fidelity.
    """
    hints_block = ""
    if reconstruction_hints:
        hints_block = "\n\nContext: " + " | ".join(reconstruction_hints)

    full_input = compressed + hints_block
    bname, model, api_key = _resolve_backend(backend, groq_model, gemini_model)

    if bname == "groq":
        output = _groq_call(full_input, RECONSTRUCT_SYSTEM, model, api_key)
    elif bname == "gemini":
        output = _gemini_call(full_input, RECONSTRUCT_SYSTEM, model, api_key)
    else:
        output = _hf_reconstruct(compressed)

    input_tok  = _token_count(compressed)
    output_tok = _token_count(output)
    return AbstractiveResult(
        output_text=output,
        input_tokens=input_tok,
        output_tokens=output_tok,
        compression_ratio=round(output_tok / max(input_tok, 1), 2),
        model_used=model,
        backend=bname,
        mode="reconstruct",
    )
