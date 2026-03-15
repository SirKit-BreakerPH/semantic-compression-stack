"""
webui/app.py — Flask Web Server for Semantic Compression Stack
--------------------------------------------------------------
Serves the web UI and exposes a REST API that the frontend calls.

Usage:
    python webui/app.py                        # default: port 5000
    python webui/app.py --port 8080
    python webui/app.py --host 0.0.0.0 --port 5000  # expose to network

Environment variables (all optional):
    GROQ_API_KEY    — enables Layer 5 via Groq (free tier)
    GEMINI_API_KEY  — enables Layer 5 via Gemini (free tier)
    SECRET_KEY      — Flask session secret (auto-generated if not set)
    ALLOWED_ORIGINS — comma-separated allowed iFrame origins (default: *)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS

from structure.skeleton_layer import skeletonize
from compressor.token_pruning_layer import prune
from compressor.abstractive_layer import (
    compress as abstractive_compress,
    reconstruct as abstractive_reconstruct,
    GROQ_DEFAULT_MODEL,
    GEMINI_DEFAULT_MODEL,
)

try:
    from embedding.embedding_layer import encode as embed_encode, decode as embed_decode
    _EMBED = True
except ImportError:
    _EMBED = False

try:
    from compressor.llmlingua_layer import compress as lingua_compress
    _LINGUA = True
except ImportError:
    _LINGUA = False


# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
CORS(app, origins=allowed_origins, supports_credentials=False)

# Allow iFrame embedding (remove X-Frame-Options restriction)
@app.after_request
def allow_iframe(response: Response) -> Response:
    response.headers.pop("X-Frame-Options", None)
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response


# ── Token counter ──────────────────────────────────────────────────────────
def _tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Return which backends and layers are available."""
    return jsonify({
        "layers": {
            "1": {"name": "Structural Skeleton", "available": True,  "requires": "none"},
            "2": {"name": "LLMLingua Pruning",   "available": _LINGUA, "requires": "pip install llmlingua"},
            "3": {"name": "Importance Filter",   "available": True,  "requires": "none"},
            "4": {"name": "Embedding Compress",  "available": _EMBED, "requires": "pip install sentence-transformers faiss-cpu"},
            "5": {"name": "Abstractive LLM",     "available": True,  "requires": "GROQ_API_KEY or GEMINI_API_KEY (optional)"},
        },
        "backends": {
            "groq":   {"available": bool(os.getenv("GROQ_API_KEY")),   "model": GROQ_DEFAULT_MODEL},
            "gemini": {"available": bool(os.getenv("GEMINI_API_KEY")), "model": GEMINI_DEFAULT_MODEL},
            "local":  {"available": True, "model": "bart-large-cnn (needs ~1.5GB download on first use)"},
        },
    })


@app.route("/api/compress", methods=["POST"])
def compress_endpoint():
    """
    POST /api/compress
    Body (JSON):
        text        : string — input text
        layers      : list[int] — e.g. [1, 2, 3]
        mode        : "conservative" | "balanced" | "aggressive"
        backend     : "auto" | "groq" | "gemini" | "hf"
        groq_key    : string (optional, overrides env)
        gemini_key  : string (optional, overrides env)
    """
    body = request.get_json(force=True)
    text      = body.get("text", "").strip()
    layers    = [int(l) for l in body.get("layers", [1, 3])]
    mode      = body.get("mode", "balanced")
    backend   = body.get("backend", "auto")

    # Allow keys passed from UI (store in env for this request only)
    if body.get("groq_key"):
        os.environ["GROQ_API_KEY"] = body["groq_key"]
    if body.get("gemini_key"):
        os.environ["GEMINI_API_KEY"] = body["gemini_key"]

    if not text:
        return jsonify({"error": "text is required"}), 400

    original_tokens = _tokens(text)
    current = text
    layer_stats = []
    t_total = time.perf_counter()

    keep_map   = {"conservative": 0.75, "balanced": 0.60, "aggressive": 0.40}
    ratio_map  = {"conservative": 0.50, "balanced": 0.35, "aggressive": 0.20}
    lingua_map = {"conservative": 0.70, "balanced": 0.50, "aggressive": 0.30}

    try:
        # Layer 1
        if 1 in layers:
            t = time.perf_counter()
            r = skeletonize(current)
            current = r.skeleton
            layer_stats.append({
                "layer": 1, "name": "Structural Skeleton",
                "input": r.original_tokens, "output": r.skeleton_tokens,
                "ratio": r.compression_ratio, "content_type": r.content_type,
                "ms": round((time.perf_counter() - t) * 1000),
            })

        # Layer 2
        if 2 in layers:
            if not _LINGUA:
                layer_stats.append({"layer": 2, "name": "LLMLingua", "skipped": True,
                                    "reason": "not installed — run: pip install llmlingua"})
            else:
                t = time.perf_counter()
                r = lingua_compress(current, target_token_rate=lingua_map[mode])
                current = r.compressed_text
                layer_stats.append({
                    "layer": 2, "name": "LLMLingua",
                    "input": r.original_tokens, "output": r.compressed_tokens,
                    "ratio": r.compression_ratio,
                    "ms": round((time.perf_counter() - t) * 1000),
                })

        # Layer 3
        if 3 in layers:
            t = time.perf_counter()
            r = prune(current, keep_ratio=keep_map[mode])
            current = r.pruned_text
            layer_stats.append({
                "layer": 3, "name": "Importance Filter",
                "input": r.original_tokens, "output": r.pruned_tokens,
                "ratio": r.compression_ratio, "removed": r.removed_sentences,
                "ms": round((time.perf_counter() - t) * 1000),
            })

        # Layer 4
        if 4 in layers:
            if not _EMBED:
                layer_stats.append({"layer": 4, "name": "Embedding", "skipped": True,
                                    "reason": "not installed — run: pip install sentence-transformers faiss-cpu"})
            else:
                t = time.perf_counter()
                packet = embed_encode(current)
                current = embed_decode(packet)
                layer_stats.append({
                    "layer": 4, "name": "Embedding Compress",
                    "chunks": len(packet.chunks),
                    "embed_shape": list(packet.embeddings.shape),
                    "ms": round((time.perf_counter() - t) * 1000),
                })

        # Layer 5
        if 5 in layers:
            t = time.perf_counter()
            r = abstractive_compress(current, target_ratio=ratio_map[mode], backend=backend)
            current = r.output_text
            layer_stats.append({
                "layer": 5, "name": "Abstractive LLM",
                "input": r.input_tokens, "output": r.output_tokens,
                "ratio": r.compression_ratio, "model": r.model_used, "backend": r.backend,
                "ms": round((time.perf_counter() - t) * 1000),
            })

    except Exception as exc:
        return jsonify({"error": str(exc), "layer_stats": layer_stats}), 500

    final_tokens = _tokens(current)
    return jsonify({
        "compressed": current,
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "total_ratio": round(original_tokens / max(final_tokens, 1), 2),
        "saved_pct": round((1 - final_tokens / original_tokens) * 100, 1),
        "total_ms": round((time.perf_counter() - t_total) * 1000),
        "layer_stats": layer_stats,
    })


@app.route("/api/decompress", methods=["POST"])
def decompress_endpoint():
    """
    POST /api/decompress
    Body (JSON):
        text      : string — compressed text
        backend   : "auto" | "groq" | "gemini" | "hf"
        groq_key  : string (optional)
        gemini_key: string (optional)
    """
    body = request.get_json(force=True)
    text    = body.get("text", "").strip()
    backend = body.get("backend", "auto")

    if body.get("groq_key"):
        os.environ["GROQ_API_KEY"] = body["groq_key"]
    if body.get("gemini_key"):
        os.environ["GEMINI_API_KEY"] = body["gemini_key"]

    if not text:
        return jsonify({"error": "text is required"}), 400

    t = time.perf_counter()
    try:
        result = abstractive_reconstruct(
            compressed=text,
            reconstruction_hints=[
                "Text was processed through structural skeletonization, "
                "importance filtering, and possible abstractive compression.",
            ],
            backend=backend,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "reconstructed": result.output_text,
        "input_tokens":  result.input_tokens,
        "output_tokens": result.output_tokens,
        "expansion_ratio": result.compression_ratio,
        "model": result.model_used,
        "backend": result.backend,
        "ms": round((time.perf_counter() - t) * 1000),
    })


@app.route("/api/benchmark", methods=["POST"])
def benchmark_endpoint():
    """Quick benchmark across modes for a given text."""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    original_tokens = _tokens(text)
    results = []

    for mode in ["conservative", "balanced", "aggressive"]:
        keep = {"conservative": 0.75, "balanced": 0.60, "aggressive": 0.40}[mode]
        t = time.perf_counter()
        r1 = skeletonize(text)
        r3 = prune(r1.skeleton, keep_ratio=keep)
        ms = round((time.perf_counter() - t) * 1000)
        final = _tokens(r3.pruned_text)
        results.append({
            "mode": mode,
            "layers": "1+3",
            "original_tokens": original_tokens,
            "final_tokens": final,
            "ratio": round(original_tokens / max(final, 1), 2),
            "saved_pct": round((1 - final / original_tokens) * 100, 1),
            "ms": ms,
        })

    return jsonify({"results": results})


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n🌐 Semantic Compression Web UI")
    print(f"   URL  : http://{args.host}:{args.port}")
    print(f"   Groq : {'✅ configured' if os.getenv('GROQ_API_KEY') else '⚠️  not set (Layer 5 disabled)'}")
    print(f"   Gemini: {'✅ configured' if os.getenv('GEMINI_API_KEY') else '⚠️  not set (Layer 5 disabled)'}")
    print(f"   iFrame: <iframe src=\"http://{args.host}:{args.port}\" ...></iframe>\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
