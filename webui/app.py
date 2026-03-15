"""
webui/app.py — Flask Web Server for Semantic Compression Stack
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request, Response, send_file
from flask_cors import CORS
import io

from structure.skeleton_layer import skeletonize
from compressor.token_pruning_layer import prune, prune_safe as prune
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
try:
    from compressor.tbsa_layer import compress as tbsa_compress
    _TBSA = True
except ImportError:
    _TBSA = False

try:
    from compressor.dlm_layer import compress as dlm_compress, decompress as dlm_decompress
    _DLM = True
except ImportError:
    _DLM = False


# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*"), supports_credentials=False)


@app.after_request
def allow_iframe(response: Response) -> Response:
    response.headers.pop("X-Frame-Options", None)
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response


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
    return jsonify({
        "layers": {
            "1": {"name": "Structural Skeleton", "available": True,  "requires": "none"},
            "2": {"name": "LLMLingua Pruning",   "available": _LINGUA, "requires": "pip install llmlingua"},
            "3": {"name": "Importance Filter",   "available": True,  "requires": "none"},
            "4": {"name": "Embedding Compress",  "available": _EMBED, "requires": "pip install sentence-transformers faiss-cpu"},
            "5": {"name": "Abstractive LLM",     "available": True,  "requires": "GROQ_API_KEY or GEMINI_API_KEY"},
            "6": {"name": "TBSA Structural",      "available": True,  "requires": "none"},
            "7": {"name": "DLM Lexicon Mapping",  "available": True,  "requires": "none"},
        },
        "backends": {
            "groq":   {"available": bool(os.getenv("GROQ_API_KEY")),   "model": GROQ_DEFAULT_MODEL},
            "gemini": {"available": bool(os.getenv("GEMINI_API_KEY")), "model": GEMINI_DEFAULT_MODEL},
            "local":  {"available": True, "model": "bart-large-cnn"},
        },
    })


# ── File upload ────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Accept a text file and return its content."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html", ".yaml", ".yml"}
    ext = Path(f.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Type '{ext}' not supported. Use: {', '.join(sorted(allowed))}"}), 400

    try:
        text = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    return jsonify({
        "text": text,
        "filename": f.filename,
        "tokens": _tokens(text),
        "chars": len(text),
    })


# ── File download ──────────────────────────────────────────────────────────
@app.route("/api/download", methods=["POST"])
def download_file():
    """Return text as a downloadable file."""
    body     = request.get_json(force=True)
    text     = body.get("text", "")
    filename = body.get("filename", "output.txt")
    buf = io.BytesIO(text.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="text/plain", as_attachment=True, download_name=filename)


# ── Compress ───────────────────────────────────────────────────────────────
@app.route("/api/compress", methods=["POST"])
def compress_endpoint():
    body    = request.get_json(force=True)
    text    = body.get("text", "").strip()
    layers  = [int(l) for l in body.get("layers", [1, 3])]
    mode    = body.get("mode", "balanced")
    backend = body.get("backend", "auto")

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

    # Layer 1
    if 1 in layers:
        try:
            t = time.perf_counter()
            r = skeletonize(current)
            current = r.skeleton
            layer_stats.append({
                "layer": 1, "name": "Structural Skeleton",
                "input": r.original_tokens, "output": r.skeleton_tokens,
                "ratio": r.compression_ratio, "content_type": r.content_type,
                "ms": round((time.perf_counter() - t) * 1000),
            })
        except Exception as e:
            layer_stats.append({"layer": 1, "name": "Structural Skeleton",
                                 "skipped": True, "reason": str(e)[:150]})

    # Layer 2
    if 2 in layers:
        if not _LINGUA:
            layer_stats.append({"layer": 2, "name": "LLMLingua", "skipped": True,
                                 "reason": "not installed — pip install llmlingua"})
        else:
            try:
                t = time.perf_counter()
                r = lingua_compress(current, target_token_rate=lingua_map[mode])
                current = r.compressed_text
                layer_stats.append({
                    "layer": 2, "name": "LLMLingua",
                    "input": r.original_tokens, "output": r.compressed_tokens,
                    "ratio": r.compression_ratio,
                    "ms": round((time.perf_counter() - t) * 1000),
                })
            except Exception as e:
                layer_stats.append({"layer": 2, "name": "LLMLingua",
                                     "skipped": True, "reason": str(e)[:150]})

    # Layer 3
    if 3 in layers:
        try:
            t = time.perf_counter()
            r = prune(current, keep_ratio=keep_map[mode])
            current = r.pruned_text
            layer_stats.append({
                "layer": 3, "name": "Importance Filter",
                "input": r.original_tokens, "output": r.pruned_tokens,
                "ratio": r.compression_ratio, "removed": r.removed_sentences,
                "ms": round((time.perf_counter() - t) * 1000),
            })
        except Exception as e:
            layer_stats.append({"layer": 3, "name": "Importance Filter",
                                 "skipped": True, "reason": str(e)[:150]})

    # Layer 4 — wrapped individually, non-fatal on error
    if 4 in layers:
        if not _EMBED:
            layer_stats.append({"layer": 4, "name": "Embedding Compress", "skipped": True,
                                 "reason": "not installed — pip install sentence-transformers faiss-cpu"})
        else:
            try:
                t = time.perf_counter()
                packet  = embed_encode(current)
                current = embed_decode(packet)
                layer_stats.append({
                    "layer": 4, "name": "Embedding Compress",
                    "chunks": len(packet.chunks),
                    "embed_shape": list(packet.embeddings.shape),
                    "ms": round((time.perf_counter() - t) * 1000),
                })
            except Exception as e:
                layer_stats.append({"layer": 4, "name": "Embedding Compress",
                                     "skipped": True, "reason": str(e)[:150]})

    # Layer 5
    if 5 in layers:
        try:
            t = time.perf_counter()
            r = abstractive_compress(current, target_ratio=ratio_map[mode], backend=backend)
            current = r.output_text
            layer_stats.append({
                "layer": 5, "name": "Abstractive LLM",
                "input": r.input_tokens, "output": r.output_tokens,
                "ratio": r.compression_ratio, "model": r.model_used, "backend": r.backend,
                "ms": round((time.perf_counter() - t) * 1000),
            })
        except Exception as e:
            layer_stats.append({"layer": 5, "name": "Abstractive LLM",
                                 "skipped": True, "reason": str(e)[:200]})

    # Layer 6: TBSA
    if 6 in layers:
        try:
            t = time.perf_counter()
            aggr = {"conservative": 0.3, "balanced": 0.6, "aggressive": 0.85}[mode]
            r = tbsa_compress(current, aggressiveness=aggr)
            current = r.compressed_text
            layer_stats.append({"layer": 6, "name": "TBSA Structural",
                "input": r.original_tokens, "output": r.compressed_tokens,
                "ratio": r.compression_ratio, "ms": round((time.perf_counter()-t)*1000)})
        except Exception as e:
            layer_stats.append({"layer": 6, "name": "TBSA Structural",
                "skipped": True, "reason": str(e)[:150]})

    # Layer 7: DLM
    if 7 in layers:
        try:
            t = time.perf_counter()
            r = dlm_compress(current)
            current = r.compressed_text
            layer_stats.append({"layer": 7, "name": "DLM Lexicon Mapping",
                "input": r.original_tokens, "output": r.compressed_tokens,
                "ratio": r.compression_ratio, "replacements": r.replacements_made,
                "ms": round((time.perf_counter()-t)*1000)})
        except Exception as e:
            layer_stats.append({"layer": 7, "name": "DLM Lexicon Mapping",
                "skipped": True, "reason": str(e)[:150]})

    final_tokens = _tokens(current)
    return jsonify({
        "compressed": current,
        "original_tokens": original_tokens,
        "final_tokens": final_tokens,
        "total_ratio": round(original_tokens / max(final_tokens, 1), 2),
        "saved_pct": round((1 - final_tokens / max(original_tokens, 1)) * 100, 1),
        "total_ms": round((time.perf_counter() - t_total) * 1000),
        "layer_stats": layer_stats,
    })


# ── Decompress ─────────────────────────────────────────────────────────────
@app.route("/api/decompress", methods=["POST"])
def decompress_endpoint():
    body    = request.get_json(force=True)
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
            reconstruction_hints=["Text passed through structural + importance compression."],
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


# ── Benchmark ──────────────────────────────────────────────────────────────
@app.route("/api/benchmark", methods=["POST"])
def benchmark_endpoint():
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    original_tokens = _tokens(text)
    results = []
    for mode in ["conservative", "balanced", "aggressive"]:
        keep = {"conservative": 0.75, "balanced": 0.60, "aggressive": 0.40}[mode]
        t = time.perf_counter()
        try:
            r1    = skeletonize(text)
            r3    = prune(r1.skeleton, keep_ratio=keep)
            ms    = round((time.perf_counter() - t) * 1000)
            final = _tokens(r3.pruned_text)
            results.append({
                "mode": mode, "layers": "1+3",
                "original_tokens": original_tokens,
                "final_tokens": final,
                "ratio": round(original_tokens / max(final, 1), 2),
                "saved_pct": round((1 - final / max(original_tokens, 1)) * 100, 1),
                "ms": ms,
            })
        except Exception as e:
            results.append({"mode": mode, "error": str(e)})

    return jsonify({"results": results})


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n🌐 Semantic Compression Web UI")
    print(f"   URL   : http://{args.host}:{args.port}")
    print(f"   Groq  : {'✅ configured' if os.getenv('GROQ_API_KEY') else '⚠️  not set'}")
    print(f"   Gemini: {'✅ configured' if os.getenv('GEMINI_API_KEY') else '⚠️  not set'}")
    print(f"   Upload: up to 10 MB\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
# TBSA and DLM imports appended
