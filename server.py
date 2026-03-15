"""
server.py — Flask Web Server for Semantic Compression Stack
------------------------------------------------------------
Serves the UI and exposes the pipeline as a REST API.
WordPress (or any site) can embed it via <iframe>.

Usage:
    python server.py                   # default: http://localhost:5000
    python server.py --port 8080
    python server.py --host 0.0.0.0    # expose to network (for Codespaces)

Endpoints:
    GET  /                  → UI (index.html)
    POST /api/compress      → Run compression pipeline
    POST /api/decompress    → Run reconstruction pipeline
    GET  /api/health        → Health check / available layers
"""

from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, request, jsonify, render_template, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Run: pip install flask flask-cors")
    sys.exit(1)

# ── Import compression layers (graceful if not installed) ──────────────────
from structure.skeleton_layer import skeletonize
from compressor.token_pruning_layer import prune

try:
    from compressor.llmlingua_layer import compress as lingua_compress
    LINGUA_OK = True
except Exception:
    LINGUA_OK = False

try:
    from embedding.embedding_layer import encode as embed_encode, decode as embed_decode
    EMBED_OK = True
except Exception:
    EMBED_OK = False

try:
    from compressor.abstractive_layer import compress as abstract_compress, reconstruct as abstract_reconstruct
    ABSTRACT_OK = True
except Exception:
    ABSTRACT_OK = False


# ──────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Allow WordPress iframe embedding from any origin

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


MODE_SETTINGS = {
    "conservative": {"lingua_rate": 0.7, "prune_keep": 0.75, "abstract_ratio": 0.5},
    "balanced":     {"lingua_rate": 0.5, "prune_keep": 0.60, "abstract_ratio": 0.35},
    "aggressive":   {"lingua_rate": 0.3, "prune_keep": 0.40, "abstract_ratio": 0.20},
}


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "layers": {
            "1_structural":   True,
            "2_llmlingua":    LINGUA_OK,
            "3_pruning":      True,
            "4_embedding":    EMBED_OK,
            "5_abstractive":  ABSTRACT_OK,
        }
    })


@app.route("/api/compress", methods=["POST"])
def compress():
    try:
        body = request.get_json()
        text    = body.get("text", "").strip()
        layers  = [int(l) for l in body.get("layers", [1, 3])]
        mode    = body.get("mode", "balanced")
        if mode not in MODE_SETTINGS:
            mode = "balanced"
        cfg = MODE_SETTINGS[mode]

        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text) > 200_000:
            return jsonify({"error": "Text too large (max 200,000 chars)"}), 400

        original_tokens = _count_tokens(text)
        current = text
        layer_stats = []
        t_total = time.perf_counter()

        # Layer 1
        if 1 in layers:
            t = time.perf_counter()
            r = skeletonize(current)
            current = r.skeleton
            layer_stats.append({
                "layer": 1, "name": "Structural Skeleton",
                "input": r.original_tokens, "output": r.skeleton_tokens,
                "ratio": r.compression_ratio, "ms": round((time.perf_counter()-t)*1000, 1),
                "detail": r.content_type,
            })

        # Layer 2
        if 2 in layers:
            if not LINGUA_OK:
                layer_stats.append({"layer": 2, "name": "LLMLingua", "skipped": True,
                                    "reason": "pip install llmlingua"})
            else:
                t = time.perf_counter()
                r = lingua_compress(current, target_token_rate=cfg["lingua_rate"])
                current = r.compressed_text
                layer_stats.append({
                    "layer": 2, "name": "LLMLingua",
                    "input": r.original_tokens, "output": r.compressed_tokens,
                    "ratio": r.compression_ratio, "ms": round((time.perf_counter()-t)*1000, 1),
                })

        # Layer 3
        if 3 in layers:
            t = time.perf_counter()
            r = prune(current, keep_ratio=cfg["prune_keep"])
            current = r.pruned_text
            layer_stats.append({
                "layer": 3, "name": "Importance Filtering",
                "input": r.original_tokens, "output": r.pruned_tokens,
                "ratio": r.compression_ratio, "ms": round((time.perf_counter()-t)*1000, 1),
                "detail": f"{r.removed_sentences} sentences removed",
            })

        # Layer 4
        if 4 in layers:
            if not EMBED_OK:
                layer_stats.append({"layer": 4, "name": "Embedding", "skipped": True,
                                    "reason": "pip install sentence-transformers faiss-cpu"})
            else:
                t = time.perf_counter()
                packet = embed_encode(current)
                current = embed_decode(packet)
                layer_stats.append({
                    "layer": 4, "name": "Embedding Compression",
                    "input": _count_tokens(text), "output": len(packet.chunks),
                    "ratio": round(len(text)/max(len(packet.to_bytes()), 1)*100, 1),
                    "ms": round((time.perf_counter()-t)*1000, 1),
                    "detail": f"{len(packet.chunks)} chunks",
                })

        # Layer 5
        if 5 in layers:
            if not ABSTRACT_OK:
                layer_stats.append({"layer": 5, "name": "Abstractive", "skipped": True,
                                    "reason": "pip install transformers torch"})
            else:
                t = time.perf_counter()
                r = abstract_compress(current, target_ratio=cfg["abstract_ratio"])
                current = r.output_text
                layer_stats.append({
                    "layer": 5, "name": "Abstractive (LLM)",
                    "input": r.input_tokens, "output": r.output_tokens,
                    "ratio": r.compression_ratio, "ms": round((time.perf_counter()-t)*1000, 1),
                    "detail": r.model_used,
                })

        final_tokens = _count_tokens(current)
        total_ratio = round(original_tokens / max(final_tokens, 1), 2)

        return jsonify({
            "compressed": current,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "total_ratio": total_ratio,
            "saved_pct": round((1 - final_tokens / original_tokens) * 100, 1),
            "elapsed_ms": round((time.perf_counter() - t_total) * 1000, 1),
            "layer_stats": layer_stats,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/decompress", methods=["POST"])
def decompress():
    try:
        body = request.get_json()
        text = body.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        if not ABSTRACT_OK:
            return jsonify({"error": "Reconstruction requires: pip install transformers torch"}), 400

        t = time.perf_counter()
        r = abstract_reconstruct(text)
        elapsed = round((time.perf_counter() - t) * 1000, 1)

        return jsonify({
            "reconstructed": r.output_text,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "expansion_ratio": r.compression_ratio,
            "elapsed_ms": elapsed,
            "model": r.model_used,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n🗜  Semantic Compression Server")
    print(f"   URL  : http://{args.host}:{args.port}")
    print(f"   L1+L3: ✅ ready")
    print(f"   L2 LLMLingua:   {'✅' if LINGUA_OK else '⚠️  pip install llmlingua'}")
    print(f"   L4 Embeddings:  {'✅' if EMBED_OK  else '⚠️  pip install sentence-transformers faiss-cpu'}")
    print(f"   L5 Abstractive: {'✅' if ABSTRACT_OK else '⚠️  pip install transformers torch'}")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
