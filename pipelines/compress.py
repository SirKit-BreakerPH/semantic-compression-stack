"""
compress.py — Full 5-Layer Compression Pipeline
------------------------------------------------
Usage:
    python pipelines/compress.py input.txt [--output compressed.bin] [--report]
    python pipelines/compress.py input.txt --layers 1,2,3   # skip layers 4,5
    python pipelines/compress.py input.txt --mode aggressive

Architecture:
    Layer 1 → Structural skeletonization (PromptPacker-style)
    Layer 2 → Semantic token pruning (LLMLingua)
    Layer 3 → Importance filtering (imptokens-style)
    Layer 4 → Embedding compression (DragonMemory-style)
    Layer 5 → Abstractive compression (Cmprsr-style)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from structure.skeleton_layer import skeletonize
from compressor.token_pruning_layer import prune
from compressor.abstractive_layer import compress as abstractive_compress
from embedding.embedding_layer import encode as embed_encode, save as embed_save

try:
    from compressor.llmlingua_layer import compress as lingua_compress
    _LINGUA_AVAILABLE = True
except ImportError:
    _LINGUA_AVAILABLE = False


@dataclass
class CompressionReport:
    input_file: str
    output_file: str
    original_tokens: int
    final_tokens: int
    total_ratio: float
    layers_run: list[str]
    layer_stats: list[dict]
    elapsed_seconds: float
    mode: str


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


def compress_pipeline(
    text: str,
    layers: list[int],
    mode: str = "balanced",
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """
    Run the multi-layer compression pipeline.

    Args:
        text:    Input text.
        layers:  Which layers to run (e.g. [1,2,3]).
        mode:    "conservative" | "balanced" | "aggressive"
        verbose: Print progress.

    Returns:
        (compressed_text, layer_stats_list)
    """
    def log(msg): 
        if verbose:
            print(f"  {msg}", flush=True)

    stats: list[dict] = []
    current = text

    # ── Layer 1: Structural skeleton ──────────────────────────────────────
    if 1 in layers:
        log("Layer 1 → Structural skeletonization...")
        t = time.perf_counter()
        result = skeletonize(current)
        current = result.skeleton
        elapsed = time.perf_counter() - t
        stats.append({
            "layer": 1,
            "name": "structural",
            "input_tokens": result.original_tokens,
            "output_tokens": result.skeleton_tokens,
            "ratio": result.compression_ratio,
            "content_type": result.content_type,
            "elapsed_s": round(elapsed, 2),
        })
        log(f"    {result.original_tokens} → {result.skeleton_tokens} tokens ({result.compression_ratio}×)")

    # ── Layer 2: LLMLingua semantic pruning ───────────────────────────────
    if 2 in layers:
        if not _LINGUA_AVAILABLE:
            log("Layer 2 → LLMLingua not installed, skipping (pip install llmlingua)")
        else:
            rate = {"conservative": 0.7, "balanced": 0.5, "aggressive": 0.3}[mode]
            log(f"Layer 2 → LLMLingua semantic pruning (rate={rate})...")
            t = time.perf_counter()
            result = lingua_compress(current, target_token_rate=rate)
            current = result.compressed_text
            elapsed = time.perf_counter() - t
            stats.append({
                "layer": 2,
                "name": "llmlingua",
                "input_tokens": result.original_tokens,
                "output_tokens": result.compressed_tokens,
                "ratio": result.compression_ratio,
                "elapsed_s": round(elapsed, 2),
            })
            log(f"    {result.original_tokens} → {result.compressed_tokens} tokens ({result.compression_ratio}×)")

    # ── Layer 3: Token importance pruning ─────────────────────────────────
    if 3 in layers:
        keep = {"conservative": 0.75, "balanced": 0.6, "aggressive": 0.4}[mode]
        log(f"Layer 3 → Token importance filtering (keep={keep})...")
        t = time.perf_counter()
        result = prune(current, keep_ratio=keep)
        current = result.pruned_text
        elapsed = time.perf_counter() - t
        stats.append({
            "layer": 3,
            "name": "token_pruning",
            "input_tokens": result.original_tokens,
            "output_tokens": result.pruned_tokens,
            "ratio": result.compression_ratio,
            "removed_sentences": result.removed_sentences,
            "elapsed_s": round(elapsed, 2),
        })
        log(f"    {result.original_tokens} → {result.pruned_tokens} tokens ({result.compression_ratio}×)")

    # ── Layer 5: Abstractive compression (before embedding, uses text) ────
    if 5 in layers:
        ratio = {"conservative": 0.5, "balanced": 0.35, "aggressive": 0.2}[mode]
        log(f"Layer 5 → Abstractive compression (target_ratio={ratio})...")
        t = time.perf_counter()
        result = abstractive_compress(current, target_ratio=ratio)
        current = result.output_text
        elapsed = time.perf_counter() - t
        stats.append({
            "layer": 5,
            "name": "abstractive",
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "ratio": result.compression_ratio,
            "model": result.model_used,
            "elapsed_s": round(elapsed, 2),
        })
        log(f"    {result.input_tokens} → {result.output_tokens} tokens ({result.compression_ratio}×) [{result.model_used}]")

    return current, stats


def run(
    input_path: str,
    output_path: str,
    layers: list[int],
    mode: str,
    embed_output: Optional[str],
    report: bool,
    verbose: bool,
) -> CompressionReport:
    print(f"\n🗜  Semantic Compression Stack")
    print(f"   Input : {input_path}")
    print(f"   Layers: {layers}")
    print(f"   Mode  : {mode}\n")

    text = Path(input_path).read_text(encoding="utf-8")
    original_tokens = _count_tokens(text)
    t_start = time.perf_counter()

    compressed, stats = compress_pipeline(text, layers, mode, verbose)

    # ── Layer 4: Embedding compression (binary output) ────────────────────
    if 4 in layers:
        print(f"\n  Layer 4 → Embedding compression...")
        t = time.perf_counter()
        packet = embed_encode(compressed)
        embed_path = embed_output or output_path.replace(".txt", ".emb.bin")
        embed_save(packet, embed_path)
        elapsed = time.perf_counter() - t
        char_size = len(packet.to_bytes())
        stats.append({
            "layer": 4,
            "name": "embedding",
            "chunks": len(packet.chunks),
            "embed_dim": packet.embeddings.shape[1],
            "output_bytes": char_size,
            "elapsed_s": round(elapsed, 2),
        })
        print(f"    Saved {len(packet.chunks)} chunks → {embed_path} ({char_size:,} bytes)")

    # Save text output
    Path(output_path).write_text(compressed, encoding="utf-8")
    final_tokens = _count_tokens(compressed)
    elapsed_total = time.perf_counter() - t_start

    rpt = CompressionReport(
        input_file=input_path,
        output_file=output_path,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        total_ratio=round(original_tokens / max(final_tokens, 1), 2),
        layers_run=[s["name"] for s in stats],
        layer_stats=stats,
        elapsed_seconds=round(elapsed_total, 2),
        mode=mode,
    )

    print(f"\n✅ Compression complete")
    print(f"   {original_tokens:,} → {final_tokens:,} tokens  ({rpt.total_ratio}× reduction)")
    print(f"   Elapsed: {elapsed_total:.1f}s")
    print(f"   Output : {output_path}")

    if report:
        rpt_path = output_path + ".report.json"
        Path(rpt_path).write_text(json.dumps(asdict(rpt), indent=2))
        print(f"   Report : {rpt_path}")

    return rpt


def main():
    parser = argparse.ArgumentParser(description="Semantic Compression Pipeline")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument(
        "--layers", "-l", default="1,2,3,4,5",
        help="Comma-separated layers to run (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--mode", "-m", default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Compression aggressiveness"
    )
    parser.add_argument("--embed-output", default=None, help="Path for .bin embedding file")
    parser.add_argument("--report", action="store_true", help="Save JSON report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]
    output = args.output or args.input.rsplit(".", 1)[0] + ".compressed.txt"

    run(
        input_path=args.input,
        output_path=output,
        layers=layers,
        mode=args.mode,
        embed_output=args.embed_output,
        report=args.report,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
