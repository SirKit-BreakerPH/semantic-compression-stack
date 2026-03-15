"""
decompress.py — Full Round-Trip Reconstruction Pipeline
--------------------------------------------------------
Usage:
    python pipelines/decompress.py compressed.txt [--embed compressed.emb.bin]
    python pipelines/decompress.py compressed.txt --output restored.txt

Architecture (reverse of compress.py):
    Layer 4 → Embedding decode (if .bin file provided)
    Layer 5 → LLM reconstruction (abstractive expansion)
    Layer 3 → (no reversal needed; expansion handled by LLM)
    Layer 2 → (no reversal; LLM fills gaps)
    Layer 1 → Skeleton expansion (LLM-guided)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from compressor.abstractive_layer import reconstruct as llm_reconstruct
from compressor.token_pruning_layer import restore_hint
from compressor.llmlingua_layer import decompress_hint

try:
    from embedding.embedding_layer import load as embed_load, decode as embed_decode
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False


@dataclass
class ReconstructionReport:
    input_file: str
    output_file: str
    compressed_tokens: int
    reconstructed_tokens: int
    expansion_ratio: float
    layers_reversed: list[str]
    elapsed_seconds: float


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


def reconstruct_pipeline(
    compressed_text: str,
    embed_path: Optional[str] = None,
    verbose: bool = True,
) -> tuple[str, list[str]]:
    """
    Reconstruct compressed text to near-original.

    Returns:
        (reconstructed_text, layers_reversed_list)
    """
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    current = compressed_text
    hints: list[str] = []
    layers_reversed: list[str] = []

    # ── Reverse Layer 4: Embedding decode ─────────────────────────────────
    if embed_path and _EMBED_AVAILABLE and Path(embed_path).exists():
        log("Reverse Layer 4 → Decoding embeddings...")
        t = time.perf_counter()
        packet = embed_load(embed_path)
        decoded = embed_decode(packet)
        if decoded and len(decoded) > len(current):
            current = decoded
            hints.append(f"Decoded from {len(packet.chunks)} semantic embedding chunks.")
        log(f"    Recovered {len(packet.chunks)} chunks ({time.perf_counter()-t:.1f}s)")
        layers_reversed.append("embedding")

    # Add reconstruction hints from prior layers
    hints.append(restore_hint(""))   # pruning hint
    hints.append(decompress_hint(""))  # lingua hint

    # ── Reverse Layers 2, 3, 5: LLM reconstruction ────────────────────────
    log("Reverse Layers 2–5 → LLM semantic reconstruction...")
    t = time.perf_counter()
    result = llm_reconstruct(
        compressed=current,
        reconstruction_hints=[
            "The input text was passed through multiple compression layers: "
            "structural skeletonization, semantic token pruning, importance filtering, "
            "and/or abstractive summarization. Reconstruct the full original document.",
            f"Original compressed content length: ~{len(current)} chars.",
        ],
    )
    current = result.output_text
    elapsed = time.perf_counter() - t
    log(f"    Reconstructed {result.input_tokens} → {result.output_tokens} tokens "
        f"({elapsed:.1f}s) [{result.model_used}]")
    layers_reversed.append("llm_reconstruction")

    return current, layers_reversed


def run(
    input_path: str,
    output_path: str,
    embed_path: Optional[str],
    report: bool,
    verbose: bool,
) -> ReconstructionReport:
    print(f"\n🔄 Semantic Reconstruction Pipeline")
    print(f"   Input : {input_path}")
    if embed_path:
        print(f"   Embeds: {embed_path}")
    print()

    compressed_text = Path(input_path).read_text(encoding="utf-8")
    compressed_tokens = _count_tokens(compressed_text)
    t_start = time.perf_counter()

    reconstructed, layers = reconstruct_pipeline(
        compressed_text=compressed_text,
        embed_path=embed_path,
        verbose=verbose,
    )

    Path(output_path).write_text(reconstructed, encoding="utf-8")
    reconstructed_tokens = _count_tokens(reconstructed)
    elapsed = time.perf_counter() - t_start

    rpt = ReconstructionReport(
        input_file=input_path,
        output_file=output_path,
        compressed_tokens=compressed_tokens,
        reconstructed_tokens=reconstructed_tokens,
        expansion_ratio=round(reconstructed_tokens / max(compressed_tokens, 1), 2),
        layers_reversed=layers,
        elapsed_seconds=round(elapsed, 2),
    )

    print(f"\n✅ Reconstruction complete")
    print(f"   {compressed_tokens:,} → {reconstructed_tokens:,} tokens ({rpt.expansion_ratio}× expansion)")
    print(f"   Elapsed: {elapsed:.1f}s")
    print(f"   Output : {output_path}")

    if report:
        rpt_path = output_path + ".report.json"
        Path(rpt_path).write_text(json.dumps(
            {
                "input_file": rpt.input_file,
                "output_file": rpt.output_file,
                "compressed_tokens": rpt.compressed_tokens,
                "reconstructed_tokens": rpt.reconstructed_tokens,
                "expansion_ratio": rpt.expansion_ratio,
                "layers_reversed": rpt.layers_reversed,
                "elapsed_seconds": rpt.elapsed_seconds,
            },
            indent=2,
        ))
        print(f"   Report : {rpt_path}")

    return rpt


def main():
    parser = argparse.ArgumentParser(description="Semantic Reconstruction Pipeline")
    parser.add_argument("input", help="Compressed text file path")
    parser.add_argument("--output", "-o", default=None, help="Output restored file path")
    parser.add_argument("--embed", "-e", default=None, help="Path to .emb.bin embedding file")
    parser.add_argument("--report", action="store_true", help="Save JSON report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress")
    args = parser.parse_args()

    output = args.output or args.input.replace(".compressed", "").rsplit(".", 1)[0] + ".restored.txt"

    run(
        input_path=args.input,
        output_path=output,
        embed_path=args.embed,
        report=args.report,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
