"""
benchmark.py — Compression Benchmark Tool
------------------------------------------
Measures compression ratio, speed, and (optionally) semantic similarity
across all layers and modes.

Usage:
    python benchmark.py                        # benchmark sample data
    python benchmark.py --input myfile.txt
    python benchmark.py --semantic             # compute BLEU/cosine scores too
    python benchmark.py --matrix               # all modes × layer combos
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from structure.skeleton_layer import skeletonize
from compressor.token_pruning_layer import prune


# ── Optional heavy imports ─────────────────────────────────────────────────
def _try_import(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        return None


# ── Token counter ──────────────────────────────────────────────────────────
def _tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


# ── Semantic similarity (cosine of sentence embeddings) ───────────────────
def _cosine_sim(a: str, b: str) -> Optional[float]:
    st = _try_import("sentence_transformers")
    np = _try_import("numpy")
    if not st or not np:
        return None
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ea, eb = model.encode([a[:512], b[:512]])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


@dataclass
class BenchResult:
    label: str
    layers: str
    mode: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    elapsed_ms: float
    semantic_sim: Optional[float]


def _run_layers_1_3(text: str, mode: str) -> tuple[str, float]:
    """Run layers 1 and 3 with the given mode settings."""
    keep = {"conservative": 0.75, "balanced": 0.60, "aggressive": 0.40}[mode]
    t = time.perf_counter()
    r1 = skeletonize(text)
    r3 = prune(r1.skeleton, keep_ratio=keep)
    elapsed = (time.perf_counter() - t) * 1000
    return r3.pruned_text, elapsed


def _run_layer_1_only(text: str) -> tuple[str, float]:
    t = time.perf_counter()
    r1 = skeletonize(text)
    elapsed = (time.perf_counter() - t) * 1000
    return r1.skeleton, elapsed


def _run_layer_3_only(text: str, mode: str) -> tuple[str, float]:
    keep = {"conservative": 0.75, "balanced": 0.60, "aggressive": 0.40}[mode]
    t = time.perf_counter()
    r3 = prune(text, keep_ratio=keep)
    elapsed = (time.perf_counter() - t) * 1000
    return r3.pruned_text, elapsed


def benchmark(
    text: str,
    semantic: bool = False,
    matrix: bool = False,
) -> list[BenchResult]:
    results: list[BenchResult] = []
    original_tokens = _tokens(text)

    configs: list[tuple[str, str, callable]] = []

    if matrix:
        for mode in ["conservative", "balanced", "aggressive"]:
            configs.append((f"L1 only", "1", lambda t, m=mode: _run_layer_1_only(t)))
            configs.append((f"L3 only [{mode}]", "3", lambda t, m=mode: _run_layer_3_only(t, m)))
            configs.append((f"L1+L3 [{mode}]", "1,3", lambda t, m=mode: _run_layers_1_3(t, m)))
    else:
        configs = [
            ("L1 only", "1", lambda t: _run_layer_1_only(t)),
            ("L3 only [balanced]", "3", lambda t: _run_layer_3_only(t, "balanced")),
            ("L1+L3 [conservative]", "1,3", lambda t: _run_layers_1_3(t, "conservative")),
            ("L1+L3 [balanced]", "1,3", lambda t: _run_layers_1_3(t, "balanced")),
            ("L1+L3 [aggressive]", "1,3", lambda t: _run_layers_1_3(t, "aggressive")),
        ]

    for label, layers, fn in configs:
        compressed, elapsed_ms = fn(text)
        comp_tokens = _tokens(compressed)
        ratio = round(original_tokens / max(comp_tokens, 1), 2)
        sim = _cosine_sim(text, compressed) if semantic else None

        results.append(BenchResult(
            label=label,
            layers=layers,
            mode=label,
            original_tokens=original_tokens,
            compressed_tokens=comp_tokens,
            ratio=ratio,
            elapsed_ms=round(elapsed_ms, 1),
            semantic_sim=sim,
        ))

    return results


def _print_table(results: list[BenchResult], input_name: str) -> None:
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(
            title=f"Compression Benchmark — {input_name}",
            box=box.ROUNDED,
            show_lines=True,
        )

        table.add_column("Config", style="cyan", no_wrap=True)
        table.add_column("Orig tokens", justify="right")
        table.add_column("Comp tokens", justify="right")
        table.add_column("Ratio", justify="right", style="green bold")
        table.add_column("Time (ms)", justify="right", style="yellow")
        if results[0].semantic_sim is not None:
            table.add_column("Sem. sim", justify="right", style="magenta")

        for r in results:
            row = [
                r.label,
                f"{r.original_tokens:,}",
                f"{r.compressed_tokens:,}",
                f"{r.ratio}×",
                f"{r.elapsed_ms:.0f}ms",
            ]
            if r.semantic_sim is not None:
                row.append(f"{r.semantic_sim:.3f}")
            table.add_row(*row)

        console.print(table)

    except ImportError:
        # Plain-text fallback
        header = f"\n{'Config':<30} {'Orig':>8} {'Comp':>8} {'Ratio':>8} {'Time':>10}"
        if results[0].semantic_sim is not None:
            header += f" {'Sim':>7}"
        print(header)
        print("─" * len(header))
        for r in results:
            line = f"{r.label:<30} {r.original_tokens:>8,} {r.compressed_tokens:>8,} {r.ratio:>7}× {r.elapsed_ms:>8.0f}ms"
            if r.semantic_sim is not None:
                line += f" {r.semantic_sim:>7.3f}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Compression Benchmark")
    parser.add_argument("--input", "-i", default=None, help="File to benchmark (default: sample_data/)")
    parser.add_argument("--semantic", "-s", action="store_true", help="Compute semantic similarity (requires sentence-transformers)")
    parser.add_argument("--matrix", "-m", action="store_true", help="Benchmark all mode × layer combinations")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    if args.input:
        files = [Path(args.input)]
    else:
        files = sorted(Path("sample_data").glob("*.txt"))
        if not files:
            print("No .txt files in sample_data/. Provide --input <file>.")
            sys.exit(1)

    all_results = []
    for fpath in files:
        text = fpath.read_text(encoding="utf-8")
        print(f"\n📄 Benchmarking: {fpath.name} ({len(text.split()):,} tokens)")
        results = benchmark(text, semantic=args.semantic, matrix=args.matrix)
        _print_table(results, fpath.name)
        all_results.extend(results)

    if args.json:
        import json
        from dataclasses import asdict
        print(json.dumps([asdict(r) for r in all_results], indent=2))


if __name__ == "__main__":
    main()
