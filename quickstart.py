#!/usr/bin/env python3
"""
quickstart.py — Zero-config demo of the compression stack
----------------------------------------------------------
Runs layers 1 and 3 (no models needed) on a built-in text sample
and shows you exactly what happens at each stage.

Usage:
    python quickstart.py
    python quickstart.py --input myfile.txt
    python quickstart.py --aggressive
"""

from __future__ import annotations
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Rich for pretty output (falls back to plain text) ─────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich import box
    console = Console()
    def header(title): console.print(Rule(f"[bold cyan]{title}[/]"))
    def panel(content, title, style="dim"): console.print(Panel(content, title=title, border_style=style))
    def stat(label, value, color="green"): console.print(f"  [{color}]{label}:[/] {value}")
    RICH = True
except ImportError:
    RICH = False
    def header(title): print(f"\n{'─'*60}\n {title}\n{'─'*60}")
    def panel(content, title, style=None): print(f"[{title}]\n{content}\n")
    def stat(label, value, color=None): print(f"  {label}: {value}")


DEMO_TEXT = """\
# Understanding LLM Context Compression

Large language models (LLMs) are limited by a fixed context window — the maximum
number of tokens they can process in a single call. As of 2024, even the largest
models top out at around 200,000 tokens, which may seem large but is easily
exhausted by codebases, long documents, or multi-turn conversations.

## The Core Problem

When your input exceeds the context limit, you have three options:
1. Truncate — simply cut off the content (high information loss)
2. Chunk — process in pieces and merge results (loses cross-chunk context)
3. Compress — reduce token count while preserving meaning (the best approach)

This stack implements option 3 using a multi-layer pipeline.

## The Solution: Multi-Layer Semantic Compression

```python
class CompressionPipeline:
    \"\"\"
    Orchestrates multiple compression layers for maximum reduction
    while preserving semantic fidelity.
    \"\"\"

    def __init__(self, layers: list[str], mode: str = "balanced"):
        self.layers = layers
        self.mode = mode
        self._stats: dict = {}

    def compress(self, text: str) -> CompressionResult:
        \"\"\"Apply all configured layers in sequence.\"\"\"
        current = text
        for layer_name in self.layers:
            layer = self._get_layer(layer_name)
            current = layer.apply(current)
            self._stats[layer_name] = layer.stats()
        return CompressionResult(text=current, stats=self._stats)

    def decompress(self, compressed: str) -> str:
        \"\"\"Reconstruct original document from compressed form.\"\"\"
        return self._reconstruction_model.expand(compressed)
```

## Practical Results

In benchmarks on mixed technical documents (prose + code), this pipeline achieves:
- Layer 1 (structural): 1.5–2× compression in under 5ms
- Layer 3 (pruning): additional 1.5–2× in under 10ms
- Combined L1+L3: 2–5× with zero external dependencies

Adding LLMLingua (Layer 2) and abstractive compression (Layer 5) extends
the total reduction to 10–40× at the cost of requiring model inference.
The decompression pipeline achieves 90–99% semantic recovery at balanced settings.
""".strip()


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split())


def run_demo(text: str, mode: str = "balanced") -> None:
    from structure.skeleton_layer import skeletonize
    from compressor.token_pruning_layer import prune

    keep = {"conservative": 0.75, "balanced": 0.60, "aggressive": 0.40}[mode]

    if RICH:
        console.print()
        console.print(Panel.fit(
            "[bold]Semantic Compression Stack[/] — quickstart demo\n"
            "[dim]Layers 1+3 · No GPU · No API key · ~instant[/]",
            border_style="cyan"
        ))
        console.print()

    # ── Original ──────────────────────────────────────────────────────────
    orig_tokens = _count_tokens(text)
    header("ORIGINAL INPUT")
    panel(textwrap.shorten(text, 400, placeholder=" [...]"), f"{orig_tokens} tokens")

    # ── Layer 1 ───────────────────────────────────────────────────────────
    import time
    t = time.perf_counter()
    r1 = skeletonize(text)
    ms1 = (time.perf_counter() - t) * 1000

    header(f"LAYER 1 — Structural Skeleton  [{r1.content_type}]")
    stat("Tokens", f"{r1.original_tokens:,} → {r1.skeleton_tokens:,}")
    stat("Ratio", f"{r1.compression_ratio}×")
    stat("Time", f"{ms1:.0f}ms")
    print()
    panel(textwrap.shorten(r1.skeleton, 600, placeholder="\n[...truncated for display...]"), "Skeleton output")

    # ── Layer 3 ───────────────────────────────────────────────────────────
    t = time.perf_counter()
    r3 = prune(r1.skeleton, keep_ratio=keep)
    ms3 = (time.perf_counter() - t) * 1000

    header(f"LAYER 3 — Importance Filtering  [keep={keep}, mode={mode}]")
    stat("Tokens", f"{r3.original_tokens:,} → {r3.pruned_tokens:,}")
    stat("Ratio", f"{r3.compression_ratio}×")
    stat("Removed sentences", str(r3.removed_sentences))
    stat("Time", f"{ms3:.0f}ms")
    print()
    panel(r3.pruned_text, "Pruned output")

    # ── Summary ───────────────────────────────────────────────────────────
    total_ratio = round(orig_tokens / max(r3.pruned_tokens, 1), 2)
    total_ms = ms1 + ms3
    saved_pct = round((1 - r3.pruned_tokens / orig_tokens) * 100, 1)

    header("SUMMARY")

    if RICH:
        table = Table(box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Original tokens", f"{orig_tokens:,}")
        table.add_row("Compressed tokens", f"{r3.pruned_tokens:,}")
        table.add_row("Total ratio", f"[green]{total_ratio}×[/]")
        table.add_row("Tokens saved", f"[green]{saved_pct}%[/]")
        table.add_row("Total time", f"{total_ms:.0f}ms")
        table.add_row("Mode", mode)
        console.print(table)
    else:
        print(f"  Original:   {orig_tokens:,} tokens")
        print(f"  Compressed: {r3.pruned_tokens:,} tokens")
        print(f"  Ratio:      {total_ratio}×")
        print(f"  Saved:      {saved_pct}%")
        print(f"  Time:       {total_ms:.0f}ms")

    print()
    if RICH:
        console.print(Panel(
            "[bold green]✅ Done![/] To compress your own files:\n\n"
            "  [cyan]python pipelines/compress.py myfile.txt --layers 1,2,3 --mode balanced[/]\n\n"
            "To benchmark all modes:\n\n"
            "  [cyan]python benchmark.py --matrix[/]",
            border_style="green",
        ))
    else:
        print("✅ Done! Next steps:")
        print("  python pipelines/compress.py myfile.txt --layers 1,2,3 --mode balanced")
        print("  python benchmark.py --matrix")
    print()


def main():
    parser = argparse.ArgumentParser(description="Semantic Compression Quickstart Demo")
    parser.add_argument("--input", "-i", default=None, help="File to compress (uses built-in demo if omitted)")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive compression mode")
    parser.add_argument("--conservative", action="store_true", help="Use conservative compression mode")
    args = parser.parse_args()

    mode = "balanced"
    if args.aggressive:
        mode = "aggressive"
    elif args.conservative:
        mode = "conservative"

    text = DEMO_TEXT
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")

    run_demo(text, mode=mode)


if __name__ == "__main__":
    main()
