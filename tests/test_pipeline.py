"""
tests/test_pipeline.py — Unit tests for each compression layer
--------------------------------------------------------------
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Sample data ────────────────────────────────────────────────────────────

SAMPLE_PROSE = """
Artificial intelligence has undergone remarkable transformation over the past decade.
Deep learning models, particularly large language models, have demonstrated capabilities
that were previously considered exclusively human. The development of transformer architectures
in 2017 marked a turning point that enabled training on unprecedented scales of data.

Modern AI systems can now perform complex reasoning, generate creative content, and solve
mathematical problems at expert level. However, significant challenges remain around
alignment, interpretability, and computational efficiency. Researchers are actively working
on solutions to these fundamental problems.

The economic impact of AI is substantial and growing. Industries from healthcare to finance
are being transformed by automated systems that can process vast amounts of data in real time.
Productivity gains of 20-40% have been reported in some sectors, while concerns about
displacement of workers have prompted policy discussions in many countries.
""".strip()

SAMPLE_CODE = """
import os
import json
from pathlib import Path
from typing import Optional, List

class DocumentProcessor:
    \"\"\"Processes and compresses documents for LLM context.\"\"\"
    
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 8192):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._cache: dict = {}
    
    def load_document(self, path: str) -> str:
        \"\"\"Load a document from disk, with caching.\"\"\"
        if path in self._cache:
            return self._cache[path]
        content = Path(path).read_text(encoding="utf-8")
        self._cache[path] = content
        return content
    
    def process_batch(self, paths: List[str]) -> List[str]:
        \"\"\"Process multiple documents in parallel.\"\"\"
        results = []
        for p in paths:
            doc = self.load_document(p)
            compressed = self._compress(doc)
            results.append(compressed)
        return results
    
    def _compress(self, text: str) -> str:
        \"\"\"Internal compression logic.\"\"\"
        # Apply semantic compression
        tokens = text.split()
        filtered = [t for t in tokens if len(t) > 2]
        return " ".join(filtered[:self.max_tokens])
    
    def save_result(self, result: str, output_path: str) -> None:
        \"\"\"Save processed result to disk.\"\"\"
        Path(output_path).write_text(result, encoding="utf-8")
        print(f"Saved {len(result)} chars to {output_path}")
""".strip()


# ── Layer 1: Structural ────────────────────────────────────────────────────
def test_skeleton_prose():
    from structure.skeleton_layer import skeletonize
    result = skeletonize(SAMPLE_PROSE)
    assert result.skeleton_tokens < result.original_tokens
    assert result.compression_ratio >= 1.0
    assert result.content_type == "prose"
    assert len(result.skeleton) > 50  # not empty
    print(f"  Prose: {result.original_tokens} → {result.skeleton_tokens} tokens ({result.compression_ratio}×)")


def test_skeleton_code():
    from structure.skeleton_layer import skeletonize
    result = skeletonize(SAMPLE_CODE)
    assert result.skeleton_tokens < result.original_tokens
    assert result.content_type == "python"
    assert "DocumentProcessor" in result.skeleton
    assert "load_document" in result.skeleton
    print(f"  Code: {result.original_tokens} → {result.skeleton_tokens} tokens ({result.compression_ratio}×)")


# ── Layer 3: Token pruning ─────────────────────────────────────────────────
def test_token_pruning():
    from compressor.token_pruning_layer import prune
    result = prune(SAMPLE_PROSE, keep_ratio=0.6)
    assert result.pruned_tokens < result.original_tokens
    assert result.compression_ratio >= 1.0
    assert len(result.pruned_text) > 50
    print(f"  Pruning: {result.original_tokens} → {result.pruned_tokens} tokens ({result.compression_ratio}×)")


def test_token_pruning_aggressive():
    from compressor.token_pruning_layer import prune
    result = prune(SAMPLE_PROSE, keep_ratio=0.3)
    assert result.compression_ratio > 1.5
    print(f"  Aggressive pruning: {result.compression_ratio}×")


# ── Layer 4: Embeddings ────────────────────────────────────────────────────
def test_embedding_roundtrip():
    from embedding.embedding_layer import encode, decode
    packet = encode(SAMPLE_PROSE, chunk_tokens=64, pca_components=32, quantize=True)
    assert len(packet.chunks) > 0
    assert packet.embeddings is not None
    restored = decode(packet)
    assert len(restored) > 50
    # Check that key words are preserved in chunks
    assert any("artificial intelligence" in c.lower() or "deep learning" in c.lower()
               for c in packet.chunks)
    print(f"  Embedding: {len(packet.chunks)} chunks, {packet.embeddings.shape} shape")


def test_embedding_serialization(tmp_path):
    from embedding.embedding_layer import encode, decode, save, load
    packet = encode(SAMPLE_PROSE[:200])
    path = tmp_path / "test.emb.bin"
    save(packet, path)
    loaded = load(path)
    restored = decode(loaded)
    assert len(restored) > 10
    print(f"  Serialization: {path.stat().st_size:,} bytes")


# ── Full pipeline (layers 1+3 only, no LLM required) ─────────────────────
def test_pipeline_no_llm():
    from structure.skeleton_layer import skeletonize
    from compressor.token_pruning_layer import prune

    original_tokens = len(SAMPLE_PROSE.split())

    r1 = skeletonize(SAMPLE_PROSE)
    r3 = prune(r1.skeleton, keep_ratio=0.6)

    final_tokens = r3.pruned_tokens
    total_ratio = original_tokens / max(final_tokens, 1)

    assert total_ratio >= 1.5, f"Expected >1.5× compression, got {total_ratio}"
    print(f"\n  End-to-end (L1+L3): {original_tokens} → {final_tokens} tokens ({total_ratio:.1f}×)")


# ── Run standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    tests = [
        test_skeleton_prose,
        test_skeleton_code,
        test_token_pruning,
        test_token_pruning_aggressive,
        test_embedding_roundtrip,
        test_pipeline_no_llm,
    ]
    passed = 0
    for test in tests:
        name = test.__name__
        try:
            # Handle tmp_path for serialization test
            if "serialization" in name:
                import tempfile, pathlib
                with tempfile.TemporaryDirectory() as d:
                    test(pathlib.Path(d))
            else:
                test()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
