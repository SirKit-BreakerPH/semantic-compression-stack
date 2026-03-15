# Semantic Compression Stack

A full **5-layer LLM context compression pipeline** with round-trip decompression,
designed to run in **GitHub Codespaces** out of the box.

```
original file
    ↓ Layer 1 — Structural skeleton  (PromptPacker-style, tree-sitter)
    ↓ Layer 2 — Semantic pruning     (LLMLingua)
    ↓ Layer 3 — Importance filtering (imptokens-style, pure Python)
    ↓ Layer 4 — Embedding compress   (DragonMemory-style, sentence-transformers)
    ↓ Layer 5 — Abstractive compress (Cmprsr-style, local BART or LLM API)
compressed output
    ↓ LLM reconstruction
    ↓ Embedding decode
restored document  (~90–99% semantic fidelity)
```

---

## Quick Start (Codespaces)

1. Open this repo in GitHub Codespaces
2. Wait for `postCreateCommand` to finish installing dependencies (~3 min)
3. Run:

```bash
# Compress a file (layers 1+2+3 — no GPU needed)
python pipelines/compress.py sample_data/sample_mixed.txt \
    --layers 1,2,3 \
    --mode balanced \
    --report

# Full 5-layer compression (layers 4+5 need more RAM/time)
python pipelines/compress.py sample_data/sample_mixed.txt \
    --layers 1,2,3,4,5 \
    --mode aggressive

# Decompress / reconstruct
python pipelines/decompress.py sample_data/sample_mixed.compressed.txt \
    --embed sample_data/sample_mixed.emb.bin \
    --report
```

---

## Layer Architecture

### Layer 1 — Structural Skeletonization
**File:** `structure/skeleton_layer.py`  
**Inspired by:** PromptPacker

Auto-detects content type:
- **Code (Python/JS):** Uses tree-sitter AST → keeps imports, signatures, docstrings; strips bodies
- **Prose:** Extracts first sentence of each paragraph + high-signal sentences
- **Mixed:** Applies code skeleton to fenced blocks, prose extraction to text

Typical compression: **2–4×**  
No model required, fully deterministic.

---

### Layer 2 — Semantic Token Pruning
**File:** `compressor/llmlingua_layer.py`  
**Tool:** [LLMLingua](https://github.com/microsoft/LLMLingua) (Microsoft, real, production-grade)

Removes low-perplexity tokens while preserving meaning.
Uses a small BERT-based model to score token importance.

```bash
pip install llmlingua   # already in requirements.txt
```

Typical compression: **2–5×**  
Lossy but semantically faithful.

> **Note:** If LLMLingua is unavailable, Layer 2 is automatically skipped.

---

### Layer 3 — Token Importance / Density Filtering
**File:** `compressor/token_pruning_layer.py`  
**Inspired by:** imptokens

Pure Python implementation using TF-IDF scoring + positional weighting + boilerplate detection.
No Rust required — equivalent logic, Python-native.

Typical compression: **1.5–2.5×**  
Works well on logs, verbose prose, and repetitive text.

> **Optional:** Install the Rust `imptokens` binary for faster throughput on large files:
> ```bash
> cargo install imptokens   # Rust required
> ```

---

### Layer 4 — Semantic Embedding Compression
**File:** `embedding/embedding_layer.py`  
**Inspired by:** DragonMemory

Encodes text as sentence embeddings (all-MiniLM-L6-v2), applies PCA + int8 quantization.
Stores original chunks + compressed embedding matrix in a binary packet (`.emb.bin`).

Decompression: exact text recovery from stored chunks (lossless within packet).

Typical compression: **~8–16×** over raw text  
Uses `sentence-transformers` (local, no API key).

---

### Layer 5 — Abstractive LLM Compression
**File:** `compressor/abstractive_layer.py`  
**Inspired by:** Cmprsr

Three backend options (auto-selected by priority):

| Backend | Requirement | Quality |
|---------|------------|---------|
| `facebook/bart-large-cnn` | Local, no key needed | Good |
| OpenAI API | `OPENAI_API_KEY` env var | Better |
| Anthropic API | `ANTHROPIC_API_KEY` env var | Better |

Reconstruction also uses this layer (LLM expansion of compressed text).

---

## Compression Modes

| Mode | L1 | L2 rate | L3 keep | L5 ratio | Total |
|------|----|---------|---------|----------|-------|
| `conservative` | ✓ | 70% | 75% | 50% | ~3–5× |
| `balanced` | ✓ | 50% | 60% | 35% | ~8–15× |
| `aggressive` | ✓ | 30% | 40% | 20% | ~15–40× |

---

## Supported Content Types

| Content | Best Layers | Expected Ratio |
|---------|------------|----------------|
| Mixed docs (prose + code) | 1,2,3,4,5 | 10–30× |
| Pure prose / articles | 2,3,5 | 5–15× |
| Code / codebases | 1,3 | 3–8× |
| RAG context | 2,4 | 8–20× |
| Logs / repetitive text | 3 | 3–10× |

---

## CLI Reference

### Compress

```
python pipelines/compress.py <input> [options]

Options:
  --output, -o        Output text file path
  --layers, -l        Layers to run (default: 1,2,3,4,5)
  --mode, -m          conservative | balanced | aggressive (default: balanced)
  --embed-output      Path for .emb.bin embedding file
  --report            Save JSON report alongside output
  --quiet, -q         Suppress progress output
```

### Decompress

```
python pipelines/decompress.py <compressed.txt> [options]

Options:
  --output, -o    Restored output file path
  --embed, -e     Path to .emb.bin file (enables embedding decode)
  --report        Save JSON report
  --quiet, -q     Suppress progress
```

---

## Running Tests

```bash
# Fast tests (no LLM, no GPU needed):
python tests/test_pipeline.py

# Full pytest suite:
pip install pytest
python -m pytest tests/ -v
```

---

## Environment Variables

```bash
# For Layer 5 API backend (optional — falls back to local BART)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace cache (pre-set in devcontainer)
export HF_HOME=/workspaces/.cache/huggingface
```

---

## Realistic Expectations

| Claim | Reality |
|-------|---------|
| "20× LLMLingua compression" | Possible at aggressive settings; quality degrades |
| "16× DragonMemory compression" | Achieved in embedding space; text chunks still stored |
| "90–99% semantic recovery" | True for conservative ratios; drops at >10× |
| "100k → 5k tokens" | Achievable but aggressive; expect some information loss |

**Recommendation:** Use `--mode balanced` and `--layers 1,2,3` for most use cases.
Add layers 4 and 5 only when you need maximum compression and have a reconstruction step.

---

## Project Structure

```
semantic-compression-stack/
├── .devcontainer/
│   └── devcontainer.json        # Codespaces config (Python 3.11 + Rust)
├── compressor/
│   ├── llmlingua_layer.py       # Layer 2: LLMLingua
│   ├── token_pruning_layer.py   # Layer 3: TF-IDF importance filtering
│   └── abstractive_layer.py     # Layer 5: BART / LLM abstractive compress
├── structure/
│   └── skeleton_layer.py        # Layer 1: tree-sitter AST + prose skeleton
├── embedding/
│   └── embedding_layer.py       # Layer 4: sentence-transformers + PCA
├── pipelines/
│   ├── compress.py              # Main compression CLI
│   └── decompress.py            # Main reconstruction CLI
├── tests/
│   └── test_pipeline.py         # Unit tests (no LLM required)
├── sample_data/
│   └── sample_mixed.txt         # Test file (prose + code)
├── requirements.txt
└── README.md
```
