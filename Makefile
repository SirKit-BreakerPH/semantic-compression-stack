.PHONY: help install test benchmark compress-sample decompress-sample clean lint

PYTHON := python
SAMPLE := sample_data/sample_mixed.txt

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────
install:  ## Install all Python dependencies
	pip install -r requirements.txt --break-system-packages

install-rust:  ## Install Rust (for tree-sitter compiled grammars)
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	source ~/.cargo/env

# ── Testing ────────────────────────────────────────────────────────────────
test:  ## Run unit tests (no LLM / GPU required)
	$(PYTHON) tests/test_pipeline.py

test-pytest:  ## Run full pytest suite
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Benchmarking ───────────────────────────────────────────────────────────
benchmark:  ## Benchmark L1+L3 on sample data
	$(PYTHON) benchmark.py

benchmark-matrix:  ## Benchmark all mode × layer combinations
	$(PYTHON) benchmark.py --matrix

benchmark-file:  ## Benchmark a specific file (make benchmark-file FILE=my.txt)
	$(PYTHON) benchmark.py --input $(FILE)

# ── Compression ────────────────────────────────────────────────────────────
compress-fast:  ## Compress sample with L1+L3 (no GPU/API needed)
	$(PYTHON) pipelines/compress.py $(SAMPLE) \
		--layers 1,2,3 \
		--mode balanced \
		--report

compress-full:  ## Full 5-layer compression on sample (needs models)
	$(PYTHON) pipelines/compress.py $(SAMPLE) \
		--layers 1,2,3,4,5 \
		--mode balanced \
		--report

compress-aggressive:  ## Aggressive 5-layer compression
	$(PYTHON) pipelines/compress.py $(SAMPLE) \
		--layers 1,2,3,4,5 \
		--mode aggressive \
		--report

compress-file:  ## Compress any file (make compress-file FILE=my.txt MODE=balanced)
	$(PYTHON) pipelines/compress.py $(FILE) \
		--layers 1,2,3,4,5 \
		--mode $(or $(MODE),balanced) \
		--report

# ── Decompression ──────────────────────────────────────────────────────────
decompress-sample:  ## Reconstruct last compressed sample
	$(PYTHON) pipelines/decompress.py \
		sample_data/sample_mixed.compressed.txt \
		--embed sample_data/sample_mixed.emb.bin \
		--report

decompress-file:  ## Decompress a file (make decompress-file FILE=compressed.txt EMBED=file.emb.bin)
	$(PYTHON) pipelines/decompress.py $(FILE) \
		$(if $(EMBED),--embed $(EMBED),) \
		--report

# ── Utilities ──────────────────────────────────────────────────────────────
clean:  ## Remove all generated files
	find . -name "*.compressed.txt" -delete
	find . -name "*.restored.txt" -delete
	find . -name "*.emb.bin" -delete
	find . -name "*.report.json" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

lint:  ## Lint Python files
	$(PYTHON) -m py_compile structure/skeleton_layer.py
	$(PYTHON) -m py_compile compressor/token_pruning_layer.py
	$(PYTHON) -m py_compile compressor/llmlingua_layer.py
	$(PYTHON) -m py_compile compressor/abstractive_layer.py
	$(PYTHON) -m py_compile embedding/embedding_layer.py
	$(PYTHON) -m py_compile pipelines/compress.py
	$(PYTHON) -m py_compile pipelines/decompress.py
	$(PYTHON) -m py_compile benchmark.py
	@echo "✅ All files pass syntax check"

# ── Quick start ────────────────────────────────────────────────────────────
quickstart:  ## Full demo: install → test → benchmark → compress → decompress
	@echo "\n🚀 Running quickstart demo..."
	$(MAKE) test
	$(MAKE) benchmark
	$(MAKE) compress-fast
	@echo "\n✅ Quickstart complete. Check sample_data/ for outputs."
