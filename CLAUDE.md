# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rustmolbpe is a high-performance BPE (Byte Pair Encoding) tokenizer for molecular SMILES strings, written in Rust with Python bindings via PyO3. It's designed for cheminformatics and molecular machine learning applications.

## Build Commands

```bash
# Create virtual environment with uv (first time)
uv venv .venv
source .venv/bin/activate
uv pip install maturin pytest pytest-cov

# Development build (fast, unoptimized)
maturin develop

# Release build (optimized)
maturin develop --release

# Build wheel for distribution
maturin build --release
```

## Testing

```bash
# Run Rust tests
cargo test

# Run Python tests
pytest tests/python/ -v

# Run a single Python test
pytest tests/python/test_tokenizer.py::TestPadding::test_pad_right_default -v

# Run all tests (Rust + Python)
cargo test && pytest tests/python/ -v
```

## Code Quality

```bash
# Format Rust code (required before commit)
cargo fmt

# Check formatting (CI will fail if not formatted)
cargo fmt --check

# Run Rust linter (CI will fail on warnings)
cargo clippy -- -D warnings
```

## Architecture

### Modular Structure (`src/`)

The codebase is organized into focused modules:

```
src/
├── lib.rs              # PyO3 module + SmilesTokenizer struct + re-exports (~730 lines)
├── constants.rs        # SMILES pattern, special tokens, type aliases (~30 lines)
├── word.rs             # Word, MergeJob structs for training internals (~175 lines)
├── training.rs         # BPE training algorithm (~130 lines)
├── encoding.rs         # encode/decode methods (~230 lines)
├── vocabulary.rs       # load/save vocabulary, query methods (~280 lines)
├── padding.rs          # pad, encode_batch_padded (~175 lines)
├── serialization.rs    # __reduce__, __setstate__ pickle support (~170 lines)
└── utils.rs            # atomwise_tokenize helper (~145 lines)
```

### Core Components

- **SMILES Pre-tokenization** (`utils.rs`): Regex-based atom-level tokenization that handles multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures, stereochemistry
- **BPE Training** (`training.rs`, `word.rs`): Parallel pair counting with Rayon, heap-based merge selection, incremental updates
- **Greedy Encoding** (`encoding.rs`): Longest-match encoding for optimal compression (not merge-order BPE)
- **Special Tokens** (`constants.rs`): Fixed IDs 0-3 for PAD, UNK, BOS, EOS
- **Vocabulary I/O** (`vocabulary.rs`): SMILESPE-compatible format loading/saving
- **Serialization** (`serialization.rs`): Full pickle support for multiprocessing

### Python Bindings

PyO3 exposes `SmilesTokenizer` class and `atomwise_tokenize()` function. Type stubs in `rustmolbpe.pyi` for IDE support. The package includes `py.typed` marker for PEP 561 compliance. See the `.pyi` file for the full API surface.

### Pre-trained Vocabularies

- `data/chembl36_vocab.txt` - ChEMBL 36 (2.8M molecules, 7,715 merges)
- `data/pubchem_10M_vocab.txt` - PubChem (10M molecules, 6,385 merges)

## Vocabulary Format

SMILESPE-compatible format where each line is a merge rule:
```
token1 token2
```
Example: `c c` means merge `c` + `c` into `cc`.

## CI/CD

- GitHub Actions runs tests on Python 3.9-3.13
- Linting with `cargo fmt --check` and `cargo clippy`
- Code coverage with Codecov (Rust via cargo-tarpaulin, Python via pytest-cov)
- PyPI publishing workflow on release (`publish.yml`)

### Debugging CI Failures

Use `gh_cli` (not `gh`) to investigate GitHub Actions failures:

```bash
# List recent workflow runs
gh_cli run list --limit 5

# View details of a specific run
gh_cli run view <run_id>

# View failed job logs
gh_cli run view <run_id> --log-failed

# View specific job logs
gh_cli run view <run_id> --job=<job_id> --log
```

Common CI issues:
- `maturin develop` requires a virtual environment; in CI, use `uv` to create a venv:
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv pip install maturin pytest pytest-cov
  maturin develop --release
  ```
- Coverage job uses cargo-tarpaulin for Rust and pytest-cov for Python

## Utility Scripts

- `train_chembl.py` - Train vocabulary on ChEMBL 36 dataset
- `train_pubchem.py` - Train vocabulary on PubChem 10M dataset
- `preprocess_pubchem.py` - Preprocess raw PubChem data for training
- `benchmark.py` - Performance benchmarking

## Gotchas

- After any Rust code change, you must re-run `maturin develop` before Python tests will reflect the change
- The encoding algorithm is **greedy longest-match**, not standard merge-order BPE — this is intentional for better compression
- Special token IDs 0-3 (PAD, UNK, BOS, EOS) are hardcoded in `constants.rs` and must not be reassigned
- `gh_cli` (not `gh`) must be used to interact with GitHub Actions in this environment
