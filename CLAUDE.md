# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rustmolbpe is a high-performance BPE (Byte Pair Encoding) tokenizer for molecular SMILES strings, written in Rust with Python bindings via PyO3. It's designed for cheminformatics and molecular machine learning applications.

## Build Commands

```bash
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

### Core Implementation (`src/lib.rs`)

Single-file Rust implementation (~1200 lines) containing:

- **SMILES Pre-tokenization**: Regex-based atom-level tokenization that handles multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures, stereochemistry
- **BPE Training**: Parallel pair counting with Rayon, heap-based merge selection, incremental updates
- **Greedy Encoding**: Longest-match encoding for optimal compression (not merge-order BPE)
- **Special Tokens**: Fixed IDs 0-3 for PAD, UNK, BOS, EOS

Key data structures:
```rust
pub struct SmilesTokenizer {
    merges: HashMap<(u32, u32), u32>,      // Pair -> merged ID
    atom_to_id: AHashMap<CompactString, u32>,  // Atom string -> ID
    id_to_atom: Vec<CompactString>,         // ID -> atom string
    compiled_pattern: Regex,                 // SMILES tokenization regex
}
```

### Python Bindings

PyO3 exposes `SmilesTokenizer` class and `atomwise_tokenize()` function. Type stubs in `rustmolbpe.pyi` for IDE support.

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

- GitHub Actions runs tests on Python 3.9-3.12
- Linting with `cargo fmt --check` and `cargo clippy`
- PyPI publishing workflow on release (`publish.yml`)
