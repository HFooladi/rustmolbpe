# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ByteBPETokenizer` — a fifth tokenizer class extending the ladder with
  byte-level BPE. It pre-tokenizes on raw UTF-8 bytes, so the base alphabet is
  always the 256 byte values: every input is representable, no `<unk>` token is
  ever produced, and encode/decode is a guaranteed lossless round-trip.
  SMILESPE and HuggingFace file I/O are not supported (those formats store
  chemically-readable tokens, not raw bytes); use `pickle` to persist a
  byte-level tokenizer.
- `AtomBPETokenizer` — an exact alias of `SmilesTokenizer` (both names bind to
  the same class object, so `AtomBPETokenizer is SmilesTokenizer`). The alias
  name matches the `CharBPETokenizer` / `ByteBPETokenizer` pattern and makes the
  atom-level BPE granularity explicit. `SmilesTokenizer` remains the canonical
  name; nothing is deprecated.

## [0.3.0] - 2026-05-17

### Added

- Three new tokenizer classes alongside `SmilesTokenizer`, forming a ladder
  from simplest to most advanced:
  - `CharTokenizer` — character-level splitting (one token per character).
  - `AtomTokenizer` — atom-level regex splitting (multi-character atoms kept
    whole), no merges.
  - `CharBPETokenizer` — BPE trained on characters.
  - `SmilesTokenizer` — BPE trained on atoms ("SPE"), unchanged and
    backward-compatible.
- `has_vocabulary()` method on every tokenizer (reports whether a base
  vocabulary has been built, distinct from `is_trained()` which reports merges).
- HuggingFace `tokenizers` JSON interop:
  - `save_huggingface(path)` exports a tokenizer to the `tokenizer.json`
    format. `CharTokenizer`/`CharBPETokenizer` export as a `BPE` model and
    `AtomTokenizer` as a `WordLevel` model with an atom-regex `Split`
    pre-tokenizer. `SmilesTokenizer` (atom-level BPE) raises
    `NotImplementedError` — it cannot be expressed as a stock HuggingFace fast
    tokenizer.
  - `from_huggingface(path)` classmethod imports such a file back, rejecting
    files whose granularity/merge profile does not match the class.
  - New `hf` install extra (`pip install rustmolbpe[hf]`) pulls in the
    `tokenizers` library used by the export cross-check tests.
- `tokenizer_stats.py` — script that computes per-molecule token-count
  statistics for every tokenizer over the ChEMBL 36 dataset (console table,
  CSV, and histogram figure). Install extras with `pip install rustmolbpe[stats]`.
- Runtime `rustmolbpe.__version__`, populated from `CARGO_PKG_VERSION` so it
  always matches the compiled binary. The module also exports an explicit
  `__all__`.
- Code coverage reporting with Codecov integration in CI
- Coverage threshold enforcement via `codecov.yml` configuration
- 10 new Rust unit tests for core functions (22 total)
- 13 new Python edge case tests (71 total)
- Troubleshooting section in README
- Enhanced docstrings for public Rust functions
- GitHub issue templates (bug report, feature request)
- Pull request template
- `CODE_OF_CONDUCT.md` (Contributor Covenant)
- `SECURITY.md` with vulnerability reporting guidelines

### Changed

- Internal refactor: tokenizer logic extracted into a shared, granularity-agnostic
  `TokenizerCore`; the four tokenizer classes are thin wrappers generated from
  one declarative macro. Pickle state is now versioned (v2) and carries the
  pre-tokenizer granularity; legacy v1 pickles still load as atom-level.
- `CharTokenizer` and `AtomTokenizer` raise `NotImplementedError` from
  `load_vocabulary` / `save_vocabulary` (the SMILESPE format only stores merge
  rules).
- `pyproject.toml` now declares `dynamic = ["version"]`, making `Cargo.toml`
  the single source of truth for the package version (previously duplicated).

## [0.2.0] - 2025-01-21

### Added

- `py.typed` marker file for PEP 561 compliance, enabling type checking in downstream projects
- `is_trained()` method to check if a tokenizer has been trained or has a vocabulary loaded
- `get_merges()` method to retrieve learned merge rules as `(left, right, merged)` string tuples
- Pickle serialization support via `__reduce__` and `__setstate__` methods
- Multiprocessing compatibility for parallel tokenization workflows
- Comprehensive test suite for new features (15 new tests)

### Changed

- Added `module = "rustmolbpe"` attribute to `SmilesTokenizer` class for proper pickle support
- Updated `pyproject.toml` to include type stubs and `py.typed` marker in wheel distribution

## [0.1.0] - 2025-01-08

### Added

- Initial release of rustmolbpe
- SMILES-aware BPE tokenizer with atom-level pre-tokenization
- Support for multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures, and stereochemistry
- Special tokens: PAD, UNK, BOS, EOS at fixed IDs 0-3
- `SmilesTokenizer` class with:
  - `train_from_iterator()` - Train BPE from SMILES iterator with streaming support
  - `encode()` / `decode()` - Single sequence encoding/decoding
  - `batch_encode()` / `batch_decode()` - Parallel batch processing
  - `pad()` - Pad sequences to equal length with attention masks
  - `encode_batch_padded()` - Encode and pad in one step
  - `load_vocabulary()` / `save_vocabulary()` - SMILESPE-compatible format
- `atomwise_tokenize()` utility function
- Pre-trained vocabularies:
  - ChEMBL 36 (2.8M drug-like molecules, 7,715 merges)
  - PubChem (10M diverse molecules, 6,385 merges)
- Python bindings via PyO3
- Parallel processing with Rayon

### Performance

- 25-35x faster encoding than SMILESPE
- 16-18x faster training than SMILESPE
- ~200,000-280,000 SMILES/second batch encoding

[Unreleased]: https://github.com/HFooladi/rustmolbpe/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/HFooladi/rustmolbpe/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/HFooladi/rustmolbpe/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/HFooladi/rustmolbpe/releases/tag/v0.1.0
