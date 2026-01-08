# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/HFooladi/rustmolbpe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HFooladi/rustmolbpe/releases/tag/v0.1.0
