# rustmolbpe Roadmap

This document outlines the planned improvements and future direction for rustmolbpe. Items are organized by release timeline and include effort estimates.

**Effort Legend:**
- 🟢 Easy - A few hours to a day
- 🟡 Medium - A few days to a week
- 🔴 Hard - Multiple weeks or significant complexity

---

## Short-term (v0.2.0)

Quick wins and important fixes that improve usability without major architectural changes.

### Python API Improvements
- [x] 🟢 Add `py.typed` marker for PEP 561 compliance
- [x] 🟢 Add `is_trained()` method to check tokenizer state
- [x] 🟢 ~~Add `get_vocab_size()` method~~ (skipped - `vocab_size` property already exists)
- [x] 🟡 Add `get_merges()` method to inspect learned merge rules
- [x] 🟡 Implement `__reduce__`/`__setstate__` for pickle support

### Testing & Quality
- [x] 🟢 Add code coverage reporting with codecov
- [x] 🟢 Add Rust unit tests for core functions in lib.rs
- [x] 🟡 Add error handling edge case tests
- [x] 🟡 Set up coverage threshold enforcement in CI

### Documentation
- [x] 🟢 Add troubleshooting section to README
- [x] 🟢 Create CHANGELOG.md with proper versioning history
- [x] 🟢 Add docstrings to all public Rust functions

### Infrastructure
- [x] 🟢 Add issue templates (bug report, feature request)
- [x] 🟢 Add pull request template
- [x] 🟢 Add CODE_OF_CONDUCT.md
- [x] 🟢 Add SECURITY.md with vulnerability reporting guidelines

---

## Medium-term (v0.3.0 - v0.5.0)

Feature enhancements that improve performance, flexibility, and integration capabilities.

### Performance Improvements (v0.3.0)
- [ ] 🔴 Implement trie-based encoding for O(n log V) complexity (currently O(n*m))
- [ ] 🟡 Add configurable thread count for parallel operations
- [ ] 🟡 Memory-mapped vocabulary loading for large vocabularies
- [ ] 🟡 Batch encoding optimizations with better memory reuse

### Custom Special Tokens (v0.3.0)
- [ ] 🟡 Support user-defined special tokens beyond PAD/UNK/BOS/EOS
- [ ] 🟡 Configurable special token IDs
- [ ] 🟢 Add `add_special_tokens()` method

### ML Framework Integration (v0.4.0)

*Priority: PyTorch and NumPy first*

- [ ] 🟢 NumPy array output options for `encode()` and `encode_batch()`
- [ ] 🟡 PyTorch tensor output support (optional dependency)
- [ ] 🟡 DataLoader-compatible dataset wrapper class
- [ ] 🟡 Collate function for variable-length sequences
- [ ] 🟡 GPU-friendly batch preparation utilities

### Vocabulary Features (v0.4.0)
- [ ] 🟢 Add vocabulary format version field
- [ ] 🟡 Vocabulary validation during loading (detect malformed files)
- [ ] 🟡 `get_statistics()` method (vocab size, merge count, token frequencies)
- [ ] 🟡 Vocabulary merging utility (combine two vocabularies)

### Tokenizer Serialization (v0.5.0)
- [x] 🟡 JSON export/import for tokenizer state — `save_huggingface` /
  `from_huggingface` (HuggingFace `tokenizers` JSON format)
- [ ] 🟡 Save/load configuration separately from vocabulary
- [ ] 🟢 Version compatibility checking on load

### Testing Enhancements
- [ ] 🟡 Performance regression benchmarks in CI
- [ ] 🟡 Threading safety tests with concurrent access
- [ ] 🟡 Property-based testing with hypothesis
- [ ] 🟢 Fuzz testing for SMILES parsing edge cases

---

## Long-term (v1.0.0+)

Major features and stabilization for production readiness.

### Performance (v1.0.0)
- [ ] 🔴 Optional SIMD acceleration for batch operations
- [ ] 🔴 Streaming encode/decode for memory-constrained environments
- [ ] 🟡 Lazy vocabulary loading

### Advanced Vocabulary Features
- [ ] 🔴 Vocabulary pruning (remove low-frequency merges)
- [ ] 🔴 Custom atom pattern support (user-defined tokenization regex)
- [ ] 🟡 Vocabulary analysis and visualization tools
- [ ] 🟡 Merge rule importance scoring

### API Stabilization (v1.0.0)
- [ ] 🟡 Semantic versioning guarantees
- [ ] 🟡 Deprecation policy and migration guides
- [ ] 🟢 API stability markers

### Extended Framework Support
- [x] 🟡 HuggingFace Tokenizers compatibility layer — `save_huggingface` /
  `from_huggingface` export to / import from `tokenizer.json` (atom-level BPE
  excepted; see README)
- [ ] 🟡 Integration guide for transformers library
- [ ] 🔴 JAX/Flax tensor support

### Documentation
- [ ] 🟡 Algorithm deep-dive documentation (BPE internals, SMILES parsing)
- [ ] 🟡 Performance tuning guide
- [ ] 🟡 ML framework integration tutorials
- [ ] 🟢 Migration guides from other SMILES tokenizers

### Infrastructure
- [ ] 🟡 Git LFS for large data files (vocabularies)
- [ ] 🟢 Pre-commit hooks configuration
- [ ] 🟡 Docker development environment
- [ ] 🟡 Automated changelog generation from commits

---

## Community & Documentation

Ongoing efforts to build community and improve accessibility.

### Community Building
- [ ] Add discussion categories on GitHub
- [ ] Create examples/ directory with common use cases
- [ ] Jupyter notebook tutorials
- [ ] Benchmarks comparing with other SMILES tokenizers

### Documentation Improvements
- [ ] API reference site (using pdoc or mkdocs)
- [ ] Architecture documentation
- [ ] Contributing guide improvements
- [ ] FAQ section

---

## Won't Do / Out of Scope

Items explicitly not planned for this project:

- **General-purpose BPE**: This tokenizer is specifically designed for SMILES strings. For general text, use [tiktoken](https://github.com/openai/tiktoken) or similar.
- **Training from SMILES files directly**: Users should preprocess their data. The tokenizer trains from in-memory string lists.
- **GUI/Web interface**: This is a library, not an application.
- **Non-canonical SMILES normalization**: Canonicalization should be done before tokenization using RDKit or similar.
- **Reaction SMILES support**: Focus is on molecule SMILES. Reaction handling (with `>>`) may produce unexpected results.
- **SELFIES support**: Different representation format; consider [selfies](https://github.com/aspuru-guzik-group/selfies) library instead.

---

## Contributing

We welcome contributions! If you're interested in working on any roadmap item:

1. Check if there's an existing issue for the feature
2. Open an issue to discuss your approach before starting work
3. Reference the roadmap item in your PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **PATCH** (0.1.x): Bug fixes, documentation updates
- **MINOR** (0.x.0): New features, backward-compatible changes
- **MAJOR** (x.0.0): Breaking API changes

---

*Last updated: January 2025*
