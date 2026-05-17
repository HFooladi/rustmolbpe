# rustmolbpe Roadmap

This document outlines the planned improvements and future direction for rustmolbpe. Items are organized by release timeline and include effort estimates.

**Effort Legend:**
- 🟢 Easy - A few hours to a day
- 🟡 Medium - A few days to a week
- 🔴 Hard - Multiple weeks or significant complexity

---

## Completed (v0.1.0 – v0.3.0)

Work already shipped. Kept here for context; see [CHANGELOG.md](CHANGELOG.md) for the per-release detail.

### v0.3.0 — Tokenizer ladder & interop
- [x] 🔴 Four-tokenizer ladder sharing one `TokenizerCore` — `CharTokenizer`, `AtomTokenizer`, `CharBPETokenizer`, `SmilesTokenizer`
- [x] 🔴 HuggingFace `tokenizer.json` interop — `save_huggingface()` / `from_huggingface()` (atom-level BPE export excepted)
- [x] 🟡 Granularity-aware pickle format (v2); legacy v1 pickles still load
- [x] 🟡 `tokenizer_stats.py` — per-molecule token-count statistics across all four tokenizers
- [x] 🟢 `has_vocabulary()` method (distinct from `is_trained()`)
- [x] 🟢 Runtime `rustmolbpe.__version__`, with `Cargo.toml` as the single source of truth (`dynamic` version in `pyproject.toml`)

### v0.2.0 — API, testing & infrastructure
- [x] 🟢 `py.typed` marker for PEP 561 compliance
- [x] 🟢 `is_trained()` method to check tokenizer state
- [x] 🟡 `get_merges()` method to inspect learned merge rules
- [x] 🟡 `__reduce__`/`__setstate__` for pickle support
- [x] 🟢 Code coverage reporting (Codecov) with threshold enforcement in CI
- [x] 🟢 Rust unit tests for core functions
- [x] 🟡 Error-handling edge-case tests
- [x] 🟢 Troubleshooting section in README; `CHANGELOG.md`; docstrings on public Rust functions
- [x] 🟢 Issue templates, PR template, `CODE_OF_CONDUCT.md`, `SECURITY.md`

---

## v0.4.0 — ML Framework Integration

**Current priority.** Make the tokenizer drop-in for ML training pipelines so users don't hand-roll glue code.

- [ ] 🟢 NumPy array output options for `encode()` and `batch_encode()`
- [ ] 🟡 PyTorch tensor output support (optional dependency)
- [ ] 🟡 DataLoader-compatible dataset wrapper class
- [ ] 🟡 Collate function for variable-length sequences
- [ ] 🟡 GPU-friendly batch preparation utilities
- [ ] 🟡 Integration guide for the `transformers` library (using `__call__` + `save_huggingface`)

---

## v0.5.0 — Custom Special Tokens & Performance

Flexibility for non-default vocabularies, plus the lower-risk performance wins.

### Custom Special Tokens
- [ ] 🟡 Support user-defined special tokens beyond PAD/UNK/BOS/EOS
- [ ] 🟡 Configurable special token IDs
- [ ] 🟢 Add `add_special_tokens()` method
- [ ] 🟢 `skip_special_tokens` option on `decode()` / `batch_decode()`

### Performance
- [ ] 🟡 Configurable thread count for parallel operations
- [ ] 🟡 Batch encoding optimizations with better memory reuse
- [ ] 🟡 Memory-mapped vocabulary loading for large vocabularies

---

## v0.6.0 — Vocabulary & Serialization

Make vocabularies inspectable, verifiable, and composable.

- [ ] 🟢 Add vocabulary format version field
- [ ] 🟡 Vocabulary validation during loading (detect malformed files)
- [ ] 🟡 `get_statistics()` method (vocab size, merge count, token frequencies)
- [ ] 🟡 Vocabulary merging utility (combine two vocabularies)
- [ ] 🟡 Save/load configuration separately from vocabulary
- [ ] 🟢 Version compatibility checking on load
- [ ] 🔴 `SmilesTokenizer` → HuggingFace export via a custom pre-tokenizer component (currently raises `NotImplementedError` — atom-level BPE cannot be expressed as a stock HF fast tokenizer)

---

## v1.0.0 — Stabilization & Advanced Performance

Major features and stabilization for production readiness.

### Performance
- [ ] 🔴 Trie-based encoding for O(n log V) complexity (currently O(n·m))
- [ ] 🔴 Optional SIMD acceleration for batch operations
- [ ] 🔴 Streaming encode/decode for memory-constrained environments
- [ ] 🟡 Lazy vocabulary loading

### Advanced Vocabulary Features
- [ ] 🔴 Vocabulary pruning (remove low-frequency merges)
- [ ] 🔴 Custom atom pattern support (user-defined tokenization regex)
- [ ] 🟡 Vocabulary analysis and visualization tools
- [ ] 🟡 Merge rule importance scoring

### API Stabilization
- [ ] 🟡 Semantic versioning guarantees
- [ ] 🟡 Deprecation policy and migration guides
- [ ] 🟢 API stability markers

### Extended Framework Support
- [ ] 🔴 JAX/Flax tensor support

---

## Testing & Quality (ongoing)

- [ ] 🟡 Performance regression benchmarks in CI
- [ ] 🟡 Threading safety tests with concurrent access
- [ ] 🟡 Property-based testing with hypothesis
- [ ] 🟢 Fuzz testing for SMILES parsing edge cases

---

## Documentation & Community (ongoing)

- [ ] 🟡 API reference site (using pdoc or mkdocs)
- [ ] 🟡 Architecture documentation (BPE internals, SMILES parsing deep-dive)
- [ ] 🟡 Performance tuning guide
- [ ] 🟡 ML framework integration tutorials (Jupyter notebooks)
- [ ] 🟢 `examples/` directory with common use cases
- [ ] 🟢 Migration guides from other SMILES tokenizers
- [ ] 🟢 FAQ section
- [ ] 🟢 Contributing guide improvements
- [ ] 🟢 GitHub discussion categories

---

## Infrastructure (ongoing)

- [ ] 🟡 Prebuilt cross-platform binary wheels in CI (manylinux / macOS / Windows; `abi3` to cut the build matrix)
- [ ] 🟡 conda-forge package
- [ ] 🟡 Git LFS for large data files (vocabularies, training data)
- [ ] 🟡 Docker development environment
- [ ] 🟡 Automated changelog generation from commits
- [ ] 🟢 Pre-commit hooks configuration

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

*Last updated: May 2026*
