# Vocabulary I/O: preserve merge order and add a lossless tokenizer file

- **Date:** 2026-09-16
- **Status:** approved design, pre-implementation

## Problem

Measured on `main` at `87d8479`:

1. **Loading a SMILESPE vocabulary loses merge priority order.**
   `vocabulary::load_vocabulary` assigns IDs to every token that appears in a
   merge rule, sorted alphabetically, including intermediate merged tokens.
   `get_merges` and `save_vocabulary` then recover "order" by sorting on merged
   token ID, which is not file order for a loaded vocabulary. Consequences:
   - Load → save of `data/chembl36_vocab.txt` leaves only 135 of 3,807 lines in
     place. The original SmilesPE library tokenizes 2,995 of 3,000 ChEMBL
     molecules differently with the re-saved file.
   - `save_huggingface` on a loaded `CharBPETokenizer` exports merges in the wrong
     priority. HuggingFace `tokenizers` output differs for 1,989 of 2,000
     molecules versus exporting the trained tokenizer.
   - `get_merges()` on a loaded vocabulary is not in learning order.
2. **Token IDs after `load_vocabulary` differ from the trained tokenizer, and
   never-merged base tokens become `<unk>`.** Training orders base tokens by
   corpus frequency, and the SMILESPE format stores only merge rules. This
   cannot be fixed within that format.

## Goals

- Merge priority order survives load, save, `get_merges`, HuggingFace export and pickle.
- **Token IDs assigned by `load_vocabulary` stay exactly as today.** Models trained
  on IDs from loaded vocabularies (e.g. `chembl36_vocab.txt`) keep working.
- A lossless, human-readable way to persist any tokenizer: identical IDs, merges
  and encodings after a round trip, for all five classes.

## Non-goals

- Reproducing trained IDs from a SMILESPE file (impossible, see Problem 2).
- Changing the SMILESPE file format (SmilesPE requires every line to be exactly
  two space-separated units; extra lines would be parsed as merge rules).
- `ByteBPETokenizer` HuggingFace export (separate roadmap item).

## Design

### Part 1: merges as an ordered list

- `TokenizerCore::merges` changes from `StdHashMap<Pair, u32>` to
  `Vec<(Pair, u32)>`: `((left_id, right_id), merged_id)` in priority order.
  No production code looks merges up by pair (encoding is greedy longest-match
  over `atom_to_id`), so a list is sufficient and order is kept by construction.
- Producers:
  - `training::train_core_incremental` pushes merges in learning order.
  - `vocabulary::load_vocabulary` pushes in file order. ID assignment is
    **unchanged**. A repeated `(left, right)` line is skipped, keeping the first
    occurrence (SmilesPE semantics; the old map also held one entry per pair).
  - `huggingface::restore_from_hf_json` pushes in the file's merge order.
  - Pickle restore: see below.
- Consumers `get_merges`, `save_vocabulary` and `to_hf_json` (via `get_merges`)
  iterate the list without sorting. `num_merges`, `base_vocab_size` and
  `is_trained` use the list length, as before.
- **Pickle** stays at version 2:
  - `__reduce__` writes `merges` in priority order and adds
    `"merges_ordered": True`.
  - `__setstate__` keeps list order when `merges_ordered` is true. Otherwise
    (older pickles, whose list came from hash-map iteration) it sorts by merged
    ID, which reproduces today's order.
  - rustmolbpe 0.4.0 ignores unknown keys, so it can still read new pickles.

### Part 2: lossless tokenizer file (`save` / `from_file`)

- API, generated once in `define_tokenizer!` for all five classes:
  - `tok.save(path: str) -> None`
  - `Cls.from_file(path: str) -> Cls` (classmethod)
- File layout (JSON, UTF-8, written by new module `src/native_format.rs` using
  the existing `serde` / `serde_json` dependencies):

  ```json
  {
    "format": "rustmolbpe",
    "version": 1,
    "pretokenizer": "atom",
    "vocab": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "c", "..."],
    "merges": [[4, 4, 60], [5, 5, 61]]
  }
  ```

  - `pretokenizer`: `"atom" | "char" | "byte"` (the existing `PreTokenizerKind` tag).
  - `vocab`: token strings; list index = token ID. Byte-level tokens use the
    existing byte↔char bijection, so they are valid JSON strings.
  - `merges`: `[left_id, right_id, merged_id]` in priority order.
- Loading validates and raises:
  - `IOError`: the file cannot be read or written.
  - `ValueError`:
    - invalid JSON;
    - `format` is not `"rustmolbpe"`, or `version` is unsupported;
    - `pretokenizer` does not match the class (same rule as pickle);
    - `merges` is non-empty for `CharTokenizer` / `AtomTokenizer`;
    - `vocab[0..4]` is not `<pad>`, `<unk>`, `<bos>`, `<eos>`;
    - any merge ID is out of range, or `vocab[merged] != vocab[left] + vocab[right]`.
- A file with no merges loads into any class of matching granularity, including
  a BPE class. The result is an untrained tokenizer with that base vocabulary,
  the same state as saving an untrained BPE tokenizer.
- `atom_to_id` is rebuilt from `vocab` the same way pickle restore does it.

### Docs and messages

- `load_vocabulary` / `save_vocabulary` docs (`.pyi`, `docs/api.md`, README):
  SMILESPE stores merge rules only. Loading reassigns IDs, and base tokens that
  never merged are not included. Use `save` / `from_file` to persist a tokenizer
  for a model.
- The persistence table in `docs/quickstart.md` gains a `save`/`from_file` column
  (supported by all classes).
- `ByteBPETokenizer` error messages that currently say "use pickle" recommend
  `save()` / `from_file()` instead.
- CHANGELOG `[Unreleased]`: **Fixed** merge order after `load_vocabulary`
  (re-save, HuggingFace export, `get_merges`, pickle). **Added** `save` /
  `from_file`.
- CLAUDE.md architecture list gains `native_format.rs`.

## Testing (written first)

- Python:
  - `load_vocabulary("data/chembl36_vocab.txt")` then `save_vocabulary` gives a
    byte-identical file.
  - For CharBPE and SPE: trained → `save_vocabulary` → `load_vocabulary` keeps the
    trained `get_merges()` order. `save_huggingface` of a loaded CharBPE gives the
    same `merges` list as for the trained tokenizer.
  - A pickle round trip of a loaded tokenizer preserves merge order.
  - An old-style pickle state (no `merges_ordered`) still restores, ordered by
    merged ID.
  - Loaded IDs unchanged (hand-recorded from current `main`): `chembl36_vocab.txt`
    encodes paracetamol to `[2338, 539]`, `CCO` to `[353]`, `c1ccccc1` to `[782]`.
  - `save` → `from_file` for all five classes gives identical `get_vocabulary()`,
    `get_merges()`, `vocab_size`, `base_vocab_size`, `num_merges` and encodings.
    A trained ByteBPE still round-trips non-ASCII input without `<unk>`.
  - Each `ValueError` case above, plus a missing file giving `IOError`.
- Rust: `load_vocabulary` keeps file order and skips duplicate pairs; native-format
  round trip and validation at the core level.
- Regression: `min_frequency=1` training output (merges, vocab, encodings) is
  unchanged from `main` on a ChEMBL sample.
