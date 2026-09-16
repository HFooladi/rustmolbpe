# Examples

Runnable scripts that introduce rustmolbpe. Each one is self-contained and
prints what it demonstrates.

```bash
pip install rustmolbpe
python examples/compare_tokenizers.py
```

| Script | What it shows | Needs |
|---|---|---|
| [`basic_usage.py`](basic_usage.py) | Loading the pre-trained ChEMBL vocabulary, encoding and decoding, special tokens, atom-level splitting, vocabulary lookup | pre-trained vocabulary |
| [`compare_tokenizers.py`](compare_tokenizers.py) | The five tokenizers (`CharTokenizer`, `AtomTokenizer`, `CharBPETokenizer`, `SmilesTokenizer`, `ByteBPETokenizer`) side by side: vocabulary sizes, tokens per molecule, handling of unseen elements | — |
| [`train_tokenizer.py`](train_tokenizer.py) | Training a BPE tokenizer from an iterator, compression, saving and reloading with `save()` / `from_file()`, SMILESPE export | — |
| [`batch_processing.py`](batch_processing.py) | Batch encoding, padding and truncation, attention masks, left padding, the `tokenizer(...)` call interface, batch vs one-by-one speed | pre-trained vocabulary |
| [`persistence_and_interop.py`](persistence_and_interop.py) | `save()` / `from_file()`, pickle, multiprocessing, HuggingFace `tokenizer.json` export | optional: `tokenizers` (`pip install "rustmolbpe[hf]"`) |

A good order for a first read: `basic_usage.py`, `compare_tokenizers.py`,
`train_tokenizer.py`, then the other two.

## Pre-trained vocabulary

`basic_usage.py` and `batch_processing.py` load `data/chembl36_vocab.txt` from
this repository. They find it wherever you run them from, as long as the script
stays inside a clone of the repository. Otherwise, download
[`chembl36_vocab.txt`](https://github.com/HFooladi/rustmolbpe/blob/main/data/chembl36_vocab.txt)
and pass its path:

```bash
python basic_usage.py path/to/chembl36_vocab.txt
```

The other scripts train on small molecule lists included in the script and run
anywhere.

## Testing

`tests/python/test_examples.py` runs every script from an unrelated directory
and checks the result it demonstrates, so the examples run in CI with the rest
of the test suite. A new script must be added to that test's
`_EXPECTED_OUTPUT` table.
