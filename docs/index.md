# rustmolbpe

[![PyPI](https://img.shields.io/pypi/v/rustmolbpe)](https://pypi.org/project/rustmolbpe/)
[![Python versions](https://img.shields.io/pypi/pyversions/rustmolbpe)](https://pypi.org/project/rustmolbpe/)

A high-performance BPE (Byte Pair Encoding) tokenizer for molecular SMILES written in Rust with Python bindings.

## Features

- **Five tokenizers, one API**: from a plain character-level tokenizer up to atom-level and byte-level BPE — pick the granularity you need and compare them directly
- **SMILES-aware tokenization**: Correctly handles multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures, and stereochemistry
- **Fast training**: Parallel processing with Rayon for efficient training on large molecular datasets
- **Streaming support**: Train on datasets of any size with configurable buffer sizes
- **Special tokens**: Built-in PAD, UNK, BOS, EOS tokens for sequence modeling
- **Batch padding**: Ready for transformer models with attention masks
- **SMILESPE compatibility**: Load and save vocabularies in SMILESPE format
- **HuggingFace interop**: Export to / import from the `tokenizers` `tokenizer.json` format
- **Pickle support**: Full serialization support for multiprocessing workflows
- **Type hints**: PEP 561 compliant with `py.typed` marker

## Tokenizers

| Class              | Granularity  | Learns merges | Description                                                     |
|--------------------|--------------|---------------|-----------------------------------------------------------------|
| `CharTokenizer`    | character    | no            | Splits a SMILES string into individual characters               |
| `AtomTokenizer`    | atom (regex) | no            | Splits into atoms/structural tokens (Br, Cl, [C@@H] kept whole) |
| `CharBPETokenizer` | character    | yes           | BPE merges learned on top of character splitting                |
| `SmilesTokenizer`  | atom (regex) | yes           | BPE merges learned on top of atom splitting ("SPE")             |
| `ByteBPETokenizer` | byte (UTF-8) | yes           | BPE merges on raw bytes; never emits `<unk>` once trained       |

`SmilesTokenizer` is also exported as `AtomBPETokenizer`, an exact alias of the
same class. See [Choosing a tokenizer](quickstart.md#choosing-a-tokenizer) for
how the classes differ in practice.

## Performance

rustmolbpe is significantly faster than the original Python SMILESPE implementation:

| Operation | Speedup |
|-----------|---------|
| Encoding | 25-35x faster |
| Training | 16-18x faster |

### Throughput

- **Batch encoding**: ~200,000-280,000 SMILES/second
- **Training**: 2.8M molecules in ~100 seconds

## Installation

Requires Python 3.10 or newer.

### From PyPI

```bash
pip install rustmolbpe
```

### From source

```bash
# Clone the repository
git clone https://github.com/HFooladi/rustmolbpe.git
cd rustmolbpe

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies and build
uv pip install maturin
maturin develop --release
```

## Quick Example

```python
import rustmolbpe

# Create tokenizer and load vocabulary
tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("data/chembl36_vocab.txt")

# Encode SMILES
ids = tokenizer.encode("CC(=O)Nc1ccc(O)cc1")  # paracetamol
print(ids)  # [2338, 539]

# Decode back
smiles = tokenizer.decode(ids)
print(smiles)  # CC(=O)Nc1ccc(O)cc1

# Batch processing with padding (for ML)
result = tokenizer.encode_batch_padded(
    ["CCO", "c1ccccc1", "CC(=O)O"],
    add_special_tokens=True,
    return_attention_mask=True
)
print(result["input_ids"])
print(result["attention_mask"])
```

## Pre-trained Vocabularies

Pre-trained atom-level BPE (`SmilesTokenizer`) vocabularies are included:

- `data/chembl36_vocab.txt` - Trained on ChEMBL 36 (2.8M drug-like molecules, 3,807 merges)
- `data/pubchem_10M_vocab.txt` - Trained on PubChem (10M diverse molecules, 2,410 merges)

## License

MIT License - see [LICENSE](https://github.com/HFooladi/rustmolbpe/blob/main/LICENSE)
