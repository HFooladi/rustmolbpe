# Quick Start

This guide will help you get started with rustmolbpe.

## Installation

rustmolbpe requires Python 3.10 or newer.

### From PyPI (Recommended)

```bash
pip install rustmolbpe
```

### From Source

```bash
git clone https://github.com/HFooladi/rustmolbpe.git
cd rustmolbpe

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and build
uv pip install maturin
maturin develop --release
```

## Choosing a tokenizer

rustmolbpe provides five tokenizer classes with an identical API, so you can
swap one for another and compare them directly. They differ in how a SMILES
string is split into base units, and in whether BPE merges are learned on top:

| Class              | Base units   | Learns merges | Typical use                                              |
|--------------------|--------------|---------------|----------------------------------------------------------|
| `CharTokenizer`    | characters   | no            | Simplest baseline; one token per character               |
| `AtomTokenizer`    | atoms        | no            | Chemically meaningful tokens without training            |
| `CharBPETokenizer` | characters   | yes           | Compact sequences; exportable to HuggingFace             |
| `SmilesTokenizer`  | atoms        | yes           | Compact, chemically aware sequences ("SPE")              |
| `ByteBPETokenizer` | UTF-8 bytes  | yes           | Any input is representable; never emits `<unk>` once trained |

`SmilesTokenizer` is also available as `AtomBPETokenizer` (the same class).

```python
import rustmolbpe

char = rustmolbpe.CharTokenizer()
atom = rustmolbpe.AtomTokenizer()

# Character-level: chlorine "Cl" is two tokens ('C', 'l')
len(char.encode("CCl"))  # 3

# Atom-level: chlorine "Cl" is a single token
len(atom.encode("CCl"))  # 2
```

`CharTokenizer` and `AtomTokenizer` have no merges: `train_from_iterator` only
builds the base vocabulary (`vocab_size` is ignored). Use `has_vocabulary()` to
check whether a base vocabulary has been built and `is_trained()` to check for
learned merges.

`ByteBPETokenizer` always uses the 256 byte values as its base alphabet
(`base_vocab_size == 260` once trained, including the 4 special tokens). Because
`vocab_size` counts that base alphabet, it learns fewer merges than
`CharBPETokenizer` at the same `vocab_size`.

How each class can be saved:

| Class              | SMILESPE vocabulary file | HuggingFace `tokenizer.json` | pickle |
|--------------------|--------------------------|------------------------------|--------|
| `CharTokenizer`    | —                        | yes                          | yes    |
| `AtomTokenizer`    | —                        | yes                          | yes    |
| `CharBPETokenizer` | yes                      | yes                          | yes    |
| `SmilesTokenizer`  | yes                      | —                            | yes    |
| `ByteBPETokenizer` | —                        | —                            | yes    |

Unsupported formats raise `NotImplementedError`.

## Basic Usage

### Loading a Pre-trained Tokenizer

```python
import rustmolbpe

# Create tokenizer
tokenizer = rustmolbpe.SmilesTokenizer()

# Load pre-trained vocabulary
tokenizer.load_vocabulary("data/chembl36_vocab.txt")

print(f"Vocabulary size: {tokenizer.vocab_size}")
```

### Encoding and Decoding

```python
# Encode a SMILES string
smiles = "CC(=O)Nc1ccc(O)cc1"  # paracetamol
ids = tokenizer.encode(smiles)
print(f"Token IDs: {ids}")

# Decode back to SMILES
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
assert decoded == smiles
```

### Using Special Tokens

```python
# Encode with BOS/EOS tokens
ids = tokenizer.encode("CCO", add_special_tokens=True)
# Result: [2, ..., 3]  where 2=BOS, 3=EOS

# Special token IDs are fixed:
print(f"PAD: {tokenizer.pad_token_id}")  # 0
print(f"UNK: {tokenizer.unk_token_id}")  # 1
print(f"BOS: {tokenizer.bos_token_id}")  # 2
print(f"EOS: {tokenizer.eos_token_id}")  # 3
```

## Batch Processing

### Basic Batch Encoding

```python
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]

# Parallel batch encoding
ids_list = tokenizer.batch_encode(smiles_list)

# Batch decoding
decoded_list = tokenizer.batch_decode(ids_list)
```

### Padded Batches for ML

For training neural networks, you need padded sequences:

```python
result = tokenizer.encode_batch_padded(
    smiles_list,
    max_length=20,           # Pad/truncate to this length
    padding="right",          # Pad on the right
    truncation=True,          # Truncate if longer
    add_special_tokens=True,  # Add BOS/EOS
    return_attention_mask=True
)

input_ids = result["input_ids"]       # Padded token IDs
attention_mask = result["attention_mask"]  # 1 for real tokens, 0 for padding
```

## Training Your Own Tokenizer

```python
def smiles_generator(filepath):
    """Generator for streaming large files."""
    with open(filepath) as f:
        for line in f:
            yield line.strip()

# Create and train
tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.train_from_iterator(
    smiles_generator("molecules.smi"),
    vocab_size=8000,
    buffer_size=16384,
    min_frequency=2
)

# Save vocabulary
tokenizer.save_vocabulary("my_vocab.txt")
```

## Atom-level Tokenization

For inspection or custom processing:

```python
atoms = rustmolbpe.atomwise_tokenize("c1ccccc1")
print(atoms)  # ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']

atoms = rustmolbpe.atomwise_tokenize("[C@@H](O)C")
print(atoms)  # ['[C@@H]', '(', 'O', ')', 'C']
```

## Vocabulary Inspection

```python
# Get all tokens
vocab = tokenizer.get_vocabulary()
for token, token_id in vocab[:10]:
    print(f"ID {token_id}: '{token}'")

# Lookup
token_id = tokenizer.token_to_id("CC")
token = tokenizer.id_to_token(token_id)

# Check if tokenizer is trained
print(tokenizer.is_trained())  # True

# Inspect merge rules
merges = tokenizer.get_merges()
print(merges[:3])  # First 3 merge rules
```

## Serialization

### Pickle Support

Tokenizers can be pickled for saving or multiprocessing:

```python
import pickle

# Save to bytes
data = pickle.dumps(tokenizer)

# Restore
restored = pickle.loads(data)
assert tokenizer.encode("CCO") == restored.encode("CCO")
```

### Multiprocessing

```python
from multiprocessing import Pool

def encode_smiles(smiles):
    return tokenizer.encode(smiles)

with Pool(4) as pool:
    results = pool.map(encode_smiles, smiles_list)
```

## HuggingFace Interop

`CharTokenizer`, `AtomTokenizer` and `CharBPETokenizer` can be exported to the
HuggingFace `tokenizers` `tokenizer.json` format and loaded back:

```python
tok = rustmolbpe.CharBPETokenizer()
tok.train_from_iterator(smiles_generator("molecules.smi"), vocab_size=8000)
tok.save_huggingface("tokenizer.json")

# Load it back into rustmolbpe...
restored = rustmolbpe.CharBPETokenizer.from_huggingface("tokenizer.json")

# ...or use it from `transformers`:
from transformers import PreTrainedTokenizerFast
hf = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

`from_huggingface` raises `ValueError` if the file does not match the class it
is called on. See the [API Reference](api.md#huggingface-interop) for details.

## Next Steps

- See [API Reference](api.md) for complete documentation
- Check `examples/` directory for more detailed examples
