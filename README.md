# rustmolbpe

[![CI](https://github.com/HFooladi/rustmolbpe/actions/workflows/ci.yml/badge.svg)](https://github.com/HFooladi/rustmolbpe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/HFooladi/rustmolbpe/graph/badge.svg)](https://codecov.io/gh/HFooladi/rustmolbpe)

A high-performance BPE (Byte Pair Encoding) tokenizer for molecular SMILES written in Rust with Python bindings.

## Features

- **Four tokenizers, one API**: from a plain character-level tokenizer up to atom-level BPE — pick the granularity you need and compare them directly
- **SMILES-aware tokenization**: atom-level pre-tokenization correctly handles multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures, and stereochemistry
- **Fast training**: Parallel processing with Rayon for efficient training on large molecular datasets
- **Streaming support**: Train on datasets of any size with configurable buffer sizes
- **SMILESPE compatibility**: Load and save vocabularies in SMILESPE format
- **HuggingFace interop**: Export to / import from the `tokenizers` `tokenizer.json` format for use with `transformers`
- **Python bindings**: Seamless integration with Python via PyO3

## Installation

### From PyPI (Recommended)

```bash
pip install rustmolbpe
```

### From source (requires Rust toolchain)

```bash
# Clone the repository
git clone https://github.com/HFooladi/rustmolbpe.git
cd rustmolbpe

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install build dependencies and build
uv pip install maturin
maturin develop --release
```

### Development install

```bash
# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dev dependencies
uv pip install maturin pytest pytest-cov

# Development build (faster, unoptimized)
maturin develop
```

## Quick Start

```python
import rustmolbpe

# Create a tokenizer
tokenizer = rustmolbpe.SmilesTokenizer()

# Train on SMILES data
smiles_data = [
    "CCO",           # ethanol
    "c1ccccc1",      # benzene
    "CC(=O)O",       # acetic acid
    # ... more SMILES
]
tokenizer.train_from_iterator(iter(smiles_data), vocab_size=1000)

# Encode SMILES
ids = tokenizer.encode("CCO")
print(ids)  # e.g., [42, 15]

# Decode back to SMILES
smiles = tokenizer.decode(ids)
print(smiles)  # "CCO"

# Batch encode (parallelized)
all_ids = tokenizer.batch_encode(["CCO", "c1ccccc1", "CC(=O)O"])

# Save/load vocabulary
tokenizer.save_vocabulary("my_vocab.txt")
tokenizer.load_vocabulary("my_vocab.txt")
```

## Tokenizers

`rustmolbpe` provides four tokenizers spanning a ladder from simplest to most
advanced. They share an identical API, so you can swap one for another and
compare their behavior:

| Class              | Granularity       | Learns merges | Description                                              |
|--------------------|-------------------|---------------|----------------------------------------------------------|
| `CharTokenizer`    | character         | no            | Splits a SMILES string into individual characters        |
| `AtomTokenizer`    | atom (regex)      | no            | Splits into atoms/structural tokens (Br, Cl, [C@@H] kept whole) |
| `CharBPETokenizer` | character         | yes           | BPE merges learned on top of character splitting         |
| `SmilesTokenizer`  | atom (regex)      | yes           | BPE merges learned on top of atom splitting ("SPE")      |

```python
import rustmolbpe

char = rustmolbpe.CharTokenizer()
atom = rustmolbpe.AtomTokenizer()

# Character-level: "Cl" is two tokens ('C', 'l')
len(char.encode("CCl"))   # 3

# Atom-level: chlorine "Cl" is a single token
len(atom.encode("CCl"))   # 2

# BPE tokenizers are trained to merge frequent units
char_bpe = rustmolbpe.CharBPETokenizer()
char_bpe.train_from_iterator(iter(smiles_data), vocab_size=1000)
```

`CharTokenizer` and `AtomTokenizer` have no merges: `train_from_iterator` only
builds the base vocabulary (`vocab_size` is ignored), `num_merges` is always 0,
and vocabulary file I/O is not supported. Use `has_vocabulary()` to check
whether a base vocabulary has been built and `is_trained()` to check for merges.

### Comparing tokenizers on ChEMBL

`tokenizer_stats.py` runs every tokenizer over the ChEMBL 36 dataset and reports
per-molecule token-count statistics (mean, median, std, percentiles) as a
console table, a CSV, and a histogram figure:

```bash
pip install rustmolbpe[stats]          # numpy + matplotlib
python tokenizer_stats.py              # full ChEMBL 36
python tokenizer_stats.py --limit 100000   # quick sample
```

## Special Tokens

The tokenizer includes special tokens for sequence modeling, always at fixed IDs:

| Token | ID | Purpose |
|-------|-----|---------|
| `<pad>` | 0 | Padding for batch processing |
| `<unk>` | 1 | Unknown/out-of-vocabulary atoms |
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 3 | End of sequence |

```python
import rustmolbpe

tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("vocab.txt")

# Access special token IDs
print(tokenizer.pad_token_id)  # 0
print(tokenizer.unk_token_id)  # 1
print(tokenizer.bos_token_id)  # 2
print(tokenizer.eos_token_id)  # 3

# Encode with BOS/EOS tokens
ids = tokenizer.encode("CCO", add_special_tokens=True)
# [2, 667, 3]  ->  [<bos>, CCO, <eos>]

# Unknown atoms are encoded as UNK
ids = tokenizer.encode("C[Xx]C")  # Unknown atom [Xx]
# Contains unk_token_id for unknown atoms
```

## Batch Padding

For training transformer models, sequences need to be padded to equal length:

```python
import rustmolbpe

tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("vocab.txt")

# Encode and pad in one step
smiles_list = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]
result = tokenizer.encode_batch_padded(
    smiles_list,
    max_length=10,           # Pad/truncate to this length
    padding="right",          # "right" or "left"
    truncation=True,          # Truncate sequences longer than max_length
    add_special_tokens=True,  # Add BOS/EOS tokens
    return_attention_mask=True
)

print(result["input_ids"])      # [[2, 667, 3, 0, 0, ...], ...]
print(result["attention_mask"]) # [[1, 1, 1, 0, 0, ...], ...]

# Or pad pre-encoded sequences
sequences = tokenizer.batch_encode(smiles_list)
result = tokenizer.pad(
    sequences,
    max_length=10,
    padding="right",
    truncation=True,
    return_attention_mask=True
)
```

## Atom-level Tokenization

The tokenizer first splits SMILES into atom-level tokens:

```python
import rustmolbpe

# Simple molecule
tokens = rustmolbpe.atomwise_tokenize("CCO")
# ['C', 'C', 'O']

# Halogen atoms
tokens = rustmolbpe.atomwise_tokenize("CBr")
# ['C', 'Br']

# Bracket atoms with charges/stereochemistry
tokens = rustmolbpe.atomwise_tokenize("[C@@H](O)C")
# ['[C@@H]', '(', 'O', ')', 'C']

# Aromatic rings
tokens = rustmolbpe.atomwise_tokenize("c1ccccc1")
# ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']
```

## API Reference

### SmilesTokenizer

```python
class SmilesTokenizer:
    def __init__(self) -> None:
        """Create a new tokenizer with special tokens initialized."""

    def train_from_iterator(
        self,
        iterator: Iterator[str],
        vocab_size: int,
        buffer_size: int = 8192,
        min_frequency: int = 2
    ) -> None:
        """Train the tokenizer from a SMILES iterator."""

    def load_vocabulary(self, path: str) -> None:
        """Load vocabulary from SMILESPE format file."""

    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary to SMILESPE format file."""

    def encode(self, smiles: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a SMILES string to token IDs.

        Args:
            smiles: SMILES string to encode
            add_special_tokens: If True, add BOS/EOS tokens
        """

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to SMILES string."""

    def batch_encode(self, smiles_list: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Encode multiple SMILES in parallel."""

    def batch_decode(self, ids_list: List[List[int]]) -> List[str]:
        """Decode multiple token sequences in parallel."""

    def pad(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding: str = "right",
        truncation: bool = False,
        return_attention_mask: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Pad sequences to equal length.

        Args:
            sequences: List of token ID sequences
            max_length: Target length (default: longest sequence)
            padding: "right" or "left"
            truncation: If True, truncate sequences longer than max_length
            return_attention_mask: If True, include attention_mask in result

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """

    def encode_batch_padded(
        self,
        smiles_list: List[str],
        max_length: Optional[int] = None,
        padding: str = "right",
        truncation: bool = False,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Encode multiple SMILES and pad to equal length.

        Convenience method combining batch_encode and pad.
        """

    # Vocabulary properties
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (special + base atoms + merges)."""

    @property
    def base_vocab_size(self) -> int:
        """Number of base atom tokens."""

    @property
    def num_merges(self) -> int:
        """Number of learned merge operations."""

    # Special token properties
    @property
    def pad_token_id(self) -> int:
        """PAD token ID (always 0)."""

    @property
    def unk_token_id(self) -> int:
        """UNK token ID (always 1)."""

    @property
    def bos_token_id(self) -> int:
        """BOS token ID (always 2)."""

    @property
    def eos_token_id(self) -> int:
        """EOS token ID (always 3)."""

    @property
    def pad_token(self) -> str:
        """PAD token string ('<pad>')."""

    @property
    def unk_token(self) -> str:
        """UNK token string ('<unk>')."""

    @property
    def bos_token(self) -> str:
        """BOS token string ('<bos>')."""

    @property
    def eos_token(self) -> str:
        """EOS token string ('<eos>')."""

    # Vocabulary access
    def get_vocabulary(self) -> List[Tuple[str, int]]:
        """Get vocabulary as (token, id) pairs."""

    def id_to_token(self, id: int) -> str:
        """Convert token ID to token string."""

    def token_to_id(self, token: str) -> int:
        """Convert token string to token ID."""

    # State inspection
    def is_trained(self) -> bool:
        """Check if tokenizer has been trained or has vocabulary loaded."""

    def get_merges(self) -> List[Tuple[str, str, str]]:
        """Get merge rules as (left, right, merged) tuples."""

    # Pickle support (for serialization and multiprocessing)
    # Implements __reduce__ and __setstate__ for full pickle compatibility
```

### Utility Functions

```python
def atomwise_tokenize(smiles: str) -> List[str]:
    """Tokenize SMILES into atom-level tokens."""
```

## Vocabulary Format

The vocabulary file format is compatible with SMILESPE:

```
c c
C C
O )
c 1
= O
...
```

Each line contains two space-separated tokens representing a merge operation.

## HuggingFace Interop

Export a trained tokenizer to the HuggingFace `tokenizers` `tokenizer.json`
format and use it anywhere in the HuggingFace ecosystem:

```python
import rustmolbpe

tok = rustmolbpe.CharBPETokenizer()
tok.train_from_iterator(smiles_generator("chembl.smi"), vocab_size=8000)
tok.save_huggingface("tokenizer.json")

# Load it back into rustmolbpe...
restored = rustmolbpe.CharBPETokenizer.from_huggingface("tokenizer.json")

# ...or use it from `transformers`:
from transformers import PreTrainedTokenizerFast
hf = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

Which tokenizers can be exported, and how they map onto HuggingFace's models:

| Class              | HuggingFace representation                     | Fidelity   |
|--------------------|------------------------------------------------|------------|
| `CharTokenizer`    | `BPE` model, no merges                         | exact      |
| `CharBPETokenizer` | `BPE` model with character merges              | see note   |
| `AtomTokenizer`    | `WordLevel` model + atom-regex `Split`         | exact      |
| `SmilesTokenizer`  | not supported — raises `NotImplementedError`   | —          |

**Notes:**

- `SmilesTokenizer` (atom-level BPE) cannot be expressed as a stock HuggingFace
  fast tokenizer: HuggingFace's `BPE` model only merges single characters within
  a pre-token, so it cannot treat a multi-character atom such as `[C@@H]` as an
  atomic unit. `save_huggingface` raises `NotImplementedError` for it — use
  `save_vocabulary` (SMILESPE format) instead, or `CharBPETokenizer` for an
  exportable BPE tokenizer.
- For `CharBPETokenizer` the vocabulary and merges transfer exactly, but
  HuggingFace applies *merge-order* BPE while rustmolbpe uses *greedy
  longest-match*, so individual token sequences may occasionally differ.
- `from_huggingface` accepts files matching the calling class's granularity and
  merge profile; a mismatch raises `ValueError`.

## Training on Large Datasets

For large datasets like ChEMBL or ZINC:

```python
import rustmolbpe

def smiles_generator(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()

tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.train_from_iterator(
    smiles_generator("chembl.smi"),
    vocab_size=8000,
    buffer_size=16384  # Larger buffer for streaming
)

tokenizer.save_vocabulary("chembl_vocab.txt")
```

## Running Tests

```bash
# Create venv and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install maturin pytest

# Build and run tests
maturin develop
cargo test                    # Rust tests
pytest tests/python/ -v       # Python tests
```

## Performance

rustmolbpe is significantly faster than the original Python SMILESPE implementation.

### Benchmark Results

Benchmarks performed on ChEMBL 36 (~2.8M drug-like molecules) and PubChem (~123M diverse molecules).

#### Encoding Speed (100k SMILES)

| Tokenizer | Dataset | Speed (SMILES/sec) | Avg Tokens | Speedup |
|-----------|---------|-------------------|------------|---------|
| SMILESPE | ChEMBL | 5,583 | 9.3 | 1x |
| rustmolbpe | ChEMBL | 196,964 | 7.6 | **35x** |
| SMILESPE | PubChem | 11,110 | 11.7 | 1x |
| rustmolbpe | PubChem | 279,834 | 6.2 | **25x** |

#### Compression (characters per token, higher = better)

| Tokenizer | ChEMBL | PubChem |
|-----------|--------|---------|
| SMILESPE | 6.26 | 3.34 |
| rustmolbpe (ChEMBL vocab) | **8.16** | 3.69 |
| rustmolbpe (PubChem vocab) | 4.57 | **5.74** |

#### Training Speed (50k SMILES, vocab_size=1000)

| Dataset | SMILESPE | rustmolbpe | Speedup |
|---------|----------|------------|---------|
| ChEMBL | 16.8s | 0.96s | **18x** |
| PubChem | 11.1s | 0.70s | **16x** |

#### Large-Scale Training

| Dataset | Molecules | Training Time | Vocab Size |
|---------|-----------|---------------|------------|
| ChEMBL 36 | 2.8M | 102s | 8,000 |
| PubChem | 10M | 407s (~7 min) | 8,000 |

### Pre-trained Vocabularies

Pre-trained vocabularies are available in the `data/` directory:

- `chembl36_vocab.txt` - Trained on ChEMBL 36 (2.8M drug-like molecules, 7,715 merges)
- `pubchem_10M_vocab.txt` - Trained on PubChem (10M diverse molecules, 6,385 merges)

```python
import rustmolbpe

# Load pre-trained ChEMBL vocabulary
tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("data/chembl36_vocab.txt")

# Encode drug molecules efficiently
ids = tokenizer.encode("CC(=O)Nc1ccc(O)cc1")  # paracetamol
```

### Running Benchmarks

```bash
# Run the benchmark script
python benchmark.py
```

## Datasets

### Downloading Training Data

**ChEMBL** (drug-like molecules):
```bash
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_chemreps.txt.gz
gunzip chembl_36_chemreps.txt.gz
```

**PubChem** (diverse molecules):
```bash
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz
```

## Troubleshooting

### Installation Issues

**"maturin not found"**
```bash
uv pip install maturin
```

**"Rust compiler not found"**
Install Rust from https://rustup.rs/:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**"uv not found"**
Install uv from https://docs.astral.sh/uv/:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Build fails with "VIRTUAL_ENV and CONDA_PREFIX both set"**
```bash
unset CONDA_PREFIX && maturin develop --release
```

### Runtime Issues

**"Unknown token" error when decoding**
The token ID is not in the vocabulary. This can happen if:
- You're using a different vocabulary than the one used for encoding
- The ID is out of range

```python
# Check vocabulary size
print(tokenizer.vocab_size)

# Verify token exists
try:
    token = tokenizer.id_to_token(some_id)
except ValueError:
    print(f"ID {some_id} not in vocabulary")
```

**Unknown atoms encoded as UNK**
Atoms not seen during training are encoded as UNK (ID 1). Train on a larger/more diverse dataset or use a pre-trained vocabulary.

```python
# Check if a SMILES contains unknown atoms
ids = tokenizer.encode("C[Xe]C")  # Xenon might be unknown
if tokenizer.unk_token_id in ids:
    print("Contains unknown atoms")
```

**Pickle/multiprocessing errors**
Ensure you're using rustmolbpe >= 0.2.0 which includes pickle support.

## Citation

If you use rustmolbpe in your research, please cite it:

```bibtex
@software{rustmolbpe,
  author = {Fooladi, Hosein},
  title = {rustmolbpe: A High-Performance BPE Tokenizer for Molecular SMILES},
  url = {https://github.com/HFooladi/rustmolbpe},
  year = {2026}
}
```

## License

MIT
