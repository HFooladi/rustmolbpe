# API Reference

## Module: rustmolbpe

### atomwise_tokenize

```python
def atomwise_tokenize(smiles: str) -> List[str]
```

Tokenize a SMILES string into atom-level tokens.

**Arguments:**

- `smiles` (str): SMILES string to tokenize

**Returns:**

- `List[str]`: List of atom-level tokens

**Examples:**

```python
>>> rustmolbpe.atomwise_tokenize("CCO")
['C', 'C', 'O']

>>> rustmolbpe.atomwise_tokenize("c1ccccc1")
['c', '1', 'c', 'c', 'c', 'c', 'c', '1']

>>> rustmolbpe.atomwise_tokenize("[C@@H](O)C")
['[C@@H]', '(', 'O', ')', 'C']

>>> rustmolbpe.atomwise_tokenize("CBr")
['C', 'Br']
```

---

## Tokenizer Classes

rustmolbpe exports five tokenizer classes. They share the API documented in
[Shared API](#shared-api) below and differ in pre-tokenization granularity and
in whether BPE merges are learned:

| Class              | Granularity  | Learns merges | Notes                                                        |
|--------------------|--------------|---------------|--------------------------------------------------------------|
| `CharTokenizer`    | character    | no            | One token per Unicode character                              |
| `AtomTokenizer`    | atom (regex) | no            | Multi-character atoms (`Br`, `Cl`, `[C@@H]`) kept whole      |
| `CharBPETokenizer` | character    | yes           | BPE on characters                                            |
| `SmilesTokenizer`  | atom (regex) | yes           | BPE on atoms ("SPE"); also exported as `AtomBPETokenizer`    |
| `ByteBPETokenizer` | byte (UTF-8) | yes           | BPE on raw bytes; lossless round-trip, no `<unk>` once trained |

> **Alias:** `AtomBPETokenizer` and `SmilesTokenizer` bind to the same class
> object — `rustmolbpe.AtomBPETokenizer is rustmolbpe.SmilesTokenizer`.

Class-specific behavior:

- **`CharTokenizer`, `AtomTokenizer`**: `train_from_iterator` only builds the
  base vocabulary; `vocab_size` is ignored and `num_merges` is always 0.
  `load_vocabulary` / `save_vocabulary` raise `NotImplementedError`.
- **`ByteBPETokenizer`**: the base alphabet is always the 256 byte values
  (`base_vocab_size == 260` once trained). `load_vocabulary`, `save_vocabulary`,
  `save_huggingface` and `from_huggingface` raise `NotImplementedError`; use
  [`save` / `from_file`](#tokenizer-file) to persist it.
- **`SmilesTokenizer`**: `save_huggingface` raises `NotImplementedError`
  (atom-level BPE cannot be expressed as a stock HuggingFace fast tokenizer).

A pickle can only be restored into a tokenizer of matching granularity (atom,
character or byte); restoring it into a tokenizer of a different granularity
raises `ValueError`.

---

## Shared API

Every tokenizer class provides the methods below. Examples use
`SmilesTokenizer`; unless noted otherwise, the other classes behave the same.

### Constructor

```python
def __init__(self) -> None
```

Create a new tokenizer with special tokens initialized.

Special tokens are always at fixed IDs:

| Token | ID | String |
|-------|-----|--------|
| PAD | 0 | `<pad>` |
| UNK | 1 | `<unk>` |
| BOS | 2 | `<bos>` |
| EOS | 3 | `<eos>` |

---

### Training Methods

#### train_from_iterator

```python
def train_from_iterator(
    self,
    iterator: Iterator[str],
    vocab_size: int,
    buffer_size: int = 8192,
    min_frequency: int = 2
) -> None
```

Train the tokenizer from a SMILES iterator.

**Arguments:**

- `iterator` (Iterator[str]): Iterator yielding SMILES strings
- `vocab_size` (int): Target vocabulary size (including special tokens and base atoms)
- `buffer_size` (int, optional): Number of SMILES to buffer for parallel processing. Default: 8192
- `min_frequency` (int, optional): Minimum number of occurrences (summed over the whole corpus) a pair needs to be merged; training stops early once no remaining pair reaches it. Default: 2

**Example:**

```python
def smiles_generator(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.train_from_iterator(
    smiles_generator("molecules.smi"),
    vocab_size=8000
)
```

---

### Vocabulary I/O

#### load_vocabulary

```python
def load_vocabulary(self, path: str) -> None
```

Load vocabulary from a SMILESPE-format file. Supported by `CharBPETokenizer`
and `SmilesTokenizer`.

The SMILESPE format stores merge rules only, in priority order. Loading assigns
token IDs that generally differ from the tokenizer that saved the file, and base
tokens that never took part in a merge are not included. Use
[`save` / `from_file`](#tokenizer-file) to persist a tokenizer with identical IDs.

**Arguments:**

- `path` (str): Path to vocabulary file

**Raises:**

- `IOError`: If file cannot be read
- `NotImplementedError`: For `CharTokenizer`, `AtomTokenizer` and `ByteBPETokenizer`

#### save_vocabulary

```python
def save_vocabulary(self, path: str) -> None
```

Save vocabulary to a SMILESPE-format file. Supported by `CharBPETokenizer`
and `SmilesTokenizer`.

**Arguments:**

- `path` (str): Path to save vocabulary file

**Raises:**

- `IOError`: If file cannot be written
- `NotImplementedError`: For `CharTokenizer`, `AtomTokenizer` and `ByteBPETokenizer`

---

### Tokenizer File

#### save

```python
def save(self, path: str) -> None
```

Save the complete tokenizer to a lossless JSON file: the full vocabulary with its
token IDs, the merges in priority order, and the pre-tokenizer granularity.
Supported by every tokenizer class. Prefer this over `save_vocabulary` to persist
a tokenizer used by a trained model.

**Raises:**

- `IOError`: If the file cannot be written

#### from_file

```python
@classmethod
def from_file(cls, path: str) -> Self
```

Load a tokenizer saved with `save`. Token IDs, merges and encodings are identical
to the saved tokenizer.

**Raises:**

- `IOError`: If the file cannot be read
- `ValueError`: If the file is not a valid rustmolbpe tokenizer file, has an unsupported version, was saved by a tokenizer of a different granularity, or contains merges and the calling class learns none

**Example:**

```python
tokenizer.save("tokenizer.json")
restored = rustmolbpe.SmilesTokenizer.from_file("tokenizer.json")
assert restored.get_vocabulary() == tokenizer.get_vocabulary()
```

**File layout** (JSON):

```json
{
  "format": "rustmolbpe",
  "version": 1,
  "pretokenizer": "atom",
  "vocab": ["<pad>", "<unk>", "<bos>", "<eos>", "c", "C", "..."],
  "merges": [[4, 4, 57], [5, 5, 58]]
}
```

`vocab` lists token strings (the index is the token ID); `merges` lists
`[left_id, right_id, merged_id]` in priority order; `pretokenizer` is `atom`,
`char` or `byte`.

---

### HuggingFace Interop

#### save_huggingface

```python
def save_huggingface(self, path: str) -> None
```

Export the tokenizer to a HuggingFace `tokenizers` JSON file, loadable by
`transformers.PreTrainedTokenizerFast(tokenizer_file=...)` and the `tokenizers`
library.

| Class              | HuggingFace representation                 |
|--------------------|--------------------------------------------|
| `CharTokenizer`    | `BPE` model, no merges                     |
| `CharBPETokenizer` | `BPE` model with character merges          |
| `AtomTokenizer`    | `WordLevel` model + atom-regex `Split`     |
| `SmilesTokenizer`  | not supported                              |
| `ByteBPETokenizer` | not supported                              |

HuggingFace applies *merge-order* BPE while rustmolbpe uses *greedy
longest-match*, so for `CharBPETokenizer` the vocabulary and merges transfer
exactly but individual token sequences may occasionally differ.

**Arguments:**

- `path` (str): Path to write the `tokenizer.json` file

**Raises:**

- `IOError`: If the file cannot be written
- `NotImplementedError`: For `SmilesTokenizer` and `ByteBPETokenizer`

#### from_huggingface

```python
@classmethod
def from_huggingface(cls, path: str) -> Self
```

Load a tokenizer from a HuggingFace `tokenizers` JSON file written by
`save_huggingface` (or a compatible character-level `BPE` / atom-level
`WordLevel` tokenizer).

**Arguments:**

- `path` (str): Path to a `tokenizer.json` file

**Returns:**

- A new tokenizer instance of the calling class

**Raises:**

- `IOError`: If the file cannot be read
- `ValueError`: If the file is malformed, or its granularity / merge profile does not match the calling class (always the case for `SmilesTokenizer`, which has no HuggingFace representation)
- `NotImplementedError`: For `ByteBPETokenizer`

**Example:**

```python
tok = rustmolbpe.CharBPETokenizer()
tok.train_from_iterator(smiles_generator("molecules.smi"), vocab_size=8000)
tok.save_huggingface("tokenizer.json")

restored = rustmolbpe.CharBPETokenizer.from_huggingface("tokenizer.json")
rustmolbpe.AtomTokenizer.from_huggingface("tokenizer.json")  # ValueError
```

---

### Encoding Methods

#### encode

```python
def encode(self, smiles: str, add_special_tokens: bool = False) -> List[int]
```

Encode a SMILES string to token IDs.

**Arguments:**

- `smiles` (str): SMILES string to encode
- `add_special_tokens` (bool, optional): If True, add BOS at start and EOS at end. Default: False

**Returns:**

- `List[int]`: List of token IDs

**Example** (with `data/chembl36_vocab.txt` loaded):

```python
ids = tokenizer.encode("CCO")  # [353]
ids = tokenizer.encode("CCO", add_special_tokens=True)  # [2, 353, 3]
```

#### batch_encode

```python
def batch_encode(
    self,
    smiles_list: List[str],
    add_special_tokens: bool = False
) -> List[List[int]]
```

Encode multiple SMILES strings in parallel.

**Arguments:**

- `smiles_list` (List[str]): List of SMILES strings
- `add_special_tokens` (bool, optional): If True, add BOS/EOS tokens. Default: False

**Returns:**

- `List[List[int]]`: List of token ID lists

#### \_\_call\_\_

```python
def __call__(
    self,
    text: str | List[str],
    padding: bool = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
    add_special_tokens: bool = False,
    return_attention_mask: bool = True
) -> Dict[str, Any]
```

Encode one or more SMILES strings with a HuggingFace-style call interface.

**Arguments:**

- `text` (str or List[str]): A SMILES string or a list of SMILES strings
- `padding` (bool, optional): If True, right-pad sequences to equal length. Default: False
- `truncation` (bool, optional): If True, truncate sequences to `max_length`. Default: False
- `max_length` (int, optional): Maximum sequence length for padding/truncation
- `add_special_tokens` (bool, optional): If True, add BOS/EOS tokens. Default: False
- `return_attention_mask` (bool, optional): If True, include `attention_mask`. Default: True

**Returns:**

- `Dict[str, Any]`: `"input_ids"` and optionally `"attention_mask"` — flat lists for a single string, nested lists for a list input

**Example** (with `data/chembl36_vocab.txt` loaded):

```python
tokenizer("CCO", add_special_tokens=True)
# {'input_ids': [2, 353, 3], 'attention_mask': [1, 1, 1]}

tokenizer(["CCO", "c1ccccc1C"], padding=True)
# {'input_ids': [[353, 0], [782, 155]], 'attention_mask': [[1, 0], [1, 1]]}
```

---

### Decoding Methods

#### decode

```python
def decode(self, ids: List[int]) -> str
```

Decode token IDs back to a SMILES string.

**Arguments:**

- `ids` (List[int]): List of token IDs

**Returns:**

- `str`: Decoded SMILES string

**Raises:**

- `ValueError`: If an ID is not in vocabulary

#### batch_decode

```python
def batch_decode(self, ids_list: List[List[int]]) -> List[str]
```

Decode multiple token sequences in parallel.

**Arguments:**

- `ids_list` (List[List[int]]): List of token ID lists

**Returns:**

- `List[str]`: List of decoded SMILES strings

---

### Padding Methods

#### pad

```python
def pad(
    self,
    sequences: List[List[int]],
    max_length: Optional[int] = None,
    padding: str = "right",
    truncation: bool = False,
    return_attention_mask: bool = True
) -> Dict[str, List[List[int]]]
```

Pad sequences to equal length.

**Arguments:**

- `sequences` (List[List[int]]): List of token ID sequences
- `max_length` (int, optional): Target length. If None, uses longest sequence length
- `padding` (str, optional): Padding side, either "right" or "left". Default: "right"
- `truncation` (bool, optional): If True, truncate sequences longer than max_length. Default: False
- `return_attention_mask` (bool, optional): If True, include attention_mask in result. Default: True

**Returns:**

- `Dict[str, List[List[int]]]`: Dictionary with keys:
    - `"input_ids"`: Padded token ID sequences
    - `"attention_mask"`: Attention masks (if requested)

#### encode_batch_padded

```python
def encode_batch_padded(
    self,
    smiles_list: List[str],
    max_length: Optional[int] = None,
    padding: str = "right",
    truncation: bool = False,
    add_special_tokens: bool = False,
    return_attention_mask: bool = True
) -> Dict[str, List[List[int]]]
```

Encode multiple SMILES and pad to equal length.

Convenience method combining `batch_encode` and `pad`.

**Arguments:**

- `smiles_list` (List[str]): List of SMILES strings
- `max_length` (int, optional): Target length. If None, uses longest sequence length
- `padding` (str, optional): Padding side, either "right" or "left". Default: "right"
- `truncation` (bool, optional): If True, truncate sequences longer than max_length. Default: False
- `add_special_tokens` (bool, optional): If True, add BOS/EOS tokens. Default: False
- `return_attention_mask` (bool, optional): If True, include attention_mask in result. Default: True

**Returns:**

- `Dict[str, List[List[int]]]`: Dictionary with "input_ids" and optionally "attention_mask"

**Example** (with `data/chembl36_vocab.txt` loaded):

```python
result = tokenizer.encode_batch_padded(
    ["CCO", "c1ccccc1"],
    max_length=6,
    add_special_tokens=True
)
print(result["input_ids"])       # [[2, 353, 3, 0, 0, 0], [2, 782, 3, 0, 0, 0]]
print(result["attention_mask"])  # [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]
```

---

### Vocabulary Access

#### get_vocabulary

```python
def get_vocabulary(self) -> List[Tuple[str, int]]
```

Get vocabulary as (token, id) pairs.

**Returns:**

- `List[Tuple[str, int]]`: List of (token_string, token_id) tuples

#### id_to_token

```python
def id_to_token(self, id: int) -> str
```

Convert token ID to token string.

**Arguments:**

- `id` (int): Token ID

**Returns:**

- `str`: Token string

**Raises:**

- `ValueError`: If ID is not in vocabulary

#### token_to_id

```python
def token_to_id(self, token: str) -> int
```

Convert token string to token ID.

**Arguments:**

- `token` (str): Token string

**Returns:**

- `int`: Token ID

**Raises:**

- `ValueError`: If token is not in vocabulary

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `vocab_size` | int | Total vocabulary size (special + base atoms + merges) |
| `base_vocab_size` | int | Number of base tokens, including the 4 special tokens (always 260 for a trained `ByteBPETokenizer`) |
| `num_merges` | int | Number of learned merge operations |
| `pad_token_id` | int | PAD token ID (always 0) |
| `unk_token_id` | int | UNK token ID (always 1) |
| `bos_token_id` | int | BOS token ID (always 2) |
| `eos_token_id` | int | EOS token ID (always 3) |
| `pad_token` | str | PAD token string (`<pad>`) |
| `unk_token` | str | UNK token string (`<unk>`) |
| `bos_token` | str | BOS token string (`<bos>`) |
| `eos_token` | str | EOS token string (`<eos>`) |

---

### State Inspection

#### is_trained

```python
def is_trained(self) -> bool
```

Check whether the tokenizer has learned (or loaded) BPE merges. Always False for
`CharTokenizer` and `AtomTokenizer`; use `has_vocabulary` for those.

**Returns:**

- `bool`: True if the tokenizer has merge rules, False otherwise

**Example:**

```python
tokenizer = rustmolbpe.SmilesTokenizer()
print(tokenizer.is_trained())  # False

tokenizer.train_from_iterator(iter(["CCO", "CCC"]), vocab_size=50)
print(tokenizer.is_trained())  # True
```

#### has_vocabulary

```python
def has_vocabulary(self) -> bool
```

Check whether a base vocabulary has been built.

**Returns:**

- `bool`: True if the vocabulary contains tokens beyond the 4 special tokens

**Example:**

```python
tokenizer = rustmolbpe.AtomTokenizer()
tokenizer.train_from_iterator(iter(["CCO", "CCl"]), vocab_size=0)
print(tokenizer.has_vocabulary())  # True
print(tokenizer.is_trained())      # False (no merges)
```

#### get_merges

```python
def get_merges(self) -> List[Tuple[str, str, str]]
```

Get the learned merge rules as tuples.

**Returns:**

- `List[Tuple[str, str, str]]`: List of (left_token, right_token, merged_token) tuples, ordered by merge priority. Empty for `CharTokenizer` and `AtomTokenizer`

**Example:**

```python
tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.train_from_iterator(iter(["CCO", "CCN", "CCO", "c1ccccc1"]), vocab_size=60)

merges = tokenizer.get_merges()
print(merges[:3])  # First 3 merge rules
# [('c', 'c', 'cc'), ('C', 'C', 'CC'), ('c', '1', 'c1')]
```

---

### Serialization

#### Pickle Support

Every tokenizer class supports Python's pickle protocol, which multiprocessing
also relies on. To store a tokenizer on disk, prefer
[`save` / `from_file`](#tokenizer-file): the file is readable JSON and, unlike a
pickle, safe to load from an untrusted source.

```python
import pickle

tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("data/chembl36_vocab.txt")

# Save to bytes
data = pickle.dumps(tokenizer)

# Restore from bytes
restored = pickle.loads(data)

# Verify
assert tokenizer.encode("CCO") == restored.encode("CCO")
```

**Multiprocessing Example:**

```python
from multiprocessing import Pool
import rustmolbpe

def encode_smiles(smiles):
    # tokenizer is pickled and sent to worker processes
    return tokenizer.encode(smiles)

tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("data/chembl36_vocab.txt")

smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
with Pool(4) as pool:
    results = pool.map(encode_smiles, smiles_list)
```
