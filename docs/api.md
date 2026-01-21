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

## Class: SmilesTokenizer

BPE tokenizer for molecular SMILES strings.

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
- `min_frequency` (int, optional): Minimum frequency for a pair to be merged. Default: 2

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

Load vocabulary from a SMILESPE-format file.

**Arguments:**

- `path` (str): Path to vocabulary file

**Raises:**

- `IOError`: If file cannot be read

#### save_vocabulary

```python
def save_vocabulary(self, path: str) -> None
```

Save vocabulary to a SMILESPE-format file.

**Arguments:**

- `path` (str): Path to save vocabulary file

**Raises:**

- `IOError`: If file cannot be written

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

**Example:**

```python
ids = tokenizer.encode("CCO")  # [42]
ids = tokenizer.encode("CCO", add_special_tokens=True)  # [2, 42, 3]
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

**Example:**

```python
result = tokenizer.encode_batch_padded(
    ["CCO", "c1ccccc1"],
    max_length=10,
    add_special_tokens=True
)
print(result["input_ids"])       # [[2, 42, 3, 0, 0, ...], [2, 15, 3, 0, 0, ...]]
print(result["attention_mask"])  # [[1, 1, 1, 0, 0, ...], [1, 1, 1, 0, 0, ...]]
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
| `base_vocab_size` | int | Number of base atom tokens |
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

Check if the tokenizer has been trained or has a vocabulary loaded.

**Returns:**

- `bool`: True if the tokenizer has merge rules, False otherwise

**Example:**

```python
tokenizer = rustmolbpe.SmilesTokenizer()
print(tokenizer.is_trained())  # False

tokenizer.train_from_iterator(iter(["CCO", "CCC"]), vocab_size=50)
print(tokenizer.is_trained())  # True
```

#### get_merges

```python
def get_merges(self) -> List[Tuple[str, str, str]]
```

Get the learned merge rules as tuples.

**Returns:**

- `List[Tuple[str, str, str]]`: List of (left_token, right_token, merged_token) tuples, ordered by merge priority

**Example:**

```python
tokenizer = rustmolbpe.SmilesTokenizer()
tokenizer.load_vocabulary("data/chembl36_vocab.txt")

merges = tokenizer.get_merges()
print(merges[:3])  # First 3 merge rules
# [('c', 'c', 'cc'), ('C', 'C', 'CC'), ('c', '1', 'c1')]
```

---

### Serialization

#### Pickle Support

`SmilesTokenizer` supports Python's pickle protocol for serialization.

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
