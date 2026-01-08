# rustmolbpe

A high-performance BPE (Byte Pair Encoding) tokenizer for molecular SMILES written in Rust with Python bindings.

## Features

- **SMILES-aware tokenization**: Correctly handles multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures, and stereochemistry
- **Fast training**: Parallel processing with Rayon for efficient training on large molecular datasets
- **Streaming support**: Train on datasets of any size with configurable buffer sizes
- **Special tokens**: Built-in PAD, UNK, BOS, EOS tokens for sequence modeling
- **Batch padding**: Ready for transformer models with attention masks
- **SMILESPE compatibility**: Load and save vocabularies in SMILESPE format

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

### From PyPI (coming soon)

```bash
pip install rustmolbpe
```

### From source

```bash
# Install maturin
pip install maturin

# Clone and build
git clone https://github.com/HFooladi/rustmolbpe.git
cd rustmolbpe
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
print(ids)  # [2864, 1077]

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

Pre-trained vocabularies are included:

- `data/chembl36_vocab.txt` - Trained on ChEMBL 36 (2.8M drug-like molecules)
- `data/pubchem_10M_vocab.txt` - Trained on PubChem (10M diverse molecules)

## License

MIT License - see [LICENSE](https://github.com/HFooladi/rustmolbpe/blob/main/LICENSE)
