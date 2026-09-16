#!/usr/bin/env python3
"""Basic usage example for rustmolbpe.

This script demonstrates:
- Loading a pre-trained vocabulary
- Encoding and decoding SMILES strings
- Using special tokens
- Accessing vocabulary information

Usage:
    python examples/basic_usage.py [VOCAB_PATH]

VOCAB_PATH defaults to the repository's data/chembl36_vocab.txt.
"""

import sys
from pathlib import Path

import rustmolbpe

VOCAB_URL = "https://github.com/HFooladi/rustmolbpe/blob/main/data/chembl36_vocab.txt"


def load_pretrained(vocab_path=None):
    """Load the pre-trained ChEMBL 36 vocabulary into a SmilesTokenizer."""
    default = Path(__file__).resolve().parent.parent / "data" / "chembl36_vocab.txt"
    vocab = Path(vocab_path) if vocab_path else default
    if not vocab.is_file():
        sys.exit(
            f"Pre-trained vocabulary not found: {vocab}\n"
            "Run this example from a clone of the repository, or download "
            f"chembl36_vocab.txt from {VOCAB_URL}\n"
            f"and pass its path: python {Path(__file__).name} path/to/chembl36_vocab.txt"
        )
    tokenizer = rustmolbpe.SmilesTokenizer()
    tokenizer.load_vocabulary(str(vocab))
    return tokenizer


def main():
    tokenizer = load_pretrained(sys.argv[1] if len(sys.argv) > 1 else None)

    print("=" * 60)
    print("rustmolbpe Basic Usage Example")
    print("=" * 60)

    # Vocabulary info
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Base tokens: {tokenizer.base_vocab_size}")
    print(f"Learned merges: {tokenizer.num_merges}")

    # Special tokens
    print("\nSpecial tokens:")
    print(f"  PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  UNK: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"  BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

    # Example molecules
    molecules = [
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ]

    print("\n" + "-" * 60)
    print("Encoding Examples")
    print("-" * 60)

    for name, smiles in molecules:
        # Encode without special tokens
        ids = tokenizer.encode(smiles)

        # Encode with special tokens (BOS/EOS)
        ids_special = tokenizer.encode(smiles, add_special_tokens=True)

        # Decode back
        decoded = tokenizer.decode(ids)

        print(f"\n{name}: {smiles}")
        print(f"  Token IDs: {ids}")
        print(f"  Tokens: {[tokenizer.id_to_token(i) for i in ids]}")
        print(f"  With BOS/EOS: {ids_special}")
        print(f"  Decoded: {decoded}")
        print(f"  {len(ids)} tokens | {len(smiles) / len(ids):.2f} chars/token")

    # Atom-level tokenization
    print("\n" + "-" * 60)
    print("Atom-level Tokenization")
    print("-" * 60)

    smiles = "[C@@H](O)(F)Cl"
    atoms = rustmolbpe.atomwise_tokenize(smiles)
    print(f"\nSMILES: {smiles}")
    print(f"Atoms: {atoms}")

    # Token lookup
    print("\n" + "-" * 60)
    print("Token Lookup")
    print("-" * 60)

    print("\nFirst 10 tokens in vocabulary:")
    vocab = tokenizer.get_vocabulary()
    for token, token_id in vocab[:10]:
        print(f"  ID {token_id}: '{token}'")

    # Bidirectional lookup
    token = "CC"
    token_id = tokenizer.token_to_id(token)
    print(f"\nToken '{token}' -> ID {token_id}")
    print(f"ID {token_id} -> Token '{tokenizer.id_to_token(token_id)}'")


if __name__ == "__main__":
    main()
