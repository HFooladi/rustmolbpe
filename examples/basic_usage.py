#!/usr/bin/env python3
"""Basic usage example for rustmolbpe.

This script demonstrates:
- Loading a pre-trained vocabulary
- Encoding and decoding SMILES strings
- Using special tokens
- Accessing vocabulary information
"""

import rustmolbpe


def main():
    # Create tokenizer and load pre-trained vocabulary
    tokenizer = rustmolbpe.SmilesTokenizer()
    tokenizer.load_vocabulary("data/chembl36_vocab.txt")

    print("=" * 60)
    print("rustmolbpe Basic Usage Example")
    print("=" * 60)

    # Vocabulary info
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Base atoms: {tokenizer.base_vocab_size}")
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
        print(f"  With BOS/EOS: {ids_special}")
        print(f"  Decoded: {decoded}")
        print(f"  Tokens: {len(ids)} | Compression: {len(smiles)/len(ids):.2f} chars/token")

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
    if token in [t for t, _ in vocab]:
        token_id = tokenizer.token_to_id(token)
        print(f"\nToken '{token}' -> ID {token_id}")
        print(f"ID {token_id} -> Token '{tokenizer.id_to_token(token_id)}'")


if __name__ == "__main__":
    main()
