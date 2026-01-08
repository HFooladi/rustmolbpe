#!/usr/bin/env python3
"""Training example for rustmolbpe.

This script demonstrates:
- Training a BPE tokenizer from scratch
- Using iterators for streaming large datasets
- Saving and loading vocabularies
- Comparing compression ratios
"""

import rustmolbpe
import time


# Sample SMILES dataset (drug-like molecules)
SAMPLE_SMILES = [
    # Simple molecules
    "CCO",  # ethanol
    "CCCO",  # propanol
    "CCCCO",  # butanol
    "CC(C)O",  # isopropanol
    "CC(=O)O",  # acetic acid
    "CC(=O)OC",  # methyl acetate
    # Aromatic compounds
    "c1ccccc1",  # benzene
    "Cc1ccccc1",  # toluene
    "c1ccc(cc1)O",  # phenol
    "c1ccc(cc1)N",  # aniline
    "c1ccc(cc1)C(=O)O",  # benzoic acid
    # Drug-like molecules
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # caffeine
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine (alternative)
    # Heterocycles
    "c1ccncc1",  # pyridine
    "c1ccc2[nH]ccc2c1",  # indole
    "c1ccc2c(c1)nc[nH]2",  # benzimidazole
    "c1cnc2ccccc2n1",  # quinazoline
    # Charged/complex
    "[NH4+]",  # ammonium
    "[O-][N+](=O)c1ccccc1",  # nitrobenzene
    "C[N+](C)(C)CCO",  # choline
    # Stereochemistry
    "C[C@H](O)CC",  # (R)-2-butanol
    "C[C@@H](O)CC",  # (S)-2-butanol
    "C/C=C/C",  # trans-2-butene
    "C/C=C\\C",  # cis-2-butene
]


def smiles_generator(smiles_list, repeat=100):
    """Generator that yields SMILES strings.

    In real use, this would read from a file.
    """
    for _ in range(repeat):
        for smiles in smiles_list:
            yield smiles


def train_example():
    """Train a tokenizer on sample data."""
    print("=" * 60)
    print("Training a BPE Tokenizer")
    print("=" * 60)

    # Create tokenizer
    tokenizer = rustmolbpe.SmilesTokenizer()
    print(f"\nNew tokenizer vocab size: {tokenizer.vocab_size} (special tokens only)")

    # Training parameters
    vocab_size = 100
    total_smiles = len(SAMPLE_SMILES) * 100  # repeated 100 times

    print(f"\nTraining on {total_smiles} SMILES...")
    print(f"Target vocab size: {vocab_size}")

    # Train
    start = time.perf_counter()
    tokenizer.train_from_iterator(
        smiles_generator(SAMPLE_SMILES, repeat=100),
        vocab_size=vocab_size,
        min_frequency=2,
    )
    elapsed = time.perf_counter() - start

    print(f"Training completed in {elapsed:.2f}s")
    print(f"\nFinal vocabulary:")
    print(f"  Total size: {tokenizer.vocab_size}")
    print(f"  Base atoms: {tokenizer.base_vocab_size}")
    print(f"  Merges: {tokenizer.num_merges}")

    return tokenizer


def compression_analysis(tokenizer):
    """Analyze compression on test molecules."""
    print("\n" + "=" * 60)
    print("Compression Analysis")
    print("=" * 60)

    test_molecules = [
        "CCO",
        "c1ccccc1",
        "CC(=O)Nc1ccc(O)cc1",
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    ]

    total_chars = 0
    total_tokens = 0

    print(f"\n{'SMILES':<40} {'Chars':<8} {'Tokens':<8} {'Ratio':<8}")
    print("-" * 64)

    for smiles in test_molecules:
        ids = tokenizer.encode(smiles)
        chars = len(smiles)
        tokens = len(ids)
        ratio = chars / tokens if tokens > 0 else 0

        total_chars += chars
        total_tokens += tokens

        print(f"{smiles:<40} {chars:<8} {tokens:<8} {ratio:<8.2f}")

    avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    print("-" * 64)
    print(f"{'Average':<40} {'':<8} {'':<8} {avg_ratio:<8.2f}")


def save_load_example(tokenizer):
    """Demonstrate saving and loading vocabulary."""
    print("\n" + "=" * 60)
    print("Save/Load Vocabulary")
    print("=" * 60)

    import tempfile
    import os

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        vocab_path = f.name

    try:
        # Save
        tokenizer.save_vocabulary(vocab_path)
        print(f"\nSaved vocabulary to: {vocab_path}")

        # Check file size
        size = os.path.getsize(vocab_path)
        print(f"File size: {size} bytes")

        # Show first few lines
        print("\nFirst 5 merge rules:")
        with open(vocab_path) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {line.strip()}")

        # Load into new tokenizer
        new_tokenizer = rustmolbpe.SmilesTokenizer()
        new_tokenizer.load_vocabulary(vocab_path)
        print(f"\nLoaded vocabulary: {new_tokenizer.vocab_size} tokens")

        # Verify
        test_smiles = "CC(=O)Nc1ccc(O)cc1"
        original_ids = tokenizer.encode(test_smiles)
        loaded_ids = new_tokenizer.encode(test_smiles)

        print(f"\nVerification with '{test_smiles}':")
        print(f"  Original: {original_ids}")
        print(f"  Loaded:   {loaded_ids}")
        print(f"  Match: {original_ids == loaded_ids}")

    finally:
        os.unlink(vocab_path)


def main():
    # Train tokenizer
    tokenizer = train_example()

    # Analyze compression
    compression_analysis(tokenizer)

    # Save/load demo
    save_load_example(tokenizer)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
