#!/usr/bin/env python3
"""Training example for rustmolbpe.

This script demonstrates:
- Training a BPE tokenizer from scratch
- Using iterators for streaming large datasets
- Saving and reloading the trained tokenizer with identical token IDs
- Exporting merge rules in the SMILESPE format
- Comparing compression ratios

Runs anywhere: the training data is included in the script.
"""

import os
import tempfile
import time

import rustmolbpe

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

    In real use, this would read from a file, one SMILES per line, so the
    dataset never has to fit in memory.
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

    # Train. min_frequency is the minimum number of times a pair must occur
    # across the corpus to be merged.
    start = time.perf_counter()
    tokenizer.train_from_iterator(
        smiles_generator(SAMPLE_SMILES, repeat=100),
        vocab_size=vocab_size,
        min_frequency=2,
    )
    elapsed = time.perf_counter() - start

    print(f"Training completed in {elapsed:.2f}s")
    print("\nFinal vocabulary:")
    print(f"  Total size: {tokenizer.vocab_size}")
    print(f"  Base tokens: {tokenizer.base_vocab_size}")
    print(f"  Merges: {tokenizer.num_merges}")
    print(f"  First 5 merges (learning order): {tokenizer.get_merges()[:5]}")

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


def save_and_reload(tokenizer):
    """Persist the trained tokenizer and load it back."""
    print("\n" + "=" * 60)
    print("Saving and Reloading")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        # save() writes the complete tokenizer (vocabulary with its IDs, merges,
        # granularity) to a JSON file; from_file() restores it exactly. Use this
        # to keep a tokenizer together with a model trained on its token IDs.
        path = os.path.join(tmp, "smiles_tokenizer.json")
        tokenizer.save(path)
        print(f"\nSaved tokenizer to {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")

        restored = rustmolbpe.SmilesTokenizer.from_file(path)

        test_smiles = "CC(=O)Nc1ccc(O)cc1"
        original_ids = tokenizer.encode(test_smiles)
        restored_ids = restored.encode(test_smiles)
        identical = (
            restored_ids == original_ids
            and restored.get_vocabulary() == tokenizer.get_vocabulary()
        )
        print(f"\nVerification with '{test_smiles}':")
        print(f"  Original: {original_ids}")
        print(f"  Restored: {restored_ids}")
        print(f"Token IDs identical after save()/from_file(): {identical}")

        # save_vocabulary() writes merge rules in the SMILESPE text format, for
        # exchanging vocabularies with other tools. It does not store token IDs:
        # loading the file assigns new IDs, so prefer save()/from_file() for a
        # tokenizer tied to a trained model.
        merges_path = os.path.join(tmp, "merges.txt")
        tokenizer.save_vocabulary(merges_path)
        print(f"\nExported SMILESPE merge rules to {os.path.basename(merges_path)}; first 5 lines:")
        with open(merges_path) as f:
            for _ in range(5):
                print(f"  {f.readline().rstrip()}")


def main():
    # Train tokenizer
    tokenizer = train_example()

    # Analyze compression
    compression_analysis(tokenizer)

    # Persist it
    save_and_reload(tokenizer)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
