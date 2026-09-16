#!/usr/bin/env python3
"""Batch processing example for rustmolbpe.

This script demonstrates:
- Batch encoding for efficiency
- Padding sequences for ML models
- Creating attention masks
- Different padding strategies
- The HuggingFace-style call interface

Usage:
    python examples/batch_processing.py [VOCAB_PATH]

VOCAB_PATH defaults to the repository's data/chembl36_vocab.txt.
"""

import sys
import time
from pathlib import Path

import rustmolbpe

VOCAB_URL = "https://github.com/HFooladi/rustmolbpe/blob/main/data/chembl36_vocab.txt"

# Size of the batch used for the speed comparison: large enough that the timing
# reflects encoding work rather than per-call noise.
BENCHMARK_SIZE = 200_000


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
    print("=" * 60)
    print("Batch Processing with rustmolbpe")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_pretrained(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nLoaded vocabulary with {tokenizer.vocab_size} tokens")

    # Sample batch of molecules
    smiles_batch = [
        "CCO",  # ethanol (short)
        "c1ccccc1",  # benzene
        "CC(=O)Nc1ccc(O)cc1",  # paracetamol
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen (longer)
    ]

    # =========================================================================
    # Basic batch encoding
    # =========================================================================
    print("\n" + "-" * 60)
    print("Basic Batch Encoding")
    print("-" * 60)

    ids_batch = tokenizer.batch_encode(smiles_batch)

    print("\nEncoded sequences (variable length):")
    for smiles, ids in zip(smiles_batch, ids_batch):
        print(f"  {smiles:<35} -> {ids}")

    # =========================================================================
    # Padded batch encoding (for ML models)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Padded Batch Encoding (for ML)")
    print("-" * 60)

    # Encode and pad in one step
    result = tokenizer.encode_batch_padded(
        smiles_batch,
        add_special_tokens=True,  # Add BOS/EOS
        return_attention_mask=True,
    )

    print("\nWith special tokens (BOS/EOS) and auto-padding:")
    print(f"  Shape: {len(result['input_ids'])} x {len(result['input_ids'][0])}")

    for i, (ids, mask) in enumerate(zip(result["input_ids"], result["attention_mask"])):
        real_len = sum(mask)
        print(f"  [{i}] len={real_len}: {ids}")

    # =========================================================================
    # Fixed length with truncation
    # =========================================================================
    print("\n" + "-" * 60)
    print("Fixed Length with Truncation")
    print("-" * 60)

    max_length = 5
    result = tokenizer.encode_batch_padded(
        smiles_batch,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )

    print(f"\nFixed to {max_length} tokens (with truncation):")
    for smiles, ids, mask in zip(
        smiles_batch, result["input_ids"], result["attention_mask"]
    ):
        print(f"  {smiles[:20]:<20} -> {ids} mask={mask}")
    print("  Note: truncation cuts the end of a long sequence, including its EOS token.")

    # =========================================================================
    # Left padding (for autoregressive models)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Left Padding (for autoregressive generation)")
    print("-" * 60)

    result_left = tokenizer.encode_batch_padded(
        smiles_batch[:3],
        padding="left",
        add_special_tokens=True,
    )

    result_right = tokenizer.encode_batch_padded(
        smiles_batch[:3],
        padding="right",
        add_special_tokens=True,
    )

    print("\nComparison (left vs right padding):")
    for i in range(3):
        print(f"  Right: {result_right['input_ids'][i]}")
        print(f"  Left:  {result_left['input_ids'][i]}")
        print()

    # =========================================================================
    # Manual padding (pad already-encoded sequences)
    # =========================================================================
    print("-" * 60)
    print("Manual Padding of Pre-encoded Sequences")
    print("-" * 60)

    # Encode first
    sequences = tokenizer.batch_encode(smiles_batch[:3])
    print("\nOriginal sequences:")
    for seq in sequences:
        print(f"  {seq}")

    # Pad separately
    padded = tokenizer.pad(sequences, max_length=10, return_attention_mask=True)
    print("\nPadded to length 10:")
    for ids, mask in zip(padded["input_ids"], padded["attention_mask"]):
        print(f"  IDs:  {ids}")
        print(f"  Mask: {mask}")
        print()

    # =========================================================================
    # HuggingFace-style call interface
    # =========================================================================
    print("-" * 60)
    print("Call interface (tokenizer(...), like HuggingFace tokenizers)")
    print("-" * 60)

    single = tokenizer("CCO", add_special_tokens=True)
    print(f"\ntokenizer('CCO', add_special_tokens=True):\n  {single}")

    batch = tokenizer(smiles_batch[:3], padding=True, add_special_tokens=True)
    print("\ntokenizer([...], padding=True, add_special_tokens=True):")
    print(f"  input_ids:      {batch['input_ids']}")
    print(f"  attention_mask: {batch['attention_mask']}")

    # =========================================================================
    # Performance comparison
    # =========================================================================
    print("\n" + "-" * 60)
    print("Performance: one-by-one vs parallel batch encoding")
    print("-" * 60)

    large_batch = (smiles_batch * (BENCHMARK_SIZE // len(smiles_batch) + 1))[:BENCHMARK_SIZE]

    # One call per molecule
    start = time.perf_counter()
    for smiles in large_batch:
        tokenizer.encode(smiles)
    seq_time = time.perf_counter() - start

    # One call for the whole batch (parallelized across CPU cores)
    start = time.perf_counter()
    tokenizer.batch_encode(large_batch)
    batch_time = time.perf_counter() - start

    print(f"\nEncoding {len(large_batch):,} SMILES:")
    print(f"  encode() in a loop: {seq_time:.3f}s ({len(large_batch) / seq_time:,.0f} SMILES/sec)")
    print(f"  batch_encode():     {batch_time:.3f}s ({len(large_batch) / batch_time:,.0f} SMILES/sec)")
    print(f"  Speedup: {seq_time / batch_time:.1f}x (varies with CPU cores and molecule length)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
