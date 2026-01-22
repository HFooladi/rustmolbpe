#!/usr/bin/env python3
"""Train rustmolbpe tokenizer on ChEMBL 36 SMILES data."""

import argparse
import time
import logging
import rustmolbpe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train rustmolbpe tokenizer on ChEMBL 36 SMILES data."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Target vocabulary size (default: 4096)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum frequency for merges (default: 1)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/chembl_36_chemreps.txt",
        help="Path to ChEMBL chemreps file (default: data/chembl_36_chemreps.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chembl36_vocab.txt",
        help="Output vocabulary file (default: data/chembl36_vocab.txt)"
    )
    return parser.parse_args()


def smiles_generator(filepath):
    """Generate SMILES strings from ChEMBL chemreps file."""
    with open(filepath, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:  # canonical_smiles is second column
                yield parts[1]


def main():
    args = parse_args()

    data_file = args.data_file
    vocab_file = args.output
    vocab_size = args.vocab_size
    min_frequency = args.min_frequency

    logger.info("=" * 60)
    logger.info("Training rustmolbpe tokenizer on ChEMBL 36")
    logger.info("=" * 60)
    logger.info(f"Data file: {data_file}")
    logger.info(f"Target vocab size: {vocab_size}")
    logger.info(f"Min frequency: {min_frequency}")
    logger.info(f"Output file: {vocab_file}")

    # Count total SMILES first
    logger.info("Counting SMILES...")
    total_smiles = sum(1 for _ in smiles_generator(data_file))
    logger.info(f"Total SMILES: {total_smiles:,}")

    # Create tokenizer
    tokenizer = rustmolbpe.SmilesTokenizer()

    # Train
    logger.info("Starting training...")
    start_time = time.time()

    tokenizer.train_from_iterator(
        smiles_generator(data_file),
        vocab_size=vocab_size,
        buffer_size=16384,
        min_frequency=min_frequency
    )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")

    # Report statistics
    logger.info("-" * 40)
    logger.info("Tokenizer Statistics:")
    logger.info(f"  Total vocab size: {tokenizer.vocab_size}")
    logger.info(f"  Base vocab size: {tokenizer.base_vocab_size}")
    logger.info(f"  Number of merges: {tokenizer.num_merges}")

    # Save vocabulary
    logger.info(f"Saving vocabulary to {vocab_file}...")
    tokenizer.save_vocabulary(vocab_file)

    # Test encoding/decoding
    logger.info("-" * 40)
    logger.info("Testing tokenizer:")
    test_smiles = [
        "CCO",                          # ethanol
        "c1ccccc1",                      # benzene
        "CC(=O)O",                       # acetic acid
        "CC(=O)Nc1ccc(O)cc1",            # paracetamol
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",   # ibuprofen
    ]

    for smi in test_smiles:
        ids = tokenizer.encode(smi)
        decoded = tokenizer.decode(ids)
        logger.info(f"  {smi}")
        logger.info(f"    -> tokens: {ids}")
        logger.info(f"    -> decoded: {decoded} (match: {decoded == smi})")

    # Benchmark encoding speed
    logger.info("-" * 40)
    logger.info("Benchmarking encoding speed...")

    # Collect sample SMILES
    sample_smiles = []
    for i, smi in enumerate(smiles_generator(data_file)):
        sample_smiles.append(smi)
        if i >= 9999:
            break

    start_time = time.time()
    _ = tokenizer.batch_encode(sample_smiles)
    batch_time = time.time() - start_time

    logger.info(f"  Encoded 10,000 SMILES in {batch_time:.3f}s")
    logger.info(f"  Speed: {10000/batch_time:,.0f} SMILES/second")

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
