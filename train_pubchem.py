#!/usr/bin/env python3
"""Train rustmolbpe tokenizer on PubChem SMILES data."""

import gzip
import time
import logging
import argparse
import rustmolbpe

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def smiles_generator(filepath, limit=None):
    """Generate SMILES strings from PubChem CID-SMILES file.

    Format: CID<tab>SMILES (one per line)
    """
    count = 0
    open_func = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'

    with open_func(filepath, mode) as f:
        for line in f:
            if limit and count >= limit:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                yield parts[1]
                count += 1


def main():
    parser = argparse.ArgumentParser(description='Train tokenizer on PubChem data')
    parser.add_argument('--limit', type=int, default=10_000_000,
                        help='Number of SMILES to use for training (default: 10M)')
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='Target vocabulary size (default: 8000)')
    parser.add_argument('--min-frequency', type=int, default=100,
                        help='Minimum frequency for BPE merges (default: 100)')
    parser.add_argument('--buffer-size', type=int, default=32768,
                        help='Buffer size for streaming (default: 32768)')
    args = parser.parse_args()

    data_file = "data/pubchem_smiles.gz"
    vocab_file = f"data/pubchem_{args.limit//1_000_000}M_vocab.txt"

    logger.info("=" * 60)
    logger.info("Training rustmolbpe tokenizer on PubChem")
    logger.info("=" * 60)
    logger.info(f"Data file: {data_file}")
    logger.info(f"Training limit: {args.limit:,} SMILES")
    logger.info(f"Target vocab size: {args.vocab_size}")
    logger.info(f"Min frequency: {args.min_frequency}")

    # Create tokenizer
    tokenizer = rustmolbpe.SmilesTokenizer()

    # Train
    logger.info("Starting training...")
    start_time = time.time()

    tokenizer.train_from_iterator(
        smiles_generator(data_file, limit=args.limit),
        vocab_size=args.vocab_size,
        buffer_size=args.buffer_size,
        min_frequency=args.min_frequency
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
        "CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",  # acetylcarnitine (PubChem #1)
    ]

    for smi in test_smiles:
        ids = tokenizer.encode(smi)
        decoded = tokenizer.decode(ids)
        logger.info(f"  {smi}")
        logger.info(f"    -> {len(ids)} tokens, decoded match: {decoded == smi}")

    # Benchmark encoding speed on sample
    logger.info("-" * 40)
    logger.info("Benchmarking encoding speed...")

    sample_smiles = list(smiles_generator(data_file, limit=100000))

    start_time = time.time()
    _ = tokenizer.batch_encode(sample_smiles)
    batch_time = time.time() - start_time

    logger.info(f"  Encoded 100,000 SMILES in {batch_time:.3f}s")
    logger.info(f"  Speed: {100000/batch_time:,.0f} SMILES/second")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"Vocabulary saved to: {vocab_file}")


if __name__ == "__main__":
    main()
