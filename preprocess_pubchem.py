#!/usr/bin/env python3
"""Preprocess PubChem SMILES to canonical (aromatic) form.

This script reads the raw PubChem CID-SMILES file (which contains kekulized SMILES)
and converts each SMILES to its canonical form using RDKit. The output is a gzipped
file with one canonical SMILES per line (no CID).

Usage:
    python preprocess_pubchem.py --limit 10000000
"""

import gzip
import time
import logging
import argparse
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form using RDKit.

    Returns None if the SMILES cannot be parsed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def process_batch(batch):
    """Process a batch of SMILES strings."""
    results = []
    for smiles in batch:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            results.append(canonical)
    return results


def read_raw_smiles(filepath, limit=None):
    """Read raw SMILES from PubChem CID-SMILES file."""
    open_func = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'

    count = 0
    with open_func(filepath, mode) as f:
        for line in f:
            if limit and count >= limit:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                yield parts[1]
                count += 1


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess PubChem SMILES to canonical form'
    )
    parser.add_argument('--input', type=str, default='data/pubchem_smiles.gz',
                        help='Input PubChem CID-SMILES file (default: data/pubchem_smiles.gz)')
    parser.add_argument('--output', type=str, default='data/pubchem_canonical_smiles.gz',
                        help='Output canonical SMILES file (default: data/pubchem_canonical_smiles.gz)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of SMILES to process (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Batch size for parallel processing (default: 10000)')
    args = parser.parse_args()

    num_workers = args.workers or cpu_count()

    logger.info("=" * 60)
    logger.info("Preprocessing PubChem SMILES to canonical form")
    logger.info("=" * 60)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Limit: {args.limit if args.limit else 'None (process all)'}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Batch size: {args.batch_size}")

    start_time = time.time()
    success = 0
    total_read = 0

    # Read all SMILES into memory (batched processing)
    logger.info("Reading input file...")
    all_smiles = list(read_raw_smiles(args.input, args.limit))
    total_read = len(all_smiles)
    logger.info(f"Read {total_read:,} SMILES")

    # Create batches
    batches = [all_smiles[i:i + args.batch_size]
               for i in range(0, len(all_smiles), args.batch_size)]
    logger.info(f"Created {len(batches)} batches")

    # Process in parallel
    logger.info("Processing with multiprocessing...")
    with gzip.open(args.output, 'wt') as outfile:
        with Pool(num_workers) as pool:
            for i, results in enumerate(pool.imap(process_batch, batches)):
                for canonical in results:
                    outfile.write(canonical + '\n')
                    success += 1

                # Progress logging every 100 batches
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = success / elapsed
                    logger.info(f"Processed {success:,} SMILES ({rate:,.0f} SMILES/sec)")

    elapsed = time.time() - start_time
    failed = total_read - success

    logger.info("-" * 40)
    logger.info("Preprocessing complete!")
    logger.info(f"Total SMILES read: {total_read:,}")
    logger.info(f"Successfully canonicalized: {success:,}")
    logger.info(f"Failed: {failed:,}")
    logger.info(f"Time elapsed: {elapsed:.2f} seconds")
    logger.info(f"Average rate: {success/elapsed:,.0f} SMILES/sec")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
