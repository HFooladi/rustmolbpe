#!/usr/bin/env python3
"""Compute token-count statistics for the four rustmolbpe tokenizers.

Runs every tokenizer over the ChEMBL 36 reference dataset and reports, per
tokenizer, the distribution of the number of tokens per molecule (mean, median,
std, min, max and percentiles).

Tokenizers compared:
    - CharTokenizer    : character-level split (no merges)
    - AtomTokenizer    : atom-level regex split (no merges)
    - CharBPETokenizer : BPE on characters, swept over several vocab sizes
    - SmilesTokenizer  : BPE on atoms ("SPE"), swept over several vocab sizes

The BPE tokenizers are trained once per vocab size and their vocabularies are
cached under ``data/`` so subsequent runs are fast. The character-level and
atom-level tokenizers need no training: with no merges, the token count of a
molecule is simply its number of characters / atoms.

Requires the optional ``stats`` dependencies: ``pip install rustmolbpe[stats]``
(numpy and matplotlib).

Examples:
    python tokenizer_stats.py                       # full ChEMBL 36
    python tokenizer_stats.py --limit 100000        # quick check on a sample
    python tokenizer_stats.py --vocab-sizes 1000,4096
"""

import argparse
import logging
import os
import time

import numpy as np

import rustmolbpe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Percentiles reported for every tokenizer.
PERCENTILES = [25, 75, 90, 95, 99]

# Chunk size for the streaming counting pass.
CHUNK_SIZE = 50_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Token-count statistics for the rustmolbpe tokenizers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-file",
        default="data/chembl_36_chemreps.txt",
        help="Path to the ChEMBL 36 chemreps file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N molecules (default: the whole dataset).",
    )
    parser.add_argument(
        "--vocab-sizes",
        default="1000,2000,4096,8000",
        help="Comma-separated BPE vocabulary sizes to sweep.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tokenizer_stats.csv",
        help="Where to write the statistics CSV.",
    )
    parser.add_argument(
        "--output-plot",
        default="results/token_count_hist.png",
        help="Where to write the token-count histogram figure.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum merge frequency when training BPE tokenizers.",
    )
    return parser.parse_args()


def smiles_generator(filepath, limit=None):
    """Yield canonical SMILES from a ChEMBL chemreps file (tab-separated).

    The canonical SMILES is the second column; the header line is skipped.
    """
    with open(filepath, "r") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                yield parts[1]


def chunked(iterable, size):
    """Yield lists of at most `size` items from `iterable`."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_bpe_tokenizer(cls, cache_path, data_file, limit, vocab_size, min_frequency):
    """Load a cached BPE vocabulary, or train it and cache it.

    Caching is only used for full-dataset runs (``limit is None``); a ``--limit``
    run always trains fresh so a partial vocabulary is never mistaken for a
    full-dataset one.
    """
    use_cache = limit is None
    tok = cls()
    if use_cache and os.path.exists(cache_path):
        logger.info("Loading cached vocabulary %s", cache_path)
        tok.load_vocabulary(cache_path)
        return tok

    logger.info(
        "Training %s (vocab_size=%d) on %s ...",
        cls.__name__,
        vocab_size,
        data_file,
    )
    start = time.time()
    tok.train_from_iterator(
        smiles_generator(data_file, limit),
        vocab_size=vocab_size,
        buffer_size=16384,
        min_frequency=min_frequency,
    )
    logger.info(
        "  trained in %.1fs (vocab_size=%d, num_merges=%d)",
        time.time() - start,
        tok.vocab_size,
        tok.num_merges,
    )
    if use_cache:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tok.save_vocabulary(cache_path)
    return tok


def build_tokenizers(data_file, limit, vocab_sizes, min_frequency):
    """Build every tokenizer configuration to evaluate.

    Returns a list of (label, tokenizer, vocab_size) tuples. `vocab_size` is the
    effective vocabulary size; for the no-merge tokenizers it is reported as-is.
    """
    configs = []

    # No-merge tokenizers: token count is independent of any trained vocabulary,
    # so a fresh instance encodes to the correct length already.
    configs.append(("CharTokenizer", rustmolbpe.CharTokenizer(), None))
    configs.append(("AtomTokenizer", rustmolbpe.AtomTokenizer(), None))

    # BPE tokenizers: train (or load) one per vocab size.
    for size in vocab_sizes:
        char_bpe = build_bpe_tokenizer(
            rustmolbpe.CharBPETokenizer,
            f"data/chembl_charbpe_{size}_vocab.txt",
            data_file,
            limit,
            size,
            min_frequency,
        )
        configs.append(("CharBPETokenizer", char_bpe, size))

    for size in vocab_sizes:
        spe = build_bpe_tokenizer(
            rustmolbpe.SmilesTokenizer,
            f"data/chembl_spe_{size}_vocab.txt",
            data_file,
            limit,
            size,
            min_frequency,
        )
        configs.append(("SmilesTokenizer", spe, size))

    return configs


def count_tokens(configs, data_file, limit):
    """Stream the dataset once and collect per-molecule token counts.

    Returns a dict mapping each config index to a numpy array of token counts.
    """
    counts = {i: [] for i in range(len(configs))}
    n_molecules = 0
    start = time.time()

    for chunk in chunked(smiles_generator(data_file, limit), CHUNK_SIZE):
        n_molecules += len(chunk)
        for i, (_label, tok, _size) in enumerate(configs):
            encoded = tok.batch_encode(chunk)
            counts[i].extend(len(ids) for ids in encoded)
        logger.info("  counted %d molecules ...", n_molecules)

    logger.info(
        "Counted %d molecules across %d tokenizers in %.1fs",
        n_molecules,
        len(configs),
        time.time() - start,
    )
    return {i: np.asarray(v, dtype=np.int64) for i, v in counts.items()}


def compute_stats(label, vocab_size, token_counts):
    """Compute the statistics row for one tokenizer."""
    row = {
        "tokenizer": label,
        "vocab_size": "-" if vocab_size is None else vocab_size,
        "n_molecules": int(token_counts.size),
        "mean": float(np.mean(token_counts)),
        "median": float(np.median(token_counts)),
        "std": float(np.std(token_counts)),
        "min": int(np.min(token_counts)),
        "max": int(np.max(token_counts)),
        "total_tokens": int(np.sum(token_counts)),
    }
    for p, value in zip(PERCENTILES, np.percentile(token_counts, PERCENTILES)):
        row[f"p{p}"] = float(value)
    return row


def print_table(rows):
    """Print the statistics as a formatted console table."""
    columns = [
        ("tokenizer", "Tokenizer", 18, "s"),
        ("vocab_size", "Vocab", 7, "s"),
        ("n_molecules", "N", 10, "d"),
        ("mean", "Mean", 9, ".2f"),
        ("median", "Median", 8, ".1f"),
        ("std", "Std", 8, ".2f"),
        ("min", "Min", 6, "d"),
        ("max", "Max", 7, "d"),
        ("p25", "P25", 8, ".1f"),
        ("p75", "P75", 8, ".1f"),
        ("p90", "P90", 8, ".1f"),
        ("p95", "P95", 8, ".1f"),
        ("p99", "P99", 8, ".1f"),
    ]
    header = " ".join(f"{title:>{width}}" for _, title, width, _ in columns)
    print("\n" + "=" * len(header))
    print("Token-count statistics per tokenizer (ChEMBL 36)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for key, _title, width, fmt in columns:
            value = str(row[key]) if fmt == "s" else row[key]
            cells.append(f"{value:>{width}{fmt}}")
        print(" ".join(cells))
    print("=" * len(header) + "\n")


def write_csv(rows, path):
    """Write the statistics rows to a CSV file."""
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "tokenizer",
        "vocab_size",
        "n_molecules",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "total_tokens",
    ] + [f"p{p}" for p in PERCENTILES]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote statistics CSV to %s", path)


def plot_histograms(configs, counts, rows, path):
    """Plot overlaid token-count distributions for every tokenizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Clip the x-axis to the largest P99 so the histograms stay readable.
    x_max = max(row["p99"] for row in rows)
    bins = np.linspace(0, x_max, 80)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (label, _tok, size) in enumerate(configs):
        legend = label if size is None else f"{label} (vocab={size})"
        ax.hist(
            counts[i],
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.5,
            label=legend,
        )
    ax.set_xlabel("Tokens per molecule")
    ax.set_ylabel("Density")
    ax.set_title("Token-count distributions on ChEMBL 36")
    ax.legend(fontsize=8)
    ax.set_xlim(0, x_max)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote histogram figure to %s", path)


def main():
    args = parse_args()
    vocab_sizes = [int(s) for s in args.vocab_sizes.split(",") if s.strip()]

    if not os.path.exists(args.data_file):
        raise SystemExit(f"ChEMBL data file not found: {args.data_file}")

    logger.info("=" * 60)
    logger.info("rustmolbpe tokenizer statistics on ChEMBL 36")
    logger.info("Data file : %s", args.data_file)
    logger.info("Limit     : %s", args.limit if args.limit else "full dataset")
    logger.info("BPE sizes : %s", vocab_sizes)
    logger.info("=" * 60)

    configs = build_tokenizers(
        args.data_file, args.limit, vocab_sizes, args.min_frequency
    )
    counts = count_tokens(configs, args.data_file, args.limit)

    rows = [
        compute_stats(label, size, counts[i])
        for i, (label, _tok, size) in enumerate(configs)
    ]

    print_table(rows)
    write_csv(rows, args.output_csv)
    plot_histograms(configs, counts, rows, args.output_plot)

    logger.info("Done.")


if __name__ == "__main__":
    main()
