#!/usr/bin/env python3
"""Comprehensive benchmark: rustmolbpe vs SMILESPE on ChEMBL and PubChem."""

import sys
import time
import gzip
import io
sys.path.insert(0, '/data/local/hfooladi/LLM/tokenizer/SmilesPE')

import rustmolbpe
from SmilesPE.tokenizer import SPE_Tokenizer
from SmilesPE.learner import learn_SPE


def load_chembl_smiles(filepath, limit=None):
    """Load SMILES from ChEMBL file."""
    smiles = []
    with open(filepath, 'r') as f:
        next(f)  # Skip header
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                smiles.append(parts[1])
    return smiles


def load_pubchem_smiles(filepath, limit=None):
    """Load SMILES from PubChem CID-SMILES file."""
    smiles = []
    open_func = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'
    with open_func(filepath, mode) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[1]:
                smiles.append(parts[1])
    return smiles


def benchmark_encoding_speed(tokenizer, smiles_list, is_rust=True):
    """Benchmark encoding speed, return (time, speed, avg_tokens)."""
    start = time.perf_counter()
    if is_rust:
        results = tokenizer.batch_encode(smiles_list)
    else:
        results = [tokenizer.tokenize(s).split() for s in smiles_list]
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(r) for r in results)
    avg_tokens = total_tokens / len(smiles_list)
    speed = len(smiles_list) / elapsed

    return elapsed, speed, avg_tokens


def main():
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: rustmolbpe vs SMILESPE")
    print("=" * 80)

    # File paths
    chembl_file = "data/chembl_36_chemreps.txt"
    pubchem_file = "data/pubchem_smiles.gz"
    smilespe_vocab = "/data/local/hfooladi/LLM/tokenizer/SmilesPE/SPE_ChEMBL.txt"
    chembl_vocab = "data/chembl36_vocab.txt"
    pubchem_vocab = "data/pubchem_10M_vocab.txt"

    # Load test datasets
    print("\nLoading test datasets...")
    chembl_10k = load_chembl_smiles(chembl_file, limit=10000)
    chembl_100k = load_chembl_smiles(chembl_file, limit=100000)
    pubchem_10k = load_pubchem_smiles(pubchem_file, limit=10000)
    pubchem_100k = load_pubchem_smiles(pubchem_file, limit=100000)

    print(f"  ChEMBL: {len(chembl_10k):,} / {len(chembl_100k):,} SMILES")
    print(f"  PubChem: {len(pubchem_10k):,} / {len(pubchem_100k):,} SMILES")

    # Load tokenizers
    print("\nLoading tokenizers...")

    # SMILESPE (original)
    with open(smilespe_vocab, 'r') as f:
        spe_tokenizer = SPE_Tokenizer(f)
    print(f"  SMILESPE: loaded (3002 merges)")

    # rustmolbpe ChEMBL
    rust_chembl = rustmolbpe.SmilesTokenizer()
    rust_chembl.load_vocabulary(chembl_vocab)
    print(f"  rustmolbpe (ChEMBL): {rust_chembl.num_merges} merges")

    # rustmolbpe PubChem
    rust_pubchem = rustmolbpe.SmilesTokenizer()
    rust_pubchem.load_vocabulary(pubchem_vocab)
    print(f"  rustmolbpe (PubChem): {rust_pubchem.num_merges} merges")

    # =========================================================================
    # ENCODING SPEED BENCHMARK
    # =========================================================================
    print("\n" + "=" * 80)
    print("ENCODING SPEED BENCHMARK (100k SMILES)")
    print("=" * 80)

    results = {}

    # Test on ChEMBL data
    print("\n--- ChEMBL Test Set (100k drug-like molecules) ---")
    print(f"{'Tokenizer':<25} {'Time (s)':<12} {'Speed (/s)':<15} {'Avg Tokens':<12}")
    print("-" * 65)

    # SMILESPE on ChEMBL
    t, s, avg = benchmark_encoding_speed(spe_tokenizer, chembl_100k, is_rust=False)
    print(f"{'SMILESPE':<25} {t:<12.2f} {s:<15,.0f} {avg:<12.1f}")
    results['spe_chembl'] = (t, s, avg)

    # rustmolbpe (ChEMBL vocab) on ChEMBL
    t, s, avg = benchmark_encoding_speed(rust_chembl, chembl_100k)
    print(f"{'rustmolbpe (ChEMBL)':<25} {t:<12.2f} {s:<15,.0f} {avg:<12.1f}")
    results['rust_chembl_chembl'] = (t, s, avg)

    # rustmolbpe (PubChem vocab) on ChEMBL
    t, s, avg = benchmark_encoding_speed(rust_pubchem, chembl_100k)
    print(f"{'rustmolbpe (PubChem)':<25} {t:<12.2f} {s:<15,.0f} {avg:<12.1f}")
    results['rust_pubchem_chembl'] = (t, s, avg)

    # Test on PubChem data
    print("\n--- PubChem Test Set (100k diverse molecules) ---")
    print(f"{'Tokenizer':<25} {'Time (s)':<12} {'Speed (/s)':<15} {'Avg Tokens':<12}")
    print("-" * 65)

    # Need fresh SMILESPE (cache)
    with open(smilespe_vocab, 'r') as f:
        spe_tokenizer2 = SPE_Tokenizer(f)

    # SMILESPE on PubChem
    t, s, avg = benchmark_encoding_speed(spe_tokenizer2, pubchem_100k, is_rust=False)
    print(f"{'SMILESPE':<25} {t:<12.2f} {s:<15,.0f} {avg:<12.1f}")
    results['spe_pubchem'] = (t, s, avg)

    # rustmolbpe (ChEMBL vocab) on PubChem
    t, s, avg = benchmark_encoding_speed(rust_chembl, pubchem_100k)
    print(f"{'rustmolbpe (ChEMBL)':<25} {t:<12.2f} {s:<15,.0f} {avg:<12.1f}")
    results['rust_chembl_pubchem'] = (t, s, avg)

    # rustmolbpe (PubChem vocab) on PubChem
    t, s, avg = benchmark_encoding_speed(rust_pubchem, pubchem_100k)
    print(f"{'rustmolbpe (PubChem)':<25} {t:<12.2f} {s:<15,.0f} {avg:<12.1f}")
    results['rust_pubchem_pubchem'] = (t, s, avg)

    # =========================================================================
    # TRAINING SPEED BENCHMARK
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING SPEED BENCHMARK")
    print("=" * 80)

    train_results = {}

    for dataset_name, loader, filepath in [
        ("ChEMBL", load_chembl_smiles, chembl_file),
        ("PubChem", load_pubchem_smiles, pubchem_file),
    ]:
        print(f"\n--- Training on {dataset_name} (50k SMILES, vocab_size=1000) ---")
        train_smiles = loader(filepath, limit=50000)

        # SMILESPE training
        outfile = io.StringIO()
        start = time.perf_counter()
        learn_SPE(train_smiles, outfile, num_symbols=1000, min_frequency=2, verbose=False)
        spe_time = time.perf_counter() - start

        # rustmolbpe training
        rust_trainer = rustmolbpe.SmilesTokenizer()
        start = time.perf_counter()
        rust_trainer.train_from_iterator(iter(train_smiles), vocab_size=1000, min_frequency=1)
        rust_time = time.perf_counter() - start

        speedup = spe_time / rust_time
        print(f"  SMILESPE:   {spe_time:.2f}s")
        print(f"  rustmolbpe: {rust_time:.2f}s")
        print(f"  Speedup:    {speedup:.1f}x")

        train_results[dataset_name] = (spe_time, rust_time, speedup)

    # =========================================================================
    # COMPRESSION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPRESSION ANALYSIS (characters per token)")
    print("=" * 80)

    def calc_compression(tokenizer, smiles_list, is_rust=True):
        total_chars = sum(len(s) for s in smiles_list)
        if is_rust:
            total_tokens = sum(len(tokenizer.encode(s)) for s in smiles_list)
        else:
            total_tokens = sum(len(tokenizer.tokenize(s).split()) for s in smiles_list)
        return total_chars / total_tokens

    print(f"\n{'Tokenizer':<25} {'ChEMBL':<15} {'PubChem':<15}")
    print("-" * 55)

    # Sample for compression (use 10k for speed)
    with open(smilespe_vocab, 'r') as f:
        spe_fresh = SPE_Tokenizer(f)

    spe_chembl_comp = calc_compression(spe_fresh, chembl_10k, is_rust=False)
    with open(smilespe_vocab, 'r') as f:
        spe_fresh2 = SPE_Tokenizer(f)
    spe_pubchem_comp = calc_compression(spe_fresh2, pubchem_10k, is_rust=False)
    print(f"{'SMILESPE':<25} {spe_chembl_comp:<15.2f} {spe_pubchem_comp:<15.2f}")

    rust_chembl_chembl_comp = calc_compression(rust_chembl, chembl_10k)
    rust_chembl_pubchem_comp = calc_compression(rust_chembl, pubchem_10k)
    print(f"{'rustmolbpe (ChEMBL)':<25} {rust_chembl_chembl_comp:<15.2f} {rust_chembl_pubchem_comp:<15.2f}")

    rust_pubchem_chembl_comp = calc_compression(rust_pubchem, chembl_10k)
    rust_pubchem_pubchem_comp = calc_compression(rust_pubchem, pubchem_10k)
    print(f"{'rustmolbpe (PubChem)':<25} {rust_pubchem_chembl_comp:<15.2f} {rust_pubchem_pubchem_comp:<15.2f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n### Speed Comparison (rustmolbpe vs SMILESPE)")
    print(f"  Encoding (ChEMBL):  {results['rust_chembl_chembl'][1]/results['spe_chembl'][1]:.0f}x faster")
    print(f"  Encoding (PubChem): {results['rust_pubchem_pubchem'][1]/results['spe_pubchem'][1]:.0f}x faster")
    print(f"  Training (ChEMBL):  {train_results['ChEMBL'][2]:.0f}x faster")
    print(f"  Training (PubChem): {train_results['PubChem'][2]:.0f}x faster")

    print("\n### Throughput")
    print(f"  rustmolbpe batch encoding: {max(results['rust_chembl_chembl'][1], results['rust_pubchem_pubchem'][1]):,.0f} SMILES/sec")
    print(f"  SMILESPE encoding:         {max(results['spe_chembl'][1], results['spe_pubchem'][1]):,.0f} SMILES/sec")

    print("\n### Compression (chars/token, higher = better)")
    print(f"  Best on ChEMBL:  rustmolbpe (ChEMBL vocab) = {rust_chembl_chembl_comp:.2f}")
    print(f"  Best on PubChem: rustmolbpe (PubChem vocab) = {rust_pubchem_pubchem_comp:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
