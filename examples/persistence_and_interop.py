#!/usr/bin/env python3
"""Saving, sharing and exporting rustmolbpe tokenizers.

This script demonstrates:
- save() / from_file(): a lossless JSON file that restores identical token IDs
  (the way to keep a tokenizer together with a model trained on its IDs)
- pickle, and passing a tokenizer to multiprocessing workers
- exporting to the HuggingFace `tokenizers` tokenizer.json format
  (CharTokenizer, AtomTokenizer and CharBPETokenizer)

Runs anywhere: the training data is included in the script. Loading the
HuggingFace export back is shown when the optional `tokenizers` package is
installed (pip install "rustmolbpe[hf]").
"""

import json
import multiprocessing
import os
import pickle
import tempfile

import rustmolbpe

TRAINING_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "c1ccc2[nH]ccc2c1",
    "ClCCl",
]

MOLECULES = [
    "CC(=O)Nc1ccc(O)cc1",
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Clc1ccccc1",
] * 250


def encode_chunk(args):
    """Worker function: encode one chunk of SMILES.

    Defined at module level so multiprocessing can find it in worker processes.
    The tokenizer itself arrives pickled along with the chunk.
    """
    tokenizer, chunk = args
    return tokenizer.batch_encode(chunk)


def section(title):
    print("\n" + "-" * 64)
    print(title)
    print("-" * 64)


def main():
    print("=" * 64)
    print("Saving, sharing and exporting tokenizers")
    print("=" * 64)

    tokenizer = rustmolbpe.SmilesTokenizer()
    tokenizer.train_from_iterator(iter(TRAINING_SMILES * 50), vocab_size=200)
    print(f"\nTrained a SmilesTokenizer: {tokenizer.vocab_size} tokens, {tokenizer.num_merges} merges")

    with tempfile.TemporaryDirectory() as tmp:
        # ---------------------------------------------------------------
        section("1. save() / from_file()")
        path = os.path.join(tmp, "smiles_tokenizer.json")
        tokenizer.save(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Wrote {os.path.basename(path)} ({os.path.getsize(path):,} bytes):")
        print(f"  format={data['format']!r}, version={data['version']}, pretokenizer={data['pretokenizer']!r}")
        print(f"  vocab: {len(data['vocab'])} tokens (list index = token ID), e.g. {data['vocab'][:6]}")
        print(f"  merges: {len(data['merges'])} [left_id, right_id, merged_id] triples, e.g. {data['merges'][:2]}")

        restored = rustmolbpe.SmilesTokenizer.from_file(path)
        identical = (
            restored.get_vocabulary() == tokenizer.get_vocabulary()
            and restored.get_merges() == tokenizer.get_merges()
            and restored.batch_encode(MOLECULES) == tokenizer.batch_encode(MOLECULES)
        )
        print(f"Token IDs identical after save()/from_file(): {identical}")

        # ---------------------------------------------------------------
        section("2. pickle")
        payload = pickle.dumps(tokenizer)
        unpickled = pickle.loads(payload)
        same = unpickled.batch_encode(MOLECULES) == tokenizer.batch_encode(MOLECULES)
        print(f"Pickled size: {len(payload):,} bytes")
        print(f"Pickle round trip identical: {same}")
        print("Use pickle for passing a tokenizer between processes; for files on disk,")
        print("prefer save()/from_file(): readable JSON and safe to load from an untrusted source.")

        # ---------------------------------------------------------------
        section("3. multiprocessing")
        chunks = [MOLECULES[i : i + 250] for i in range(0, len(MOLECULES), 250)]
        # "spawn" starts clean worker processes and behaves the same on every OS.
        with multiprocessing.get_context("spawn").Pool(2) as pool:
            results = pool.map(encode_chunk, [(tokenizer, chunk) for chunk in chunks])
        parallel = [ids for chunk_ids in results for ids in chunk_ids]
        print(f"Encoded {len(MOLECULES):,} SMILES in {len(chunks)} chunks across 2 worker processes")
        print(f"Multiprocessing results identical: {parallel == tokenizer.batch_encode(MOLECULES)}")
        print("(Within one process, batch_encode() already runs in parallel.)")

        # ---------------------------------------------------------------
        section("4. HuggingFace tokenizer.json export")
        char_bpe = rustmolbpe.CharBPETokenizer()
        char_bpe.train_from_iterator(iter(TRAINING_SMILES * 50), vocab_size=200)
        hf_path = os.path.join(tmp, "tokenizer.json")
        char_bpe.save_huggingface(hf_path)
        print(f"Wrote HuggingFace tokenizer.json from a CharBPETokenizer ({os.path.getsize(hf_path):,} bytes)")

        try:
            from tokenizers import Tokenizer
        except ImportError:
            print('Install the optional `tokenizers` package (pip install "rustmolbpe[hf]") to load it back.')
        else:
            hf = Tokenizer.from_file(hf_path)
            smiles = MOLECULES[0]
            print(f"\n{smiles}:")
            print(f"  rustmolbpe: {[char_bpe.id_to_token(i) for i in char_bpe.encode(smiles)]}")
            print(f"  tokenizers: {hf.encode(smiles).tokens}")
            print("  (HuggingFace applies merges in priority order while rustmolbpe uses")
            print("   greedy longest-match, so sequences can occasionally differ.)")

        try:
            tokenizer.save_huggingface(os.path.join(tmp, "spe.json"))
        except NotImplementedError:
            print("\nSmilesTokenizer (atom-level BPE) has no HuggingFace equivalent; use save()/from_file().")


if __name__ == "__main__":
    main()
