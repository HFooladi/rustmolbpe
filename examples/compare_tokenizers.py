#!/usr/bin/env python3
"""Compare the five rustmolbpe tokenizers on the same molecules.

rustmolbpe provides a ladder of tokenizers with an identical API. They differ in
how a SMILES string is split into base units and whether BPE merges are learned:

    CharTokenizer     characters, no merges
    AtomTokenizer     atoms (Br, Cl, [C@@H] kept whole), no merges
    CharBPETokenizer  BPE merges on characters
    SmilesTokenizer   BPE merges on atoms ("SPE"); also exported as AtomBPETokenizer
    ByteBPETokenizer  BPE merges on UTF-8 bytes; never emits <unk>

This script trains all five on the same small corpus and, on molecules that
are not in that corpus, shows:
- the size of each vocabulary
- how many tokens each one needs per molecule, and what the tokens look like
- how each handles an element it never saw during training

Runs anywhere: the training data is included in the script.
"""

import rustmolbpe

TOKENIZER_CLASSES = [
    rustmolbpe.CharTokenizer,
    rustmolbpe.AtomTokenizer,
    rustmolbpe.CharBPETokenizer,
    rustmolbpe.SmilesTokenizer,
    rustmolbpe.ByteBPETokenizer,
]

# A small, diverse corpus: each molecule appears once, so BPE only learns
# fragments shared by several molecules (min_frequency defaults to 2).
TRAINING_SMILES = [
    "CCCO", "CC(C)O", "CC(=O)O", "CC(=O)OC", "CCOC(=O)C", "CC(=O)N", "CNC(=O)C",
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "OC(=O)c1ccccc1", "COc1ccccc1",
    "CC(=O)c1ccccc1", "Fc1ccccc1", "Brc1ccccc1", "ClCCl", "BrCCBr", "ClC(Cl)Cl",
    "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "c1cnc2ccccc2n1", "c1ccoc1", "c1ccsc1",
    "CC(C)Cc1ccccc1", "CC(C)C(=O)O", "OC(=O)CCC(=O)O", "NCC(=O)O", "C[C@@H](N)C(=O)O",
    "C[C@H](O)CC", "CN(C)C", "CCN(CC)CC", "O=C1CCCCC1", "C1CCNCC1", "C1COCCN1",
    "[O-][N+](=O)c1ccccc1", "C[N+](C)(C)C", "Cn1ccnc1", "O=c1cc[nH]c(=O)[nH]1",
    "CC(=O)Oc1ccccc1", "COC(=O)c1ccccc1O", "Nc1ccc(O)cc1", "Oc1ccc(Cl)cc1",
]

# None of these appear in the training corpus.
TEST_MOLECULES = [
    ("Ethanol", "CCO"),
    ("Chlorobenzene", "Clc1ccccc1"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
]

# Cisplatin: platinum ([Pt]) never appears in the training corpus.
UNSEEN_SMILES = "N[Pt](N)(Cl)Cl"


def train_all():
    """Train one tokenizer of each class on the same corpus."""
    tokenizers = {}
    for cls in TOKENIZER_CLASSES:
        tok = cls()
        # vocab_size is ignored by the no-merge tokenizers; for the BPE ones it
        # includes the base vocabulary (always 260 tokens for ByteBPETokenizer).
        tok.train_from_iterator(iter(TRAINING_SMILES), vocab_size=400)
        tokenizers[cls.__name__] = tok
    return tokenizers


def show_tokens(tokenizer, smiles):
    """The token strings for a SMILES string."""
    return [tokenizer.id_to_token(i) for i in tokenizer.encode(smiles)]


def main():
    print("=" * 72)
    print("Comparing the rustmolbpe tokenizers")
    print("=" * 72)

    tokenizers = train_all()

    print(f"\n{'Tokenizer':<18} {'vocab_size':>10} {'base':>6} {'merges':>7}")
    print("-" * 44)
    for name, tok in tokenizers.items():
        print(f"{name:<18} {tok.vocab_size:>10} {tok.base_vocab_size:>6} {tok.num_merges:>7}")

    print("\n" + "-" * 72)
    print("Tokens per molecule")
    print("-" * 72)
    short_names = [name.replace("Tokenizer", "") for name in tokenizers]
    print(f"\n{'Molecule':<15}" + "".join(f"{name:>12}" for name in short_names))
    for label, smiles in TEST_MOLECULES:
        counts = "".join(f"{len(tok.encode(smiles)):>12}" for tok in tokenizers.values())
        print(f"{label:<15}{counts}")

    smiles = "Clc1ccccc1"
    print(f"\nChlorobenzene ({smiles}) without merges:")
    for name in ("CharTokenizer", "AtomTokenizer"):
        print(f"  {name:<18} {show_tokens(tokenizers[name], smiles)}")
    print("  The character-level tokenizer splits chlorine 'Cl' into 'C' + 'l';")
    print("  the atom-level tokenizer keeps it as one unit.")

    smiles = "CC(=O)Nc1ccc(O)cc1"
    print(f"\nParacetamol ({smiles}) with learned merges:")
    for name in ("CharBPETokenizer", "SmilesTokenizer", "ByteBPETokenizer"):
        print(f"  {name:<18} {show_tokens(tokenizers[name], smiles)}")
    print("  BPE reuses fragments learned from other molecules, such as the acetyl")
    print("  group 'CC(=O)' and pieces of the benzene ring.")

    print("\n" + "-" * 72)
    print(f"An element never seen in training: {UNSEEN_SMILES} (cisplatin)")
    print("-" * 72)
    unk_counts = {}
    for name, tok in tokenizers.items():
        ids = tok.encode(UNSEEN_SMILES)
        unk_counts[name] = ids.count(tok.unk_token_id)
        print(
            f"  {name:<18} <unk> tokens: {unk_counts[name]}"
            f"  | decodes back to input: {tok.decode(ids) == UNSEEN_SMILES}"
        )
    print(f"\nByteBPETokenizer never emits <unk>: {unk_counts['ByteBPETokenizer'] == 0}")

    print("\n" + "-" * 72)
    print("Alias")
    print("-" * 72)
    print(
        "AtomBPETokenizer is SmilesTokenizer: "
        f"{rustmolbpe.AtomBPETokenizer is rustmolbpe.SmilesTokenizer}"
    )


if __name__ == "__main__":
    main()
