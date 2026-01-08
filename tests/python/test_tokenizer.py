"""Tests for rustmolbpe SMILES tokenizer."""

import pytest
import tempfile
import os


@pytest.fixture
def tokenizer():
    """Create a fresh tokenizer instance."""
    import rustmolbpe
    return rustmolbpe.SmilesTokenizer()


@pytest.fixture
def trained_tokenizer(tokenizer):
    """Create a tokenizer trained on sample SMILES."""
    smiles = [
        "CCO",  # ethanol
        "CCC",  # propane
        "CCCC",  # butane
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    ] * 100
    tokenizer.train_from_iterator(iter(smiles), vocab_size=100)
    return tokenizer


class TestAtomwiseTokenize:
    """Tests for atomwise_tokenize function."""

    def test_simple_smiles(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("CCO")
        assert tokens == ["C", "C", "O"]

    def test_halogens(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("CBr")
        assert tokens == ["C", "Br"]

        tokens = rustmolbpe.atomwise_tokenize("CCl")
        assert tokens == ["C", "Cl"]

    def test_bracket_atoms(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("[C@@H](O)C")
        assert tokens == ["[C@@H]", "(", "O", ")", "C"]

    def test_aromatic(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("c1ccccc1")
        assert tokens == ["c", "1", "c", "c", "c", "c", "c", "1"]

    def test_bonds(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("C=C#N")
        assert tokens == ["C", "=", "C", "#", "N"]

    def test_charged_atoms(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("[N+](C)(C)C")
        assert tokens == ["[N+]", "(", "C", ")", "(", "C", ")", "C"]

    def test_stereochemistry(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("C/C=C/C")
        assert tokens == ["C", "/", "C", "=", "C", "/", "C"]

    def test_two_digit_ring(self):
        import rustmolbpe
        tokens = rustmolbpe.atomwise_tokenize("C%12CC%12")
        assert tokens == ["C", "%12", "C", "C", "%12"]


class TestSmilesTokenizer:
    """Tests for SmilesTokenizer class."""

    def test_new_tokenizer(self, tokenizer):
        # New tokenizer has 4 special tokens (PAD, UNK, BOS, EOS)
        assert tokenizer.vocab_size == 4
        assert tokenizer.num_merges == 0
        # Verify special tokens
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.bos_token == "<bos>"
        assert tokenizer.eos_token == "<eos>"

    def test_train_from_iterator(self, tokenizer):
        smiles = ["CCO", "CCC", "CCCC"] * 10
        tokenizer.train_from_iterator(iter(smiles), vocab_size=50)

        assert tokenizer.vocab_size > 0
        assert tokenizer.base_vocab_size > 0

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        smiles = "CCO"
        ids = trained_tokenizer.encode(smiles)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == smiles

    def test_encode_decode_complex(self, trained_tokenizer):
        smiles = "c1ccccc1"
        ids = trained_tokenizer.encode(smiles)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == smiles

    def test_batch_encode(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC", "c1ccccc1"]
        individual = [trained_tokenizer.encode(s) for s in smiles_list]
        batched = trained_tokenizer.batch_encode(smiles_list)
        assert individual == batched

    def test_batch_decode(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC"]
        ids_list = [trained_tokenizer.encode(s) for s in smiles_list]
        decoded = trained_tokenizer.batch_decode(ids_list)
        assert decoded == smiles_list

    def test_get_vocabulary(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocabulary()
        assert len(vocab) == trained_tokenizer.vocab_size
        assert all(isinstance(token, str) and isinstance(id_, int) for token, id_ in vocab)

    def test_id_to_token(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocabulary()
        for token, id_ in vocab:
            assert trained_tokenizer.id_to_token(id_) == token

    def test_token_to_id(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocabulary()
        for token, id_ in vocab:
            assert trained_tokenizer.token_to_id(token) == id_


class TestVocabularyIO:
    """Tests for vocabulary save/load."""

    def test_save_load_roundtrip(self, trained_tokenizer):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            vocab_path = f.name

        try:
            # Save vocabulary
            trained_tokenizer.save_vocabulary(vocab_path)

            # Load into new tokenizer
            import rustmolbpe
            new_tokenizer = rustmolbpe.SmilesTokenizer()
            new_tokenizer.load_vocabulary(vocab_path)

            # Compare encoding results
            test_smiles = ["CCO", "CCC", "c1ccccc1"]
            for smi in test_smiles:
                original_ids = trained_tokenizer.encode(smi)
                loaded_ids = new_tokenizer.encode(smi)
                # Note: IDs may differ but decoded result should be the same
                original_decoded = trained_tokenizer.decode(original_ids)
                loaded_decoded = new_tokenizer.decode(loaded_ids)
                assert original_decoded == loaded_decoded == smi
        finally:
            os.unlink(vocab_path)

    def test_load_nonexistent_file(self, tokenizer):
        with pytest.raises(Exception):
            tokenizer.load_vocabulary("/nonexistent/path/vocab.txt")

    def test_vocab_file_format(self, trained_tokenizer):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            vocab_path = f.name

        try:
            trained_tokenizer.save_vocabulary(vocab_path)

            # Check file format
            with open(vocab_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(' ', 1)
                assert len(parts) == 2, f"Invalid line: {line}"
        finally:
            os.unlink(vocab_path)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_smiles(self, trained_tokenizer):
        ids = trained_tokenizer.encode("")
        assert ids == []
        decoded = trained_tokenizer.decode([])
        assert decoded == ""

    def test_unknown_token_id(self, trained_tokenizer):
        with pytest.raises(Exception):
            trained_tokenizer.decode([999999])

    def test_unknown_token_string(self, trained_tokenizer):
        with pytest.raises(Exception):
            trained_tokenizer.token_to_id("UNKNOWN_TOKEN_XYZ")

    def test_single_atom_smiles(self, trained_tokenizer):
        ids = trained_tokenizer.encode("C")
        decoded = trained_tokenizer.decode(ids)
        assert decoded == "C"


class TestTrainingOptions:
    """Tests for training options."""

    def test_vocab_size_parameter(self, tokenizer):
        smiles = ["CCO", "CCC", "CCCC"] * 100
        tokenizer.train_from_iterator(iter(smiles), vocab_size=20)
        assert tokenizer.vocab_size <= 20

    def test_min_frequency_parameter(self, tokenizer):
        # With high min_frequency, should have fewer merges
        smiles = ["CCO"] + ["CCC"] * 100
        tokenizer.train_from_iterator(iter(smiles), vocab_size=100, min_frequency=50)
        # "CCO" appears only once, so its patterns shouldn't be merged


@pytest.mark.slow
class TestLargeScale:
    """Tests for larger-scale operations."""

    def test_batch_encode_large(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC", "c1ccccc1"] * 1000
        results = trained_tokenizer.batch_encode(smiles_list)
        assert len(results) == 3000

    def test_training_determinism(self):
        """Test that training is deterministic."""
        import rustmolbpe

        smiles = ["CCO", "CCC", "CCCC", "c1ccccc1"] * 50

        tok1 = rustmolbpe.SmilesTokenizer()
        tok1.train_from_iterator(iter(smiles), vocab_size=30)

        tok2 = rustmolbpe.SmilesTokenizer()
        tok2.train_from_iterator(iter(smiles), vocab_size=30)

        # Both tokenizers should encode the same way
        test_smiles = "CCCO"
        assert tok1.encode(test_smiles) == tok2.encode(test_smiles)


class TestSpecialTokens:
    """Tests for special tokens functionality."""

    def test_special_tokens_in_new_tokenizer(self):
        import rustmolbpe
        tok = rustmolbpe.SmilesTokenizer()

        # Check special token IDs
        assert tok.pad_token_id == 0
        assert tok.unk_token_id == 1
        assert tok.bos_token_id == 2
        assert tok.eos_token_id == 3

        # Check special token strings
        assert tok.id_to_token(0) == "<pad>"
        assert tok.id_to_token(1) == "<unk>"
        assert tok.id_to_token(2) == "<bos>"
        assert tok.id_to_token(3) == "<eos>"

    def test_special_tokens_after_training(self, trained_tokenizer):
        # Special tokens should still be at IDs 0-3 after training
        assert trained_tokenizer.pad_token_id == 0
        assert trained_tokenizer.unk_token_id == 1
        assert trained_tokenizer.bos_token_id == 2
        assert trained_tokenizer.eos_token_id == 3
        assert trained_tokenizer.id_to_token(0) == "<pad>"

    def test_encode_with_special_tokens(self, trained_tokenizer):
        smiles = "CCO"
        ids_no_special = trained_tokenizer.encode(smiles)
        ids_with_special = trained_tokenizer.encode(smiles, add_special_tokens=True)

        # With special tokens should have BOS at start and EOS at end
        assert ids_with_special[0] == trained_tokenizer.bos_token_id
        assert ids_with_special[-1] == trained_tokenizer.eos_token_id
        assert ids_with_special[1:-1] == ids_no_special

    def test_batch_encode_with_special_tokens(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC"]
        batch_no_special = trained_tokenizer.batch_encode(smiles_list)
        batch_with_special = trained_tokenizer.batch_encode(smiles_list, add_special_tokens=True)

        for no_special, with_special in zip(batch_no_special, batch_with_special):
            assert with_special[0] == trained_tokenizer.bos_token_id
            assert with_special[-1] == trained_tokenizer.eos_token_id
            assert with_special[1:-1] == no_special

    def test_unknown_token_handling(self):
        import rustmolbpe
        # Train on limited vocabulary
        tok = rustmolbpe.SmilesTokenizer()
        tok.train_from_iterator(iter(["CCO", "CCC"] * 10), vocab_size=10)

        # Encode something with unknown atoms
        ids = tok.encode("CBr")  # Br likely not in small vocab
        # Should contain UNK token for unknown atoms
        assert tok.unk_token_id in ids

    def test_decode_special_tokens(self, trained_tokenizer):
        # Encoding with special tokens and decoding should include special token strings
        smiles = "CCO"
        ids = trained_tokenizer.encode(smiles, add_special_tokens=True)
        decoded = trained_tokenizer.decode(ids)
        assert decoded.startswith("<bos>")
        assert decoded.endswith("<eos>")
        assert smiles in decoded


class TestPadding:
    """Tests for padding functionality."""

    def test_pad_right_default(self, trained_tokenizer):
        sequences = [[1, 2], [3, 4, 5], [6]]
        result = trained_tokenizer.pad(sequences)

        assert result["input_ids"] == [[1, 2, 0], [3, 4, 5], [6, 0, 0]]
        assert result["attention_mask"] == [[1, 1, 0], [1, 1, 1], [1, 0, 0]]

    def test_pad_left(self, trained_tokenizer):
        sequences = [[1, 2], [3, 4, 5], [6]]
        result = trained_tokenizer.pad(sequences, padding="left")

        assert result["input_ids"] == [[0, 1, 2], [3, 4, 5], [0, 0, 6]]
        assert result["attention_mask"] == [[0, 1, 1], [1, 1, 1], [0, 0, 1]]

    def test_pad_with_max_length(self, trained_tokenizer):
        sequences = [[1, 2], [3]]
        result = trained_tokenizer.pad(sequences, max_length=5)

        assert result["input_ids"] == [[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]]
        assert result["attention_mask"] == [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]

    def test_pad_with_truncation(self, trained_tokenizer):
        sequences = [[1, 2, 3, 4, 5], [6, 7]]
        result = trained_tokenizer.pad(sequences, max_length=3, truncation=True)

        assert result["input_ids"] == [[1, 2, 3], [6, 7, 0]]
        assert result["attention_mask"] == [[1, 1, 1], [1, 1, 0]]

    def test_pad_without_attention_mask(self, trained_tokenizer):
        sequences = [[1, 2], [3, 4, 5]]
        result = trained_tokenizer.pad(sequences, return_attention_mask=False)

        assert result["input_ids"] == [[1, 2, 0], [3, 4, 5]]
        assert "attention_mask" not in result

    def test_pad_empty_sequences(self, trained_tokenizer):
        sequences = [[], [1, 2]]
        result = trained_tokenizer.pad(sequences)

        assert result["input_ids"] == [[0, 0], [1, 2]]
        assert result["attention_mask"] == [[0, 0], [1, 1]]

    def test_encode_batch_padded(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC", "CCCC"]
        result = trained_tokenizer.encode_batch_padded(smiles_list)

        # All sequences should have same length
        lengths = [len(seq) for seq in result["input_ids"]]
        assert len(set(lengths)) == 1

        # Attention mask should match padding
        for ids, mask in zip(result["input_ids"], result["attention_mask"]):
            for i, (token_id, attention) in enumerate(zip(ids, mask)):
                if token_id == trained_tokenizer.pad_token_id:
                    assert attention == 0
                else:
                    assert attention == 1

    def test_encode_batch_padded_with_special_tokens(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC"]
        result = trained_tokenizer.encode_batch_padded(
            smiles_list, add_special_tokens=True
        )

        # Each sequence should start with BOS
        for ids in result["input_ids"]:
            assert ids[0] == trained_tokenizer.bos_token_id

        # Non-padded positions should end with EOS
        for ids, mask in zip(result["input_ids"], result["attention_mask"]):
            # Find last non-padded position
            last_real = sum(mask) - 1
            assert ids[last_real] == trained_tokenizer.eos_token_id

    def test_encode_batch_padded_with_max_length(self, trained_tokenizer):
        smiles_list = ["CCO", "CCC"]
        max_len = 10
        result = trained_tokenizer.encode_batch_padded(
            smiles_list, max_length=max_len
        )

        for ids in result["input_ids"]:
            assert len(ids) == max_len
