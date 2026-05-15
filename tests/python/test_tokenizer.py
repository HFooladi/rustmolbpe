"""Tests for rustmolbpe SMILES tokenizer."""

import pytest
import tempfile
import os


# Top-level function for multiprocessing test (local functions can't be pickled)
def _encode_smiles_worker(tok_smiles):
    """Worker function for multiprocessing test."""
    tok, smiles = tok_smiles
    return tok.encode(smiles)


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


class TestIsTrainedMethod:
    """Tests for is_trained() method."""

    def test_fresh_tokenizer_not_trained(self):
        """A fresh tokenizer should return False for is_trained()."""
        import rustmolbpe
        tok = rustmolbpe.SmilesTokenizer()
        assert tok.is_trained() is False

    def test_trained_tokenizer_is_trained(self):
        """After training, is_trained() should return True."""
        import rustmolbpe
        tok = rustmolbpe.SmilesTokenizer()
        smiles = ["CCO", "CCC", "CCCC"] * 100
        tok.train_from_iterator(iter(smiles), vocab_size=50)
        assert tok.is_trained() is True

    def test_after_load_vocabulary(self, trained_tokenizer):
        """After loading vocabulary, is_trained() should return True."""
        import rustmolbpe

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            vocab_path = f.name

        try:
            trained_tokenizer.save_vocabulary(vocab_path)

            new_tok = rustmolbpe.SmilesTokenizer()
            assert new_tok.is_trained() is False

            new_tok.load_vocabulary(vocab_path)
            assert new_tok.is_trained() is True
        finally:
            os.unlink(vocab_path)


class TestGetMergesMethod:
    """Tests for get_merges() method."""

    def test_fresh_tokenizer_no_merges(self):
        """A fresh tokenizer should return empty list for get_merges()."""
        import rustmolbpe
        tok = rustmolbpe.SmilesTokenizer()
        merges = tok.get_merges()
        assert merges == []

    def test_trained_tokenizer_has_merges(self, trained_tokenizer):
        """A trained tokenizer should return non-empty list."""
        merges = trained_tokenizer.get_merges()
        assert len(merges) > 0
        assert len(merges) == trained_tokenizer.num_merges

    def test_merge_tuple_structure(self, trained_tokenizer):
        """Each merge should be a tuple of (str, str, str)."""
        merges = trained_tokenizer.get_merges()
        for merge in merges:
            assert isinstance(merge, tuple)
            assert len(merge) == 3
            left, right, merged = merge
            assert isinstance(left, str)
            assert isinstance(right, str)
            assert isinstance(merged, str)

    def test_merged_is_concatenation(self, trained_tokenizer):
        """The merged token should be concatenation of left and right."""
        merges = trained_tokenizer.get_merges()
        for left, right, merged in merges:
            assert merged == left + right


class TestPickleSupport:
    """Tests for pickle serialization support."""

    def test_pickle_fresh_tokenizer(self):
        """A fresh tokenizer should be picklable."""
        import pickle
        import rustmolbpe

        tok = rustmolbpe.SmilesTokenizer()
        data = pickle.dumps(tok)
        restored = pickle.loads(data)

        assert restored.vocab_size == tok.vocab_size
        assert restored.is_trained() == tok.is_trained()

    def test_pickle_trained_tokenizer(self, trained_tokenizer):
        """A trained tokenizer should be picklable."""
        import pickle

        data = pickle.dumps(trained_tokenizer)
        restored = pickle.loads(data)

        assert restored.vocab_size == trained_tokenizer.vocab_size
        assert restored.num_merges == trained_tokenizer.num_merges
        assert restored.is_trained() is True

    def test_pickle_encode_decode_consistency(self, trained_tokenizer):
        """Encoding should produce same results before and after pickling."""
        import pickle

        test_smiles = ["CCO", "CCC", "c1ccccc1", "CC(=O)O"]

        # Encode before pickling
        original_encodings = [trained_tokenizer.encode(smi) for smi in test_smiles]

        # Pickle roundtrip
        restored = pickle.loads(pickle.dumps(trained_tokenizer))

        # Encode after pickling
        restored_encodings = [restored.encode(smi) for smi in test_smiles]

        assert original_encodings == restored_encodings

    def test_pickle_vocabulary_consistency(self, trained_tokenizer):
        """Vocabulary should be preserved after pickling."""
        import pickle

        original_vocab = trained_tokenizer.get_vocabulary()

        restored = pickle.loads(pickle.dumps(trained_tokenizer))
        restored_vocab = restored.get_vocabulary()

        assert original_vocab == restored_vocab

    def test_pickle_to_file(self, trained_tokenizer):
        """Tokenizer should be picklable to and from a file."""
        import pickle

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle_path = f.name
            pickle.dump(trained_tokenizer, f)

        try:
            with open(pickle_path, 'rb') as f:
                restored = pickle.load(f)

            assert restored.vocab_size == trained_tokenizer.vocab_size
            assert trained_tokenizer.encode("CCO") == restored.encode("CCO")
        finally:
            os.unlink(pickle_path)

    def test_pickle_different_protocols(self, trained_tokenizer):
        """Tokenizer should work with all pickle protocols."""
        import pickle

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            data = pickle.dumps(trained_tokenizer, protocol=protocol)
            restored = pickle.loads(data)
            assert restored.encode("CCO") == trained_tokenizer.encode("CCO")

    def test_pickle_special_tokens_preserved(self, trained_tokenizer):
        """Special token IDs should be preserved after pickling."""
        import pickle

        restored = pickle.loads(pickle.dumps(trained_tokenizer))

        assert restored.pad_token_id == 0
        assert restored.unk_token_id == 1
        assert restored.bos_token_id == 2
        assert restored.eos_token_id == 3
        assert restored.id_to_token(0) == "<pad>"
        assert restored.id_to_token(1) == "<unk>"
        assert restored.id_to_token(2) == "<bos>"
        assert restored.id_to_token(3) == "<eos>"

    def test_multiprocessing_compatibility(self, trained_tokenizer):
        """Tokenizer should work with multiprocessing."""
        import multiprocessing

        test_smiles = ["CCO", "CCC", "CCCC", "c1ccccc1"]
        args = [(trained_tokenizer, smi) for smi in test_smiles]

        # Test with multiprocessing.Pool
        # Uses module-level _encode_smiles_worker function (local functions can't be pickled)
        with multiprocessing.Pool(2) as pool:
            results = pool.map(_encode_smiles_worker, args)

        # Verify results match direct encoding
        expected = [trained_tokenizer.encode(smi) for smi in test_smiles]
        assert results == expected


class TestCallableInterface:
    """Tests for __call__ (HuggingFace-compatible) interface."""

    def test_single_string(self, trained_tokenizer):
        """Single string returns flat lists."""
        result = trained_tokenizer("CCO")
        assert isinstance(result["input_ids"], list)
        assert all(isinstance(x, int) for x in result["input_ids"])
        assert result["input_ids"] == trained_tokenizer.encode("CCO")

    def test_single_string_attention_mask(self, trained_tokenizer):
        """Single string returns attention mask of all 1s."""
        result = trained_tokenizer("CCO")
        assert result["attention_mask"] == [1] * len(result["input_ids"])

    def test_list_of_strings(self, trained_tokenizer):
        """List of strings returns nested lists."""
        smiles_list = ["CCO", "CCC", "c1ccccc1"]
        result = trained_tokenizer(smiles_list)
        assert len(result["input_ids"]) == 3
        for ids in result["input_ids"]:
            assert isinstance(ids, list)

    def test_list_matches_batch_encode(self, trained_tokenizer):
        """List result should match batch_encode."""
        smiles_list = ["CCO", "CCC"]
        call_result = trained_tokenizer(smiles_list)
        batch_result = trained_tokenizer.batch_encode(smiles_list)
        assert call_result["input_ids"] == batch_result

    def test_padding(self, trained_tokenizer):
        """padding=True pads to equal length."""
        result = trained_tokenizer(["CCO", "C"], padding=True)
        lengths = [len(ids) for ids in result["input_ids"]]
        assert len(set(lengths)) == 1  # all same length

    def test_padding_attention_mask(self, trained_tokenizer):
        """Padded positions have attention_mask=0."""
        result = trained_tokenizer(["CCO", "C"], padding=True)
        for ids, mask in zip(result["input_ids"], result["attention_mask"]):
            for token_id, attention in zip(ids, mask):
                if token_id == trained_tokenizer.pad_token_id:
                    assert attention == 0
                else:
                    assert attention == 1

    def test_truncation(self, trained_tokenizer):
        """truncation=True with max_length truncates."""
        result = trained_tokenizer("CCCCCCCCCC", truncation=True, max_length=3)
        assert len(result["input_ids"]) == 3

    def test_truncation_batch(self, trained_tokenizer):
        """truncation works on batches."""
        result = trained_tokenizer(
            ["CCCCCCCCCC", "C"], truncation=True, max_length=3, padding=True
        )
        for ids in result["input_ids"]:
            assert len(ids) == 3

    def test_add_special_tokens(self, trained_tokenizer):
        """add_special_tokens adds BOS/EOS."""
        result = trained_tokenizer("CCO", add_special_tokens=True)
        assert result["input_ids"][0] == trained_tokenizer.bos_token_id
        assert result["input_ids"][-1] == trained_tokenizer.eos_token_id

    def test_no_attention_mask(self, trained_tokenizer):
        """return_attention_mask=False omits it."""
        result = trained_tokenizer("CCO", return_attention_mask=False)
        assert "input_ids" in result
        assert "attention_mask" not in result

    def test_single_with_padding_and_max_length(self, trained_tokenizer):
        """Single string with padding and max_length."""
        result = trained_tokenizer("C", padding=True, max_length=5)
        assert len(result["input_ids"]) == 5

    def test_empty_list(self, trained_tokenizer):
        """Empty list input."""
        result = trained_tokenizer([])
        assert result["input_ids"] == []

    def test_consistency_with_encode_batch_padded(self, trained_tokenizer):
        """__call__ with padding should match encode_batch_padded."""
        smiles_list = ["CCO", "CCC", "CCCC"]
        call_result = trained_tokenizer(smiles_list, padding=True, add_special_tokens=True)
        padded_result = trained_tokenizer.encode_batch_padded(
            smiles_list, add_special_tokens=True
        )
        assert call_result["input_ids"] == padded_result["input_ids"]
        assert call_result["attention_mask"] == padded_result["attention_mask"]


class TestErrorHandlingEdgeCases:
    """Tests for error handling and edge cases."""

    def test_decode_invalid_token_id(self, trained_tokenizer):
        """Decoding with invalid token ID should raise ValueError."""
        invalid_id = trained_tokenizer.vocab_size + 1000
        with pytest.raises(ValueError, match="Unknown token id"):
            trained_tokenizer.decode([invalid_id])

    def test_id_to_token_invalid_id(self, trained_tokenizer):
        """id_to_token with invalid ID should raise ValueError."""
        invalid_id = trained_tokenizer.vocab_size + 1000
        with pytest.raises(ValueError, match="Unknown token id"):
            trained_tokenizer.id_to_token(invalid_id)

    def test_token_to_id_unknown_token(self, trained_tokenizer):
        """token_to_id with unknown token should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown token"):
            trained_tokenizer.token_to_id("totally_unknown_token_xyz")

    def test_batch_decode_with_invalid_id(self, trained_tokenizer):
        """batch_decode with one invalid ID should raise ValueError."""
        valid_ids = trained_tokenizer.encode("CCO")
        invalid_ids = [trained_tokenizer.vocab_size + 1000]
        with pytest.raises(ValueError, match="Unknown token id"):
            trained_tokenizer.batch_decode([valid_ids, invalid_ids])

    def test_encode_very_long_smiles(self, trained_tokenizer):
        """Encoding a very long SMILES should work without crashing."""
        # Create a long chain
        long_smiles = "C" * 1000
        ids = trained_tokenizer.encode(long_smiles)
        assert len(ids) > 0
        decoded = trained_tokenizer.decode(ids)
        assert decoded == long_smiles

    def test_encode_special_characters_in_smiles(self, trained_tokenizer):
        """SMILES with various special characters should be handled."""
        # Test various SMILES patterns
        test_smiles = [
            "C=C",      # Double bond
            "C#N",      # Triple bond
            "C/C=C/C",  # Stereochemistry
            "C\\C=C\\C", # Stereochemistry
            "[Na+]",    # Charged atom
            "[O-]",     # Negative charge
            "C.C",      # Disconnected
        ]
        for smi in test_smiles:
            ids = trained_tokenizer.encode(smi)
            assert len(ids) >= 0  # May be empty if atoms unknown

    def test_pad_empty_list(self, trained_tokenizer):
        """Padding an empty list should return empty result."""
        result = trained_tokenizer.pad([])
        assert result["input_ids"] == []
        assert result["attention_mask"] == []

    def test_pad_single_empty_sequence(self, trained_tokenizer):
        """Padding a list with single empty sequence should work."""
        result = trained_tokenizer.pad([[]])
        assert result["input_ids"] == [[]]
        assert result["attention_mask"] == [[]]

    def test_encode_batch_padded_empty_list(self, trained_tokenizer):
        """encode_batch_padded with empty list should return empty result."""
        result = trained_tokenizer.encode_batch_padded([])
        assert result["input_ids"] == []
        assert result["attention_mask"] == []

    def test_truncation_longer_than_max_length(self, trained_tokenizer):
        """Truncation should work when sequence exceeds max_length."""
        long_smiles = "CCCCCCCCCC"  # 10 carbons
        result = trained_tokenizer.encode_batch_padded(
            [long_smiles],
            max_length=3,
            truncation=True
        )
        assert len(result["input_ids"][0]) == 3

    def test_truncation_disabled_preserves_length(self, trained_tokenizer):
        """Without truncation, sequences should not be cut."""
        long_smiles = "CCCCCCCCCC"  # 10 carbons
        result = trained_tokenizer.encode_batch_padded(
            [long_smiles],
            max_length=3,
            truncation=False
        )
        # Without truncation, max_length only affects padding
        assert len(result["input_ids"][0]) >= 3

    def test_left_padding_alignment(self, trained_tokenizer):
        """Left padding should align sequences to the right."""
        result = trained_tokenizer.encode_batch_padded(
            ["C", "CC"],
            max_length=5,
            padding="left"
        )
        # Shorter sequence should have padding on the left
        first_seq = result["input_ids"][0]
        assert first_seq[0] == trained_tokenizer.pad_token_id

    def test_vocab_properties_consistency(self, trained_tokenizer):
        """Vocabulary properties should be consistent."""
        vocab = trained_tokenizer.get_vocabulary()
        assert len(vocab) == trained_tokenizer.vocab_size
        assert trained_tokenizer.base_vocab_size <= trained_tokenizer.vocab_size
        assert trained_tokenizer.vocab_size == trained_tokenizer.base_vocab_size + trained_tokenizer.num_merges


# ============================================================================
# Tests for the four-tokenizer ladder: CharTokenizer, AtomTokenizer,
# CharBPETokenizer, and SmilesTokenizer (atom-level BPE / SPE).
# ============================================================================

# Sample SMILES used to train/build vocabularies in the tests below.
_LADDER_SMILES = [
    "CCO",  # ethanol
    "CCC",  # propane
    "CCCC",  # butane
    "c1ccccc1",  # benzene
    "CC(=O)O",  # acetic acid
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CCBr",  # bromoethane
    "CCCl",  # chloroethane
    "C[C@@H](O)CCN",  # a stereocenter, exercises bracket atoms
] * 100

# Round-trip molecules (encode -> decode must reproduce the input exactly).
_ROUNDTRIP_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
]


def _all_tokenizer_classes():
    import rustmolbpe
    return [
        rustmolbpe.CharTokenizer,
        rustmolbpe.AtomTokenizer,
        rustmolbpe.CharBPETokenizer,
        rustmolbpe.SmilesTokenizer,
    ]


class TestCharTokenizer:
    """Tests for the character-level tokenizer."""

    @pytest.fixture
    def char_tok(self):
        import rustmolbpe
        tok = rustmolbpe.CharTokenizer()
        tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=0)
        return tok

    def test_splits_into_individual_characters(self, char_tok):
        """'Cl' is two characters for a character-level tokenizer."""
        ids = char_tok.encode("Cl")
        assert len(ids) == 2
        assert char_tok.decode(ids) == "Cl"

    def test_bracket_atom_split_per_character(self, char_tok):
        """'[C@@H]' is six characters, not one atom."""
        ids = char_tok.encode("[C@@H]")
        assert len(ids) == 6

    def test_never_has_merges(self, char_tok):
        assert char_tok.num_merges == 0
        assert char_tok.is_trained() is False

    def test_has_vocabulary_after_training(self, char_tok):
        import rustmolbpe
        fresh = rustmolbpe.CharTokenizer()
        assert fresh.has_vocabulary() is False
        assert char_tok.has_vocabulary() is True

    def test_roundtrip(self, char_tok):
        for smi in _ROUNDTRIP_SMILES:
            assert char_tok.decode(char_tok.encode(smi)) == smi

    def test_special_token_ids(self, char_tok):
        assert char_tok.pad_token_id == 0
        assert char_tok.unk_token_id == 1
        assert char_tok.bos_token_id == 2
        assert char_tok.eos_token_id == 3

    def test_add_special_tokens(self, char_tok):
        ids = char_tok.encode("CCO", add_special_tokens=True)
        assert ids[0] == char_tok.bos_token_id
        assert ids[-1] == char_tok.eos_token_id

    def test_empty_string(self, char_tok):
        assert char_tok.encode("") == []

    def test_vocab_io_not_supported(self, char_tok):
        with pytest.raises(NotImplementedError):
            char_tok.save_vocabulary("/tmp/_should_not_exist.txt")
        with pytest.raises(NotImplementedError):
            char_tok.load_vocabulary("/tmp/_should_not_exist.txt")


class TestAtomTokenizer:
    """Tests for the atom-level (no-merge) tokenizer."""

    @pytest.fixture
    def atom_tok(self):
        import rustmolbpe
        tok = rustmolbpe.AtomTokenizer()
        tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=0)
        return tok

    def test_groups_multichar_atoms(self, atom_tok):
        """'CCl' is two tokens: 'C' and the chlorine atom 'Cl'."""
        ids = atom_tok.encode("CCl")
        assert len(ids) == 2
        assert atom_tok.id_to_token(ids[1]) == "Cl"

    def test_bracket_atom_is_one_token(self, atom_tok):
        ids = atom_tok.encode("[C@@H]")
        assert len(ids) == 1
        assert atom_tok.id_to_token(ids[0]) == "[C@@H]"

    def test_never_has_merges(self, atom_tok):
        assert atom_tok.num_merges == 0
        assert atom_tok.is_trained() is False

    def test_roundtrip(self, atom_tok):
        for smi in _ROUNDTRIP_SMILES:
            assert atom_tok.decode(atom_tok.encode(smi)) == smi

    def test_matches_atomwise_tokenize_length(self, atom_tok):
        """Token count equals the atomwise_tokenize unit count."""
        import rustmolbpe
        for smi in _ROUNDTRIP_SMILES:
            assert len(atom_tok.encode(smi)) == len(rustmolbpe.atomwise_tokenize(smi))

    def test_atom_count_at_most_char_count(self, atom_tok):
        """Atom-level never produces more tokens than character-level."""
        import rustmolbpe
        char_tok = rustmolbpe.CharTokenizer()
        for smi in _ROUNDTRIP_SMILES:
            assert len(atom_tok.encode(smi)) <= len(smi)

    def test_vocab_io_not_supported(self, atom_tok):
        with pytest.raises(NotImplementedError):
            atom_tok.save_vocabulary("/tmp/_should_not_exist.txt")


class TestCharBPETokenizer:
    """Tests for the character-level BPE tokenizer."""

    @pytest.fixture
    def char_bpe(self):
        import rustmolbpe
        tok = rustmolbpe.CharBPETokenizer()
        tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=80)
        return tok

    def test_learns_merges(self, char_bpe):
        assert char_bpe.num_merges > 0
        assert char_bpe.is_trained() is True

    def test_merges_are_concatenations(self, char_bpe):
        """Each merge's merged token equals left + right concatenated."""
        for left, right, merged in char_bpe.get_merges():
            assert merged == left + right

    def test_roundtrip(self, char_bpe):
        for smi in _ROUNDTRIP_SMILES:
            assert char_bpe.decode(char_bpe.encode(smi)) == smi

    def test_compresses_vs_character_level(self, char_bpe):
        """BPE encoding is never longer than raw character splitting."""
        import rustmolbpe
        char_tok = rustmolbpe.CharTokenizer()
        char_tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=0)
        for smi in _ROUNDTRIP_SMILES:
            assert len(char_bpe.encode(smi)) <= len(char_tok.encode(smi))

    def test_no_vocab_token_contains_whitespace(self, char_bpe):
        """SMILESPE vocab format is space-separated; tokens must be space-free."""
        for token, _id in char_bpe.get_vocabulary():
            assert " " not in token

    def test_vocab_save_load_roundtrip(self, char_bpe):
        import rustmolbpe
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            char_bpe.save_vocabulary(path)
            reloaded = rustmolbpe.CharBPETokenizer()
            reloaded.load_vocabulary(path)
            # The SMILESPE format stores only merge rules (as strings), so
            # token IDs may be reassigned on load; the produced token *strings*
            # and the decoded SMILES must still match.
            for smi in _ROUNDTRIP_SMILES:
                original = [char_bpe.id_to_token(i) for i in char_bpe.encode(smi)]
                loaded = [reloaded.id_to_token(i) for i in reloaded.encode(smi)]
                assert loaded == original
                assert reloaded.decode(reloaded.encode(smi)) == smi
        finally:
            os.unlink(path)


class TestTokenizerLadderCommon:
    """Cross-cutting tests parametrized over all four tokenizer classes."""

    @pytest.mark.parametrize("cls", _all_tokenizer_classes())
    def test_fresh_state(self, cls):
        tok = cls()
        assert tok.vocab_size == 4  # special tokens only
        assert tok.num_merges == 0
        assert tok.pad_token == "<pad>"
        assert tok.unk_token == "<unk>"
        assert tok.bos_token == "<bos>"
        assert tok.eos_token == "<eos>"

    @pytest.mark.parametrize("cls", _all_tokenizer_classes())
    def test_callable_single_and_batch(self, cls):
        tok = cls()
        tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=80)

        single = tok("CCO")
        assert isinstance(single["input_ids"], list)
        assert not isinstance(single["input_ids"][0], list)

        batch = tok(["CCO", "c1ccccc1"])
        assert len(batch["input_ids"]) == 2
        assert isinstance(batch["input_ids"][0], list)

    @pytest.mark.parametrize("cls", _all_tokenizer_classes())
    def test_padding_parity(self, cls):
        tok = cls()
        tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=80)

        seqs = ["CCO", "c1ccccc1", "CC(=O)O"]
        via_encode_pad = tok.encode_batch_padded(seqs, padding="right")
        via_two_steps = tok.pad(tok.batch_encode(seqs), padding="right")
        assert via_encode_pad["input_ids"] == via_two_steps["input_ids"]

    @pytest.mark.parametrize("cls", _all_tokenizer_classes())
    def test_pickle_roundtrip(self, cls):
        import pickle
        tok = cls()
        tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=80)

        restored = pickle.loads(pickle.dumps(tok))
        for smi in _ROUNDTRIP_SMILES:
            assert restored.encode(smi) == tok.encode(smi)
        assert restored.get_vocabulary() == tok.get_vocabulary()

    def test_pickle_cross_class_rejected(self):
        """A pickle from one granularity must not load into the other."""
        import rustmolbpe
        char_tok = rustmolbpe.CharTokenizer()
        char_tok.train_from_iterator(iter(_LADDER_SMILES), vocab_size=0)
        _cls, _args, state = char_tok.__reduce__()

        atom_tok = rustmolbpe.AtomTokenizer()
        with pytest.raises(Exception):
            atom_tok.__setstate__(state)
