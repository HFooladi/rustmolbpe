"""Type stubs for rustmolbpe - High-performance tokenizers for molecular SMILES."""

from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar

_T = TypeVar("_T", bound="_BaseTokenizer")

__version__: str

def atomwise_tokenize(smiles: str) -> List[str]:
    """Tokenize a SMILES string into atom-level tokens.

    Args:
        smiles: SMILES string to tokenize

    Returns:
        List of atom-level tokens

    Examples:
        >>> atomwise_tokenize("CCO")
        ['C', 'C', 'O']
        >>> atomwise_tokenize("c1ccccc1")
        ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']
        >>> atomwise_tokenize("[C@@H](O)C")
        ['[C@@H]', '(', 'O', ')', 'C']
    """
    ...

class _BaseTokenizer:
    """Shared interface for all rustmolbpe tokenizers.

    Not instantiated directly. The four concrete tokenizers
    (:class:`CharTokenizer`, :class:`AtomTokenizer`, :class:`CharBPETokenizer`,
    :class:`SmilesTokenizer`) all expose this identical API and differ only in
    pre-tokenization granularity and whether they learn BPE merges.

    Special tokens are always at fixed IDs:
        - PAD: 0
        - UNK: 1
        - BOS: 2
        - EOS: 3
    """

    def __init__(self) -> None:
        """Create a new tokenizer with special tokens initialized."""
        ...

    # Training
    def train_from_iterator(
        self,
        iterator: Iterator[str],
        vocab_size: int,
        buffer_size: int = 8192,
        min_frequency: int = 2,
    ) -> None:
        """Train the tokenizer from a SMILES iterator.

        For BPE tokenizers this builds the base vocabulary and then learns up to
        ``vocab_size - base_vocab_size`` merges. For non-BPE tokenizers
        (:class:`CharTokenizer`, :class:`AtomTokenizer`) it only builds the base
        vocabulary and ``vocab_size`` is ignored.

        Args:
            iterator: Iterator yielding SMILES strings
            vocab_size: Target vocabulary size (including special tokens and base units)
            buffer_size: Number of SMILES to buffer for parallel processing
            min_frequency: Minimum frequency for a pair to be merged
        """
        ...

    # Vocabulary I/O
    def load_vocabulary(self, path: str) -> None:
        """Load vocabulary from a SMILESPE-format file.

        Args:
            path: Path to vocabulary file

        Raises:
            IOError: If file cannot be read
            NotImplementedError: For non-BPE tokenizers (CharTokenizer, AtomTokenizer)
        """
        ...

    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary to a SMILESPE-format file.

        Args:
            path: Path to save vocabulary file

        Raises:
            IOError: If file cannot be written
            NotImplementedError: For non-BPE tokenizers (CharTokenizer, AtomTokenizer)
        """
        ...

    # HuggingFace JSON interop
    def save_huggingface(self, path: str) -> None:
        """Export the tokenizer to a HuggingFace ``tokenizers`` JSON file.

        The emitted ``tokenizer.json`` is loadable by
        ``transformers.PreTrainedTokenizerFast(tokenizer_file=...)`` and the
        ``tokenizers`` library. :class:`CharTokenizer` and
        :class:`CharBPETokenizer` export as a ``BPE`` model; :class:`AtomTokenizer`
        as a ``WordLevel`` model with an atom-regex ``Split`` pre-tokenizer.

        Note:
            HuggingFace applies merge-order BPE while rustmolbpe uses greedy
            longest-match, so for :class:`CharBPETokenizer` the vocabulary and
            merges transfer exactly but token *sequences* may differ slightly.

        Args:
            path: Path to write the ``tokenizer.json`` file.

        Raises:
            IOError: If the file cannot be written.
            NotImplementedError: For :class:`SmilesTokenizer` (atom-level BPE
                cannot be expressed as a stock HuggingFace fast tokenizer).
        """
        ...

    @classmethod
    def from_huggingface(cls: type[_T], path: str) -> _T:
        """Load a tokenizer from a HuggingFace ``tokenizers`` JSON file.

        Accepts files written by :meth:`save_huggingface` (or compatible
        character-level ``BPE`` / atom-level ``WordLevel`` tokenizers). The file's
        granularity and merge profile must match the class this is called on.

        Args:
            path: Path to a HuggingFace ``tokenizer.json`` file.

        Returns:
            A new tokenizer instance of the calling class.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file is malformed, or its granularity / merge
                profile does not match this tokenizer class.
        """
        ...

    # Encoding/Decoding
    def encode(self, smiles: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a SMILES string to token IDs.

        Args:
            smiles: SMILES string to encode
            add_special_tokens: If True, add BOS at start and EOS at end

        Returns:
            List of token IDs
        """
        ...

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a SMILES string.

        Args:
            ids: List of token IDs

        Returns:
            Decoded SMILES string

        Raises:
            ValueError: If an ID is not in vocabulary
        """
        ...

    def batch_encode(
        self, smiles_list: List[str], add_special_tokens: bool = False
    ) -> List[List[int]]:
        """Encode multiple SMILES strings in parallel.

        Args:
            smiles_list: List of SMILES strings
            add_special_tokens: If True, add BOS/EOS tokens

        Returns:
            List of token ID lists
        """
        ...

    def batch_decode(self, ids_list: List[List[int]]) -> List[str]:
        """Decode multiple token sequences in parallel.

        Args:
            ids_list: List of token ID lists

        Returns:
            List of decoded SMILES strings
        """
        ...

    # Padding
    def pad(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding: str = "right",
        truncation: bool = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """Pad sequences to equal length.

        Args:
            sequences: List of token ID sequences
            max_length: Target length. If None, uses longest sequence length
            padding: Padding side, either "right" or "left"
            truncation: If True, truncate sequences longer than max_length
            return_attention_mask: If True, include attention_mask in result

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        ...

    def encode_batch_padded(
        self,
        smiles_list: List[str],
        max_length: Optional[int] = None,
        padding: str = "right",
        truncation: bool = False,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """Encode multiple SMILES and pad to equal length.

        Convenience method combining batch_encode and pad.

        Args:
            smiles_list: List of SMILES strings
            max_length: Target length. If None, uses longest sequence length
            padding: Padding side, either "right" or "left"
            truncation: If True, truncate sequences longer than max_length
            add_special_tokens: If True, add BOS/EOS tokens
            return_attention_mask: If True, include attention_mask in result

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        ...

    # Vocabulary access
    def get_vocabulary(self) -> List[Tuple[str, int]]:
        """Get vocabulary as (token, id) pairs.

        Returns:
            List of (token_string, token_id) tuples
        """
        ...

    def id_to_token(self, id: int) -> str:
        """Convert token ID to token string.

        Args:
            id: Token ID

        Returns:
            Token string

        Raises:
            ValueError: If ID is not in vocabulary
        """
        ...

    def token_to_id(self, token: str) -> int:
        """Convert token string to token ID.

        Args:
            token: Token string

        Returns:
            Token ID

        Raises:
            ValueError: If token is not in vocabulary
        """
        ...

    # Properties
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (special + base units + merges)."""
        ...

    @property
    def base_vocab_size(self) -> int:
        """Number of base unit tokens (excluding special tokens and merges)."""
        ...

    @property
    def num_merges(self) -> int:
        """Number of learned merge operations (always 0 for non-BPE tokenizers)."""
        ...

    def is_trained(self) -> bool:
        """Check whether the tokenizer has learned BPE merges.

        Always False for non-BPE tokenizers (CharTokenizer, AtomTokenizer) even
        after a vocabulary pass; use :meth:`has_vocabulary` for those.

        Returns:
            True if the tokenizer has merge rules (trained or loaded), False otherwise.
        """
        ...

    def has_vocabulary(self) -> bool:
        """Check whether a base vocabulary has been built.

        Returns:
            True if the vocabulary contains tokens beyond the 4 special tokens.
        """
        ...

    def get_merges(self) -> List[Tuple[str, str, str]]:
        """Get the learned merge rules as (left, right, merged) tuples.

        Returns a list of tuples where each tuple contains:
        - left: The left token string in the merge rule
        - right: The right token string in the merge rule
        - merged: The resulting merged token string

        Merges are returned in order of learning priority. Empty for non-BPE
        tokenizers.

        Returns:
            List of (left, right, merged) string tuples.
        """
        ...

    def __call__(
        self,
        text: str | List[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, Any]:
        """Encode SMILES string(s), like HuggingFace tokenizers.

        Accepts a single SMILES string or a list of SMILES strings.
        Returns a dict with 'input_ids' and optionally 'attention_mask'.

        For a single string input, values are flat lists.
        For a list input, values are nested lists.

        Args:
            text: A SMILES string or list of SMILES strings
            padding: If True, pad sequences to equal length (right padding)
            truncation: If True, truncate sequences to max_length
            max_length: Maximum sequence length for padding/truncation
            add_special_tokens: If True, add BOS/EOS tokens
            return_attention_mask: If True, include attention_mask in result

        Returns:
            Dict with "input_ids" and optionally "attention_mask"

        Examples:
            >>> tokenizer("CCO")
            {'input_ids': [4, 4, 5], 'attention_mask': [1, 1, 1]}
            >>> tokenizer(["CCO", "C"], padding=True)
            {'input_ids': [[4, 4, 5], [4, 0, 0]], 'attention_mask': [[1, 1, 1], [1, 0, 0]]}
        """
        ...

    def __reduce__(self) -> Tuple[type, Tuple[()], Dict[str, Any]]:
        """Pickle support: returns (cls, args, state) for serialization."""
        ...

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the tokenizer state from pickle serialization.

        Raises:
            ValueError: If the pickled granularity does not match this class.
        """
        ...

    @property
    def pad_token_id(self) -> int:
        """PAD token ID (always 0)."""
        ...

    @property
    def unk_token_id(self) -> int:
        """UNK token ID (always 1)."""
        ...

    @property
    def bos_token_id(self) -> int:
        """BOS token ID (always 2)."""
        ...

    @property
    def eos_token_id(self) -> int:
        """EOS token ID (always 3)."""
        ...

    @property
    def pad_token(self) -> str:
        """PAD token string ('<pad>')."""
        ...

    @property
    def unk_token(self) -> str:
        """UNK token string ('<unk>')."""
        ...

    @property
    def bos_token(self) -> str:
        """BOS token string ('<bos>')."""
        ...

    @property
    def eos_token(self) -> str:
        """EOS token string ('<eos>')."""
        ...

class CharTokenizer(_BaseTokenizer):
    """Character-level SMILES tokenizer.

    Splits a SMILES string into individual Unicode characters with no BPE
    merges. ``vocab_size`` passed to :meth:`train_from_iterator` is ignored and
    ``num_merges`` is always 0. Vocabulary file I/O is not supported (the
    SMILESPE format only stores merge rules).

    Examples:
        >>> tok = CharTokenizer()
        >>> tok.encode("Cl")  # two characters, not one chlorine atom
        [...]  # length 2
    """

class AtomTokenizer(_BaseTokenizer):
    """Atom-level SMILES tokenizer.

    Splits a SMILES string into atoms and structural tokens using the SMILES
    atom regex (multi-character atoms such as ``Br``, ``Cl``, ``[C@@H]`` stay
    together) with no BPE merges. ``vocab_size`` is ignored, ``num_merges`` is
    always 0, and vocabulary file I/O is not supported.

    Examples:
        >>> tok = AtomTokenizer()
        >>> tok.encode("CCl")  # 'C' + 'Cl'
        [...]  # length 2
    """

class CharBPETokenizer(_BaseTokenizer):
    """Character-level BPE tokenizer.

    Pre-tokenizes a SMILES string into characters, then applies BPE merges
    learned by frequency during :meth:`train_from_iterator`.
    """

class SmilesTokenizer(_BaseTokenizer):
    """Atom-level BPE tokenizer (SMILES Pair Encoding, "SPE").

    Pre-tokenizes a SMILES string into atoms, then applies BPE merges learned by
    frequency during training. This is the original rustmolbpe tokenizer.

    Examples:
        >>> tokenizer = SmilesTokenizer()
        >>> tokenizer.load_vocabulary("vocab.txt")
        >>> ids = tokenizer.encode("CCO")
        >>> smiles = tokenizer.decode(ids)
    """

# AtomBPETokenizer is an exact alias of SmilesTokenizer (atom-level BPE, "SPE").
# Both names bind to the same class object: ``AtomBPETokenizer is SmilesTokenizer``.
AtomBPETokenizer = SmilesTokenizer

class ByteBPETokenizer(_BaseTokenizer):
    """Byte-level BPE tokenizer.

    Pre-tokenizes a SMILES string into raw UTF-8 bytes, then applies BPE merges
    learned by frequency during :meth:`train_from_iterator`. The base alphabet
    is always the 256 byte values, so every input is representable and ``<unk>``
    is never produced; encode/decode is a guaranteed lossless round-trip.

    ``base_vocab_size`` is therefore always 260 (4 special tokens + 256 bytes)
    once trained. SMILESPE and HuggingFace file I/O are not supported (the
    formats store chemically-readable tokens, not raw bytes); use ``pickle`` to
    persist a byte-level tokenizer. :meth:`load_vocabulary`,
    :meth:`save_vocabulary`, :meth:`save_huggingface` and
    :meth:`from_huggingface` raise ``NotImplementedError``.

    Examples:
        >>> tok = ByteBPETokenizer()
        >>> tok.train_from_iterator(["CCO", "c1ccccc1"], vocab_size=300)
        >>> tok.decode(tok.encode("CCO")) == "CCO"
        True
    """
