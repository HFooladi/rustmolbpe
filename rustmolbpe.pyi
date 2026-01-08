"""Type stubs for rustmolbpe - High-performance BPE tokenizer for molecular SMILES."""

from typing import Dict, Iterator, List, Optional, Tuple

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

class SmilesTokenizer:
    """BPE tokenizer for molecular SMILES strings.

    A high-performance tokenizer that uses atom-level pre-tokenization
    followed by Byte Pair Encoding (BPE) for subword tokenization.

    Special tokens are always at fixed IDs:
        - PAD: 0
        - UNK: 1
        - BOS: 2
        - EOS: 3

    Examples:
        >>> tokenizer = SmilesTokenizer()
        >>> tokenizer.load_vocabulary("vocab.txt")
        >>> ids = tokenizer.encode("CCO")
        >>> smiles = tokenizer.decode(ids)
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

        Args:
            iterator: Iterator yielding SMILES strings
            vocab_size: Target vocabulary size (including special tokens and base atoms)
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
        """
        ...

    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary to a SMILESPE-format file.

        Args:
            path: Path to save vocabulary file

        Raises:
            IOError: If file cannot be written
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
        """Total vocabulary size (special + base atoms + merges)."""
        ...

    @property
    def base_vocab_size(self) -> int:
        """Number of base atom tokens (excluding special tokens and merges)."""
        ...

    @property
    def num_merges(self) -> int:
        """Number of learned merge operations."""
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
