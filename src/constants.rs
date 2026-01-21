//! Constants and type aliases for the SMILES tokenizer.

/// SMILES atom-level tokenization regex pattern
/// Matches:
/// - Bracketed atoms: [C@@H], [nH], [O-], etc.
/// - Two-char elements: Br, Cl (must come before B, C)
/// - Single-char elements: C, N, O, S, P, F, I, B
/// - Aromatic atoms: b, c, n, o, s, p
/// - Bonds: =, #, -, :, ~
/// - Stereochemistry: @, /, \
/// - Branches: (, )
/// - Disconnected: .
/// - Ring numbers: single digit or %XX
/// - Other: +, ?, >, *, $
pub const SMILES_ATOM_PATTERN: &str = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])";

/// Special tokens for sequence modeling
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

/// Number of special tokens (always reserved at IDs 0-3)
pub const NUM_SPECIAL_TOKENS: u32 = 4;

/// Type alias for a pair of token IDs (used in merge rules)
pub type Pair = (u32, u32);
