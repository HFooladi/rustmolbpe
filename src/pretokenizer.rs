//! Pluggable pre-tokenization for SMILES strings.
//!
//! A [`PreTokenizer`] turns a raw SMILES string into a sequence of string units
//! before any BPE merging happens. Two granularities are supported:
//!
//! - [`PreTokenizerKind::Atom`] - regex-based atom-level splitting, where
//!   multi-character atoms (`Br`, `Cl`, `[C@@H]`) stay as single units.
//! - [`PreTokenizerKind::Char`] - character-level splitting, one Unicode scalar
//!   value per unit.
//!
//! The BPE training and greedy encoding code is granularity-agnostic: it only
//! sees `Vec<CompactString>` units, so swapping the pre-tokenizer is the only
//! difference between the atom-level and character-level tokenizers.

use compact_str::{CompactString, ToCompactString};
use fancy_regex::Regex;

use crate::constants::SMILES_ATOM_PATTERN;

/// The granularity at which a SMILES string is pre-tokenized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreTokenizerKind {
    /// Atom-level: regex split keeping multi-character atoms together.
    Atom,
    /// Character-level: one Unicode scalar value per unit.
    Char,
}

impl PreTokenizerKind {
    /// Stable string tag used for serialization (pickling).
    pub(crate) fn as_tag(&self) -> &'static str {
        match self {
            PreTokenizerKind::Atom => "atom",
            PreTokenizerKind::Char => "char",
        }
    }

    /// Parse a [`PreTokenizerKind`] from its serialization tag.
    pub(crate) fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "atom" => Some(PreTokenizerKind::Atom),
            "char" => Some(PreTokenizerKind::Char),
            _ => None,
        }
    }
}

/// A pre-tokenizer that splits a SMILES string into string units.
#[derive(Clone)]
pub(crate) struct PreTokenizer {
    kind: PreTokenizerKind,
    /// Compiled atom regex. Always built so the struct stays uniform and cheap
    /// to clone for pickling; only used by the [`PreTokenizerKind::Atom`] arm.
    pattern: Regex,
}

impl PreTokenizer {
    /// Create a pre-tokenizer for the given granularity.
    pub(crate) fn new(kind: PreTokenizerKind) -> Self {
        Self {
            kind,
            pattern: Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern"),
        }
    }

    /// The granularity of this pre-tokenizer.
    pub(crate) fn kind(&self) -> PreTokenizerKind {
        self.kind
    }

    /// Split a SMILES string into pre-tokenization units.
    ///
    /// For [`PreTokenizerKind::Char`] every unit is a single Unicode scalar
    /// value, so `units.concat() == smiles` always holds (lossless).
    /// For [`PreTokenizerKind::Atom`] characters not matched by the atom regex
    /// are dropped, matching the historical `atomwise_tokenize` behavior.
    pub(crate) fn split(&self, smiles: &str) -> Vec<CompactString> {
        match self.kind {
            PreTokenizerKind::Atom => crate::utils::atomwise_tokenize(smiles, &self.pattern),
            PreTokenizerKind::Char => smiles.chars().map(|c| c.to_compact_string()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_tag_roundtrip() {
        for kind in [PreTokenizerKind::Atom, PreTokenizerKind::Char] {
            assert_eq!(PreTokenizerKind::from_tag(kind.as_tag()), Some(kind));
        }
        assert_eq!(PreTokenizerKind::from_tag("bogus"), None);
    }

    #[test]
    fn test_char_split_keeps_every_character() {
        let pt = PreTokenizer::new(PreTokenizerKind::Char);
        // "Cl" must split into two characters, NOT one chlorine atom.
        let units = pt.split("Cl");
        assert_eq!(
            units,
            vec![CompactString::from("C"), CompactString::from("l")]
        );
    }

    #[test]
    fn test_atom_split_groups_multichar_atoms() {
        let pt = PreTokenizer::new(PreTokenizerKind::Atom);
        // "CCl" must split into "C" + "Cl" (chlorine grouped).
        let units = pt.split("CCl");
        assert_eq!(
            units,
            vec![CompactString::from("C"), CompactString::from("Cl")]
        );
    }

    #[test]
    fn test_atom_split_bracket_atom() {
        let pt = PreTokenizer::new(PreTokenizerKind::Atom);
        let units = pt.split("[C@@H]");
        assert_eq!(units, vec![CompactString::from("[C@@H]")]);
    }

    #[test]
    fn test_char_split_roundtrip_is_lossless() {
        let pt = PreTokenizer::new(PreTokenizerKind::Char);
        for smiles in ["CCO", "[C@@H](O)Cl", "c1ccccc1", "C%12CC%12", ""] {
            let units = pt.split(smiles);
            let joined: String = units.iter().map(|u| u.as_str()).collect();
            assert_eq!(joined, smiles, "char split must be lossless for {smiles}");
        }
    }

    #[test]
    fn test_char_split_handles_multibyte() {
        // A multi-byte character must be one unit, not split mid-UTF-8.
        let pt = PreTokenizer::new(PreTokenizerKind::Char);
        let units = pt.split("C\u{00e9}O");
        assert_eq!(
            units,
            vec![
                CompactString::from("C"),
                CompactString::from("\u{00e9}"),
                CompactString::from("O"),
            ]
        );
    }

    #[test]
    fn test_char_split_empty() {
        let pt = PreTokenizer::new(PreTokenizerKind::Char);
        assert!(pt.split("").is_empty());
    }
}
