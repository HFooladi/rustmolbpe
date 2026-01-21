//! Utility functions for SMILES tokenization.

use compact_str::CompactString;
use fancy_regex::Regex;
use pyo3::prelude::*;

use crate::constants::SMILES_ATOM_PATTERN;

/// Tokenize a SMILES string into atom-level tokens.
///
/// Splits a SMILES string into its constituent atoms and tokens using a regex pattern.
/// Handles multi-character atoms (Br, Cl), bracket atoms ([C@@H], [N+]), ring closures,
/// bonds, and stereochemistry markers.
pub(crate) fn atomwise_tokenize(smiles: &str, pattern: &Regex) -> Vec<CompactString> {
    let mut tokens = Vec::new();
    for m in pattern.find_iter(smiles).flatten() {
        tokens.push(CompactString::from(m.as_str()));
    }
    tokens
}

/// Tokenize a SMILES string into atom-level tokens (Python binding).
///
/// # Arguments
/// * `smiles` - The SMILES string to tokenize
///
/// # Returns
/// A vector of token strings representing the atoms and structural elements
///
/// # Example outputs
/// - `atomwise_tokenize("CCO")` returns `["C", "C", "O"]`
/// - `atomwise_tokenize("[C@@H](O)C")` returns `["[C@@H]", "(", "O", ")", "C"]`
#[pyfunction]
#[pyo3(name = "atomwise_tokenize")]
pub fn atomwise_tokenize_py(smiles: &str) -> Vec<String> {
    let pattern = Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern");
    atomwise_tokenize(smiles, &pattern)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomwise_tokenize_simple() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("CCO", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("C"),
                CompactString::from("C"),
                CompactString::from("O")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_halogen() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("CBr", &pattern);
        assert_eq!(
            tokens,
            vec![CompactString::from("C"), CompactString::from("Br")]
        );

        let tokens = atomwise_tokenize("CCl", &pattern);
        assert_eq!(
            tokens,
            vec![CompactString::from("C"), CompactString::from("Cl")]
        );
    }

    #[test]
    fn test_atomwise_tokenize_bracket() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("[C@@H](O)C", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("[C@@H]"),
                CompactString::from("("),
                CompactString::from("O"),
                CompactString::from(")"),
                CompactString::from("C")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_aromatic() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("c1ccccc1", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("c"),
                CompactString::from("1"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("1")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_ring_closure() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        // Two-digit ring closure with %
        let tokens = atomwise_tokenize("C%12CC%12", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("C"),
                CompactString::from("%12"),
                CompactString::from("C"),
                CompactString::from("C"),
                CompactString::from("%12")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_bonds() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("C=C#N", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("C"),
                CompactString::from("="),
                CompactString::from("C"),
                CompactString::from("#"),
                CompactString::from("N")
            ]
        );
    }
}
