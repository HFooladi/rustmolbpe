//! Encoding and decoding methods for the SMILES tokenizer.

use ahash::AHashMap;
use compact_str::CompactString;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::atomwise_tokenize;

/// Encode a SMILES string into token IDs using greedy longest-match.
///
/// This produces better compression than the standard BPE merge-order approach.
/// Unknown atoms are encoded as UNK token.
pub(crate) fn encode(
    smiles: &str,
    add_special_tokens: bool,
    compiled_pattern: &Regex,
    atom_to_id: &AHashMap<CompactString, u32>,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
) -> Vec<u32> {
    // Step 1: Atomwise tokenization
    let atoms = atomwise_tokenize(smiles, compiled_pattern);

    if atoms.is_empty() {
        if add_special_tokens {
            return vec![bos_token_id, eos_token_id];
        }
        return Vec::new();
    }

    // Step 2: Greedy longest-match encoding
    // At each position, find the longest sequence of atoms that matches a token
    let mut result = Vec::new();

    // Add BOS token if requested
    if add_special_tokens {
        result.push(bos_token_id);
    }

    let mut pos = 0;

    while pos < atoms.len() {
        let mut best_len = 1;
        let mut best_id = atom_to_id.get(&atoms[pos]).copied();

        // Try progressively longer matches (up to reasonable limit)
        let max_len = std::cmp::min(atoms.len() - pos, 32); // Cap at 32 atoms
        let mut concat = atoms[pos].clone();

        for len in 2..=max_len {
            concat.push_str(&atoms[pos + len - 1]);
            if let Some(&id) = atom_to_id.get(&concat) {
                best_len = len;
                best_id = Some(id);
            }
        }

        // Use UNK token for unknown atoms instead of skipping
        result.push(best_id.unwrap_or(unk_token_id));
        pos += best_len;
    }

    // Add EOS token if requested
    if add_special_tokens {
        result.push(eos_token_id);
    }

    result
}

/// Decode token IDs back to a SMILES string.
///
/// # Arguments
/// * `ids` - Vector of token IDs to decode
/// * `id_to_atom` - Vector mapping token IDs to atom strings
///
/// # Returns
/// The reconstructed SMILES string, or an error if any token ID is invalid
pub(crate) fn decode(ids: &[u32], id_to_atom: &[CompactString]) -> Result<String, String> {
    let mut result = String::new();

    for &id in ids {
        if (id as usize) < id_to_atom.len() {
            result.push_str(&id_to_atom[id as usize]);
        } else {
            return Err(format!("Unknown token id: {}", id));
        }
    }

    Ok(result)
}

/// Encode multiple SMILES strings in parallel using rayon.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_encode(
    py: Python<'_>,
    smiles_list: &[String],
    add_special_tokens: bool,
    compiled_pattern: &Regex,
    atom_to_id: &AHashMap<CompactString, u32>,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
) -> Vec<Vec<u32>> {
    py.detach(|| {
        smiles_list
            .par_iter()
            .map(|smi| {
                encode(
                    smi,
                    add_special_tokens,
                    compiled_pattern,
                    atom_to_id,
                    bos_token_id,
                    eos_token_id,
                    unk_token_id,
                )
            })
            .collect::<Vec<Vec<u32>>>()
    })
}

/// Decode multiple token ID sequences in parallel.
pub(crate) fn batch_decode(
    py: Python<'_>,
    ids_list: &[Vec<u32>],
    id_to_atom: &[CompactString],
) -> Result<Vec<String>, String> {
    py.detach(|| {
        ids_list
            .par_iter()
            .map(|ids| decode(ids, id_to_atom))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::SMILES_ATOM_PATTERN;

    fn setup_tokenizer() -> (
        Regex,
        AHashMap<CompactString, u32>,
        Vec<CompactString>,
        u32,
        u32,
        u32,
        u32,
    ) {
        let compiled_pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let mut atom_to_id = AHashMap::new();
        let mut id_to_atom = Vec::new();

        // Special tokens
        id_to_atom.push(CompactString::from("<pad>"));
        atom_to_id.insert(CompactString::from("<pad>"), 0);
        id_to_atom.push(CompactString::from("<unk>"));
        atom_to_id.insert(CompactString::from("<unk>"), 1);
        id_to_atom.push(CompactString::from("<bos>"));
        atom_to_id.insert(CompactString::from("<bos>"), 2);
        id_to_atom.push(CompactString::from("<eos>"));
        atom_to_id.insert(CompactString::from("<eos>"), 3);

        // Atoms
        id_to_atom.push(CompactString::from("C"));
        atom_to_id.insert(CompactString::from("C"), 4);
        id_to_atom.push(CompactString::from("O"));
        atom_to_id.insert(CompactString::from("O"), 5);

        (compiled_pattern, atom_to_id, id_to_atom, 0, 1, 2, 3)
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (pattern, atom_to_id, id_to_atom, _pad, unk, bos, eos) = setup_tokenizer();

        let smiles = "CCO";
        let ids = encode(smiles, false, &pattern, &atom_to_id, bos, eos, unk);
        assert_eq!(ids, vec![4, 4, 5]);

        let decoded = decode(&ids, &id_to_atom).unwrap();
        assert_eq!(decoded, smiles);
    }

    #[test]
    fn test_encode_empty_smiles() {
        let (pattern, atom_to_id, _id_to_atom, _pad, unk, bos, eos) = setup_tokenizer();

        // Without special tokens
        let ids = encode("", false, &pattern, &atom_to_id, bos, eos, unk);
        assert!(ids.is_empty());

        // With special tokens
        let ids = encode("", true, &pattern, &atom_to_id, bos, eos, unk);
        assert_eq!(ids, vec![2, 3]); // BOS, EOS
    }

    #[test]
    fn test_encode_unknown_atoms() {
        let (pattern, atom_to_id, _id_to_atom, _pad, unk, bos, eos) = setup_tokenizer();

        // Encoding "CN" should use UNK for N (not in vocabulary)
        let ids = encode("CN", false, &pattern, &atom_to_id, bos, eos, unk);
        assert_eq!(ids, vec![4, 1]); // C, UNK
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let (pattern, atom_to_id, _id_to_atom, _pad, unk, bos, eos) = setup_tokenizer();

        let ids = encode("CO", true, &pattern, &atom_to_id, bos, eos, unk);
        assert_eq!(ids, vec![2, 4, 5, 3]); // BOS, C, O, EOS
    }

    #[test]
    fn test_decode_invalid_id() {
        let (_, _, id_to_atom, _, _, _, _) = setup_tokenizer();

        let result = decode(&[999], &id_to_atom);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown token id: 999"));
    }
}
