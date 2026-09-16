//! Vocabulary loading, saving, and query methods.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;

use crate::constants::{Pair, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN};

/// Initialize special tokens in the tokenizer vocabulary.
///
/// Always adds PAD=0, UNK=1, BOS=2, EOS=3.
pub(crate) fn init_special_tokens(
    atom_to_id: &mut AHashMap<CompactString, u32>,
    id_to_atom: &mut Vec<CompactString>,
) {
    // Only add if not already present
    if id_to_atom.is_empty() {
        for (id, token) in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
            .iter()
            .enumerate()
        {
            let token_str = CompactString::from(*token);
            atom_to_id.insert(token_str.clone(), id as u32);
            id_to_atom.push(token_str);
        }
    }
}

/// Load vocabulary from a SMILESPE-format file.
///
/// Format: one merge per line, `token1 token2` (space-separated)
/// Special tokens (PAD, UNK, BOS, EOS) are always added at IDs 0-3.
pub(crate) fn load_vocabulary(
    path: &str,
    merges: &mut Vec<(Pair, u32)>,
    atom_to_id: &mut AHashMap<CompactString, u32>,
    id_to_atom: &mut Vec<CompactString>,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
    let reader = BufReader::new(file);

    // Clear existing state and initialize special tokens first
    merges.clear();
    atom_to_id.clear();
    id_to_atom.clear();
    init_special_tokens(atom_to_id, id_to_atom);

    // First pass: collect all unique atoms from merge rules
    let mut all_atoms: AHashSet<CompactString> = AHashSet::new();
    let mut merge_pairs: Vec<(CompactString, CompactString)> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Split by space - first space separates the two tokens
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid line in vocabulary file: '{}'", trimmed));
        }

        let left = CompactString::from(parts[0]);
        let right = CompactString::from(parts[1]);
        all_atoms.insert(left.clone());
        all_atoms.insert(right.clone());
        merge_pairs.push((left, right));
    }

    // Build base vocabulary from atoms (sorted for determinism)
    let mut atoms_sorted: Vec<_> = all_atoms.into_iter().collect();
    atoms_sorted.sort();

    for atom in atoms_sorted {
        let id = id_to_atom.len() as u32;
        atom_to_id.insert(atom.clone(), id);
        id_to_atom.push(atom);
    }

    // Build merges in file (priority) order, adding merged tokens to the
    // vocabulary as we go. Token IDs are assigned exactly as before; only the
    // merge list keeps the file's order. A repeated rule keeps its first
    // occurrence, matching SmilesPE.
    let mut seen_pairs: AHashSet<Pair> = AHashSet::new();
    for (left, right) in merge_pairs.iter() {
        let left_id = *atom_to_id
            .get(left)
            .ok_or_else(|| format!("Unknown token: {}", left))?;
        let right_id = *atom_to_id
            .get(right)
            .ok_or_else(|| format!("Unknown token: {}", right))?;
        if !seen_pairs.insert((left_id, right_id)) {
            continue;
        }

        let merged_str = CompactString::from(format!("{}{}", left, right));

        // Get or create the merged token ID
        let new_id = if let Some(&existing_id) = atom_to_id.get(&merged_str) {
            // Token already exists (it appeared in merge rules), use existing ID
            existing_id
        } else {
            // Create new token
            let id = id_to_atom.len() as u32;
            atom_to_id.insert(merged_str.clone(), id);
            id_to_atom.push(merged_str);
            id
        };

        merges.push(((left_id, right_id), new_id));
    }

    let base_vocab_size = id_to_atom.len().saturating_sub(merges.len()) as u32;

    log::info!(
        "Loaded vocabulary: {} base atoms, {} merges, {} total vocab",
        base_vocab_size,
        merges.len(),
        id_to_atom.len()
    );

    Ok(())
}

/// Save vocabulary to a SMILESPE-format file.
///
/// Format: one merge per line, `token1 token2` (space-separated), in priority order.
pub(crate) fn save_vocabulary(
    path: &str,
    merges: &[(Pair, u32)],
    id_to_atom: &[CompactString],
) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Cannot create file: {}", e))?;

    for &((left_id, right_id), _) in merges {
        let left_str = &id_to_atom[left_id as usize];
        let right_str = &id_to_atom[right_id as usize];
        writeln!(file, "{} {}", left_str, right_str).map_err(|e| format!("Write error: {}", e))?;
    }

    Ok(())
}

/// Return the vocabulary as a list of (token_string, token_id) tuples.
///
/// Returns all tokens in the vocabulary including special tokens (PAD, UNK, BOS, EOS),
/// base atoms, and merged tokens. Tokens are returned in ID order.
pub(crate) fn get_vocabulary(id_to_atom: &[CompactString]) -> Vec<(String, u32)> {
    id_to_atom
        .iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), id as u32))
        .collect()
}

/// Return the merge rules as (left, right, merged) string tuples.
///
/// Merges are returned in priority order (the order they were learned or loaded).
pub(crate) fn get_merges(
    merges: &[(Pair, u32)],
    id_to_atom: &[CompactString],
) -> Vec<(String, String, String)> {
    merges
        .iter()
        .map(|&((left_id, right_id), merged_id)| {
            let left_str = id_to_atom[left_id as usize].to_string();
            let right_str = id_to_atom[right_id as usize].to_string();
            let merged_str = id_to_atom[merged_id as usize].to_string();
            (left_str, right_str, merged_str)
        })
        .collect()
}

/// Get token string for a given token ID.
pub(crate) fn id_to_token(id: u32, id_to_atom: &[CompactString]) -> Result<String, String> {
    if (id as usize) < id_to_atom.len() {
        Ok(id_to_atom[id as usize].to_string())
    } else {
        Err(format!("Unknown token id: {}", id))
    }
}

/// Get token ID for a given token string.
pub(crate) fn token_to_id(
    token: &str,
    atom_to_id: &AHashMap<CompactString, u32>,
) -> Result<u32, String> {
    atom_to_id
        .get(&CompactString::from(token))
        .copied()
        .ok_or_else(|| format!("Unknown token: {}", token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_special_tokens() {
        let mut atom_to_id = AHashMap::new();
        let mut id_to_atom = Vec::new();

        init_special_tokens(&mut atom_to_id, &mut id_to_atom);

        assert_eq!(id_to_atom.len(), 4);
        assert_eq!(id_to_atom[0].as_str(), PAD_TOKEN);
        assert_eq!(id_to_atom[1].as_str(), UNK_TOKEN);
        assert_eq!(id_to_atom[2].as_str(), BOS_TOKEN);
        assert_eq!(id_to_atom[3].as_str(), EOS_TOKEN);
        assert_eq!(atom_to_id.get(&CompactString::from(PAD_TOKEN)), Some(&0));
    }

    #[test]
    fn test_get_vocabulary() {
        let mut id_to_atom = Vec::new();
        id_to_atom.push(CompactString::from("<pad>"));
        id_to_atom.push(CompactString::from("<unk>"));
        id_to_atom.push(CompactString::from("C"));
        id_to_atom.push(CompactString::from("O"));

        let vocab = get_vocabulary(&id_to_atom);

        assert_eq!(vocab.len(), 4);
        assert_eq!(vocab[0], ("<pad>".to_string(), 0));
        assert_eq!(vocab[1], ("<unk>".to_string(), 1));
        assert_eq!(vocab[2], ("C".to_string(), 2));
        assert_eq!(vocab[3], ("O".to_string(), 3));
    }

    #[test]
    fn test_get_merges() {
        let mut id_to_atom = Vec::new();
        id_to_atom.push(CompactString::from("<pad>"));
        id_to_atom.push(CompactString::from("C"));
        id_to_atom.push(CompactString::from("O"));
        id_to_atom.push(CompactString::from("CC"));
        id_to_atom.push(CompactString::from("CO"));

        let mut merges = Vec::new();
        merges.push(((1, 1), 3)); // C + C -> CC
        merges.push(((1, 2), 4)); // C + O -> CO

        let result = get_merges(&merges, &id_to_atom);

        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            ("C".to_string(), "C".to_string(), "CC".to_string())
        );
        assert_eq!(
            result[1],
            ("C".to_string(), "O".to_string(), "CO".to_string())
        );
    }

    #[test]
    fn test_id_to_token() {
        let mut id_to_atom = Vec::new();
        id_to_atom.push(CompactString::from("C"));
        id_to_atom.push(CompactString::from("O"));

        assert_eq!(id_to_token(0, &id_to_atom).unwrap(), "C");
        assert_eq!(id_to_token(1, &id_to_atom).unwrap(), "O");
        assert!(id_to_token(999, &id_to_atom).is_err());
    }

    #[test]
    fn test_token_to_id() {
        let mut atom_to_id = AHashMap::new();
        atom_to_id.insert(CompactString::from("C"), 0);
        atom_to_id.insert(CompactString::from("O"), 1);

        assert_eq!(token_to_id("C", &atom_to_id).unwrap(), 0);
        assert_eq!(token_to_id("O", &atom_to_id).unwrap(), 1);
        assert!(token_to_id("N", &atom_to_id).is_err());
    }

    #[test]
    fn test_load_vocabulary_keeps_file_order_and_ids() {
        // "c c" produces "cc", which a later rule uses, so the loader gives "cc"
        // a low (alphabetical) ID while "CC" gets a new, higher ID. The merge
        // list must still follow the file; the repeated "C C" line is ignored.
        let path =
            std::env::temp_dir().join(format!("rustmolbpe_vocab_order_{}.txt", std::process::id()));
        std::fs::write(&path, "C C\nc c\ncc O\nC C\n").unwrap();

        let mut merges = Vec::new();
        let mut atom_to_id = AHashMap::new();
        let mut id_to_atom = Vec::new();
        load_vocabulary(
            path.to_str().unwrap(),
            &mut merges,
            &mut atom_to_id,
            &mut id_to_atom,
        )
        .unwrap();

        // Specials 0-3; rule tokens sorted: C=4, O=5, c=6, cc=7; new merged
        // tokens in file order: CC=8, ccO=9.
        assert_eq!(atom_to_id.get(&CompactString::from("cc")), Some(&7));
        assert_eq!(atom_to_id.get(&CompactString::from("CC")), Some(&8));
        assert_eq!(merges, vec![((4, 4), 8), ((6, 6), 7), ((7, 5), 9)]);

        let out = std::env::temp_dir().join(format!(
            "rustmolbpe_vocab_order_out_{}.txt",
            std::process::id()
        ));
        save_vocabulary(out.to_str().unwrap(), &merges, &id_to_atom).unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "C C\nc c\ncc O\n");

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&out).unwrap();
    }
}
