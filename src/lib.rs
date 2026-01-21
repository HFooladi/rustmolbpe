//! rustmolbpe - A high-performance BPE tokenizer for molecular SMILES strings.
//!
//! This crate provides a BPE (Byte Pair Encoding) tokenizer specifically designed
//! for molecular SMILES strings. Unlike byte-level BPE tokenizers, this tokenizer
//! uses atom-level pre-tokenization where multi-character atoms (like Br, Cl, [C@@H])
//! are treated as single tokens.

mod constants;
mod encoding;
mod padding;
mod serialization;
mod training;
mod utils;
mod vocabulary;
mod word;

use std::collections::HashMap as StdHashMap;

use ahash::AHashMap;
use compact_str::CompactString;
use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

// Re-export public constants
pub use constants::{
    Pair, BOS_TOKEN, EOS_TOKEN, NUM_SPECIAL_TOKENS, PAD_TOKEN, SMILES_ATOM_PATTERN, UNK_TOKEN,
};

// Re-export the Python function
pub use utils::atomwise_tokenize_py;

use constants::Pair as PairType;
use word::Word;

/// A BPE tokenizer specifically designed for molecular SMILES strings.
///
/// Unlike byte-level BPE tokenizers, this tokenizer uses atom-level pre-tokenization
/// where multi-character atoms (like Br, Cl, [C@@H]) are treated as single tokens.
#[pyclass(module = "rustmolbpe")]
pub struct SmilesTokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<PairType, u32>,
    /// Maps atom strings to token IDs
    pub atom_to_id: AHashMap<CompactString, u32>,
    /// Reverse mapping: token ID to atom string
    pub id_to_atom: Vec<CompactString>,
    /// Compiled SMILES regex pattern
    compiled_pattern: Regex,
}

impl Default for SmilesTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Public methods for the SmilesTokenizer class that will be exposed to Python.
#[pymethods]
impl SmilesTokenizer {
    /// Create a new SmilesTokenizer with special tokens initialized
    #[new]
    pub fn new() -> Self {
        let mut tokenizer = Self {
            merges: StdHashMap::new(),
            atom_to_id: AHashMap::new(),
            id_to_atom: Vec::new(),
            compiled_pattern: Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern"),
        };
        vocabulary::init_special_tokens(&mut tokenizer.atom_to_id, &mut tokenizer.id_to_atom);
        tokenizer
    }

    /// Get the PAD token ID (always 0)
    #[getter]
    pub fn pad_token_id(&self) -> u32 {
        0
    }

    /// Get the UNK token ID (always 1)
    #[getter]
    pub fn unk_token_id(&self) -> u32 {
        1
    }

    /// Get the BOS token ID (always 2)
    #[getter]
    pub fn bos_token_id(&self) -> u32 {
        2
    }

    /// Get the EOS token ID (always 3)
    #[getter]
    pub fn eos_token_id(&self) -> u32 {
        3
    }

    /// Get the PAD token string
    #[getter]
    pub fn pad_token(&self) -> &'static str {
        PAD_TOKEN
    }

    /// Get the UNK token string
    #[getter]
    pub fn unk_token(&self) -> &'static str {
        UNK_TOKEN
    }

    /// Get the BOS token string
    #[getter]
    pub fn bos_token(&self) -> &'static str {
        BOS_TOKEN
    }

    /// Get the EOS token string
    #[getter]
    pub fn eos_token(&self) -> &'static str {
        EOS_TOKEN
    }

    /// Train from a streaming iterator of SMILES strings.
    ///
    /// # Arguments
    /// * `iterator` - Python iterator yielding SMILES strings
    /// * `vocab_size` - Target vocabulary size (base atoms + merges)
    /// * `buffer_size` - Number of SMILES to process per batch (default: 8192)
    /// * `min_frequency` - Minimum frequency for a merge to be learned (default: 2)
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, min_frequency=2))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, min_frequency=2)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        min_frequency: u32,
    ) -> PyResult<()> {
        // Clear existing state and reinitialize special tokens
        self.merges.clear();
        self.atom_to_id.clear();
        self.id_to_atom.clear();
        vocabulary::init_special_tokens(&mut self.atom_to_id, &mut self.id_to_atom);

        // Prepare Python iterator
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Global atom counts and SMILES counts
        let mut atom_counts: AHashMap<CompactString, u64> = AHashMap::new();
        let mut smiles_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();

        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "Processing SMILES from iterator (buffer_size: {})",
            buffer_size
        );
        let mut total_smiles = 0u64;

        // Helper: refill buffer from Python iterator
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::attach(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            })
        };

        // Stream ingestion loop
        let pattern = self.compiled_pattern.clone();
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_smiles += buf.len() as u64;

            // Process buffer in parallel
            let local: Vec<(AHashMap<CompactString, u64>, Vec<CompactString>)> = py.detach(|| {
                buf.par_iter()
                    .map(|smi| {
                        let atoms = utils::atomwise_tokenize(smi, &pattern);
                        let mut atom_map: AHashMap<CompactString, u64> = AHashMap::new();
                        for atom in &atoms {
                            *atom_map.entry(atom.clone()).or_default() += 1;
                        }
                        (atom_map, atoms)
                    })
                    .collect()
            });

            // Merge into global counts
            for (local_atom_counts, atoms) in local {
                for (k, v) in local_atom_counts {
                    *atom_counts.entry(k).or_default() += v;
                }
                *smiles_counts.entry(atoms).or_default() += 1;
            }

            if exhausted {
                break;
            }
        }

        log::info!(
            "Processed {} SMILES total, {} unique atom sequences, {} unique atoms",
            total_smiles,
            smiles_counts.len(),
            atom_counts.len()
        );

        // Build base vocabulary from atoms (sorted by frequency, then alphabetically)
        let mut atoms_sorted: Vec<_> = atom_counts.into_iter().collect();
        atoms_sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for (atom, _count) in atoms_sorted {
            let id = self.id_to_atom.len() as u32;
            self.atom_to_id.insert(atom.clone(), id);
            self.id_to_atom.push(atom);
        }

        let base_vocab_size = self.id_to_atom.len() as u32;
        log::info!("Built base vocabulary with {} atoms", base_vocab_size);

        if vocab_size <= base_vocab_size {
            log::info!(
                "vocab_size ({}) <= base vocab size ({}), no merges needed",
                vocab_size,
                base_vocab_size
            );
            return Ok(());
        }

        let num_merges = vocab_size - base_vocab_size;

        // Convert SMILES to Words
        let mut words = Vec::with_capacity(smiles_counts.len());
        let mut cvec = Vec::with_capacity(smiles_counts.len());

        for (atoms, count) in smiles_counts.into_iter() {
            if count >= min_frequency as i32 {
                let ids: Vec<u32> = atoms
                    .iter()
                    .map(|a| *self.atom_to_id.get(a).unwrap())
                    .collect();
                words.push(Word::new(ids));
                cvec.push(count);
            }
        }

        training::train_core_incremental(
            &mut self.merges,
            &mut self.atom_to_id,
            &mut self.id_to_atom,
            words,
            cvec,
            num_merges,
        );
        Ok(())
    }

    /// Load vocabulary from a SMILESPE-format file.
    /// Format: one merge per line, `token1 token2` (space-separated)
    /// Special tokens (PAD, UNK, BOS, EOS) are always added at IDs 0-3.
    #[pyo3(signature = (path))]
    pub fn load_vocabulary(&mut self, path: &str) -> PyResult<()> {
        vocabulary::load_vocabulary(
            path,
            &mut self.merges,
            &mut self.atom_to_id,
            &mut self.id_to_atom,
        )
        .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Save vocabulary to a SMILESPE-format file.
    /// Format: one merge per line, `token1 token2` (space-separated)
    #[pyo3(signature = (path))]
    pub fn save_vocabulary(&self, path: &str) -> PyResult<()> {
        vocabulary::save_vocabulary(path, &self.merges, &self.id_to_atom)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Return the vocabulary size (base atoms + merges)
    #[getter]
    pub fn vocab_size(&self) -> u32 {
        self.id_to_atom.len() as u32
    }

    /// Return the number of base atoms in the vocabulary
    #[getter]
    pub fn base_vocab_size(&self) -> u32 {
        (self.id_to_atom.len() - self.merges.len()) as u32
    }

    /// Return the number of learned merges
    #[getter]
    pub fn num_merges(&self) -> u32 {
        self.merges.len() as u32
    }

    /// Check if the tokenizer has been trained or has a vocabulary loaded.
    /// Returns true if there are merge rules, false otherwise.
    pub fn is_trained(&self) -> bool {
        !self.merges.is_empty()
    }

    /// Return the learned merge rules as (left, right, merged) string tuples.
    /// Merges are returned in order of learning priority (by merged token ID).
    pub fn get_merges(&self) -> Vec<(String, String, String)> {
        vocabulary::get_merges(&self.merges, &self.id_to_atom)
    }

    /// Return the vocabulary as a list of (token_string, token_id) tuples.
    ///
    /// Returns all tokens in the vocabulary including special tokens (PAD, UNK, BOS, EOS),
    /// base atoms, and merged tokens. Tokens are returned in ID order.
    pub fn get_vocabulary(&self) -> Vec<(String, u32)> {
        vocabulary::get_vocabulary(&self.id_to_atom)
    }

    /// Encode a SMILES string into token IDs using greedy longest-match.
    /// This produces better compression than the standard BPE merge-order approach.
    /// Unknown atoms are encoded as UNK token.
    #[pyo3(signature = (smiles, add_special_tokens=false))]
    pub fn encode(&self, smiles: &str, add_special_tokens: bool) -> Vec<u32> {
        encoding::encode(
            smiles,
            add_special_tokens,
            &self.compiled_pattern,
            &self.atom_to_id,
            self.bos_token_id(),
            self.eos_token_id(),
            self.unk_token_id(),
        )
    }

    /// Decode token IDs back to a SMILES string.
    ///
    /// # Arguments
    /// * `ids` - Vector of token IDs to decode
    ///
    /// # Returns
    /// The reconstructed SMILES string
    ///
    /// # Errors
    /// Returns `PyValueError` if any token ID is not in the vocabulary
    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        encoding::decode(&ids, &self.id_to_atom).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Encode multiple SMILES strings in parallel using rayon.
    #[pyo3(signature = (smiles_list, add_special_tokens=false))]
    #[pyo3(text_signature = "(self, smiles_list, add_special_tokens=False)")]
    pub fn batch_encode(
        &self,
        py: Python<'_>,
        smiles_list: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        Ok(encoding::batch_encode(
            py,
            &smiles_list,
            add_special_tokens,
            &self.compiled_pattern,
            &self.atom_to_id,
            self.bos_token_id(),
            self.eos_token_id(),
            self.unk_token_id(),
        ))
    }

    /// Decode multiple token ID sequences in parallel.
    #[pyo3(signature = (ids_list))]
    pub fn batch_decode(&self, py: Python<'_>, ids_list: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        encoding::batch_decode(py, &ids_list, &self.id_to_atom)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Pad a batch of token sequences to the same length.
    ///
    /// # Arguments
    /// * `sequences` - List of token ID sequences to pad
    /// * `max_length` - Maximum length to pad/truncate to (None = use longest sequence)
    /// * `padding` - Padding side: "right" (default) or "left"
    /// * `truncation` - Whether to truncate sequences longer than max_length
    /// * `return_attention_mask` - Whether to return attention masks
    ///
    /// # Returns
    /// A dict with 'input_ids' and optionally 'attention_mask'
    #[pyo3(signature = (sequences, max_length=None, padding="right", truncation=false, return_attention_mask=true))]
    pub fn pad(
        &self,
        sequences: Vec<Vec<u32>>,
        max_length: Option<usize>,
        padding: &str,
        truncation: bool,
        return_attention_mask: bool,
    ) -> PyResult<std::collections::HashMap<String, Vec<Vec<u32>>>> {
        Ok(padding::pad(
            sequences,
            max_length,
            padding,
            truncation,
            return_attention_mask,
            self.pad_token_id(),
        ))
    }

    /// Encode and pad a batch of SMILES strings in one call.
    ///
    /// Convenience method that combines batch_encode and pad.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (smiles_list, max_length=None, padding="right", truncation=false, add_special_tokens=false, return_attention_mask=true))]
    pub fn encode_batch_padded(
        &self,
        py: Python<'_>,
        smiles_list: Vec<String>,
        max_length: Option<usize>,
        padding: &str,
        truncation: bool,
        add_special_tokens: bool,
        return_attention_mask: bool,
    ) -> PyResult<std::collections::HashMap<String, Vec<Vec<u32>>>> {
        let sequences = self.batch_encode(py, smiles_list, add_special_tokens)?;
        self.pad(
            sequences,
            max_length,
            padding,
            truncation,
            return_attention_mask,
        )
    }

    /// Get token string for a given token ID.
    ///
    /// # Arguments
    /// * `id` - The token ID to look up
    ///
    /// # Returns
    /// The token string corresponding to the ID
    ///
    /// # Errors
    /// Returns `PyValueError` if the ID is not in the vocabulary
    pub fn id_to_token(&self, id: u32) -> PyResult<String> {
        vocabulary::id_to_token(id, &self.id_to_atom)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Get token ID for a given token string.
    ///
    /// # Arguments
    /// * `token` - The token string to look up
    ///
    /// # Returns
    /// The token ID corresponding to the string
    ///
    /// # Errors
    /// Returns `PyValueError` if the token is not in the vocabulary
    pub fn token_to_id(&self, token: &str) -> PyResult<u32> {
        vocabulary::token_to_id(token, &self.atom_to_id)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Pickle support: return (cls, args, state) for serialization
    pub fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        pyo3::Bound<'py, pyo3::types::PyType>,
        pyo3::Bound<'py, pyo3::types::PyTuple>,
        pyo3::Bound<'py, PyDict>,
    )> {
        let cls = py.get_type::<Self>();
        serialization::create_pickle_state(py, cls, &self.id_to_atom, &self.merges)
    }

    /// Pickle support: restore state from serialization
    pub fn __setstate__(&mut self, state: pyo3::Bound<'_, PyDict>) -> PyResult<()> {
        let (id_to_atom, atom_to_id, merges, compiled_pattern) =
            serialization::restore_from_pickle_state(&state)?;

        self.id_to_atom = id_to_atom;
        self.atom_to_id = atom_to_id;
        self.merges = merges;
        self.compiled_pattern = compiled_pattern;

        Ok(())
    }
}

/// A BPE tokenizer for molecular SMILES with Python bindings
#[pymodule]
fn rustmolbpe(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<SmilesTokenizer>()?;
    m.add_function(wrap_pyfunction!(utils::atomwise_tokenize_py, m)?)?;
    Ok(())
}

// ============================================================================
// RUST TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_new() {
        let tok = SmilesTokenizer::new();
        assert!(tok.merges.is_empty());
        // Special tokens are initialized: PAD, UNK, BOS, EOS
        assert_eq!(tok.atom_to_id.len(), 4);
        assert_eq!(tok.id_to_atom.len(), 4);
        assert_eq!(tok.id_to_atom[0].as_str(), PAD_TOKEN);
        assert_eq!(tok.id_to_atom[1].as_str(), UNK_TOKEN);
        assert_eq!(tok.id_to_atom[2].as_str(), BOS_TOKEN);
        assert_eq!(tok.id_to_atom[3].as_str(), EOS_TOKEN);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut tok = SmilesTokenizer::new();

        // Manually add atoms after special tokens (IDs 0-3)
        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);

        let smiles = "CCO";
        let ids = tok.encode(smiles, false);
        assert_eq!(ids, vec![4, 4, 5]);

        let decoded = tok.decode(ids).unwrap();
        assert_eq!(decoded, smiles);
    }

    #[test]
    fn test_encode_with_merge() {
        let mut tok = SmilesTokenizer::new();

        // Set up vocabulary after special tokens (IDs 0-3)
        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.id_to_atom.push(CompactString::from("CC")); // ID 6, merged token
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);
        tok.atom_to_id.insert(CompactString::from("CC"), 6);

        // Add merge rule: (4, 4) -> 6  (C + C -> CC)
        tok.merges.insert((4, 4), 6);

        let ids = tok.encode("CCO", false);
        assert_eq!(ids, vec![6, 5]); // CC, O

        let decoded = tok.decode(ids).unwrap();
        assert_eq!(decoded, "CCO");
    }

    #[test]
    fn test_is_trained() {
        let mut tok = SmilesTokenizer::new();
        assert!(!tok.is_trained());

        // Add a merge rule
        tok.merges.insert((4, 5), 6);
        assert!(tok.is_trained());
    }

    #[test]
    fn test_get_merges() {
        let mut tok = SmilesTokenizer::new();

        // Set up vocabulary
        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.id_to_atom.push(CompactString::from("CC")); // ID 6
        tok.id_to_atom.push(CompactString::from("CO")); // ID 7
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);
        tok.atom_to_id.insert(CompactString::from("CC"), 6);
        tok.atom_to_id.insert(CompactString::from("CO"), 7);

        // Add merge rules
        tok.merges.insert((4, 4), 6); // C + C -> CC
        tok.merges.insert((4, 5), 7); // C + O -> CO

        let merges = tok.get_merges();
        assert_eq!(merges.len(), 2);

        // Merges should be sorted by merged token ID
        assert_eq!(
            merges[0],
            ("C".to_string(), "C".to_string(), "CC".to_string())
        );
        assert_eq!(
            merges[1],
            ("C".to_string(), "O".to_string(), "CO".to_string())
        );
    }

    #[test]
    fn test_special_token_ids() {
        let tok = SmilesTokenizer::new();
        assert_eq!(tok.pad_token_id(), 0);
        assert_eq!(tok.unk_token_id(), 1);
        assert_eq!(tok.bos_token_id(), 2);
        assert_eq!(tok.eos_token_id(), 3);
    }

    #[test]
    fn test_special_token_strings() {
        let tok = SmilesTokenizer::new();
        assert_eq!(tok.pad_token(), "<pad>");
        assert_eq!(tok.unk_token(), "<unk>");
        assert_eq!(tok.bos_token(), "<bos>");
        assert_eq!(tok.eos_token(), "<eos>");
    }

    #[test]
    fn test_encode_empty_smiles() {
        let tok = SmilesTokenizer::new();

        // Without special tokens
        let ids = tok.encode("", false);
        assert!(ids.is_empty());

        // With special tokens
        let ids = tok.encode("", true);
        assert_eq!(ids, vec![2, 3]); // BOS, EOS
    }

    #[test]
    fn test_encode_unknown_atoms() {
        let mut tok = SmilesTokenizer::new();

        // Add only C to vocabulary
        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.atom_to_id.insert(CompactString::from("C"), 4);

        // Encoding "CO" should use UNK for O
        let ids = tok.encode("CO", false);
        assert_eq!(ids, vec![4, 1]); // C, UNK
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let mut tok = SmilesTokenizer::new();

        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);

        let ids = tok.encode("CO", true);
        assert_eq!(ids, vec![2, 4, 5, 3]); // BOS, C, O, EOS
    }

    #[test]
    fn test_get_vocabulary() {
        let mut tok = SmilesTokenizer::new();

        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);

        let vocab = tok.get_vocabulary();
        assert_eq!(vocab.len(), 6); // 4 special + 2 atoms

        // Check special tokens
        assert_eq!(vocab[0], ("<pad>".to_string(), 0));
        assert_eq!(vocab[1], ("<unk>".to_string(), 1));
        assert_eq!(vocab[2], ("<bos>".to_string(), 2));
        assert_eq!(vocab[3], ("<eos>".to_string(), 3));

        // Check atoms
        assert_eq!(vocab[4], ("C".to_string(), 4));
        assert_eq!(vocab[5], ("O".to_string(), 5));
    }

    #[test]
    fn test_vocab_size_and_base_vocab() {
        let mut tok = SmilesTokenizer::new();

        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.id_to_atom.push(CompactString::from("CC")); // ID 6, merged
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);
        tok.atom_to_id.insert(CompactString::from("CC"), 6);
        tok.merges.insert((4, 4), 6);

        assert_eq!(tok.vocab_size(), 7); // 4 special + 2 base + 1 merged
        assert_eq!(tok.base_vocab_size(), 6); // vocab_size - num_merges
        assert_eq!(tok.num_merges(), 1);
    }

    #[test]
    fn test_default_trait() {
        let tok = SmilesTokenizer::default();
        assert_eq!(tok.id_to_atom.len(), 4); // Special tokens
        assert!(!tok.is_trained());
    }
}
