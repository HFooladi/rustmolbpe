//! Granularity-agnostic tokenizer core shared by all four tokenizer classes.
//!
//! [`TokenizerCore`] holds the full BPE/encoding implementation. It is a plain
//! Rust struct (not a `#[pyclass]`); the four Python-visible tokenizers in
//! [`crate::python`] are thin wrappers that delegate to a `TokenizerCore`.
//!
//! The only configuration is:
//! - `pretokenizer` - atom-level or character-level splitting.
//! - `allow_merges` - whether `train_from_iterator` learns BPE merges. When
//!   `false` the tokenizer is a pure pre-tokenizer (character-level or
//!   atom-level), and vocabulary file I/O is disabled.

use std::collections::HashMap as StdHashMap;

use ahash::AHashMap;
use compact_str::CompactString;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::constants::Pair;
use crate::pretokenizer::{PreTokenizer, PreTokenizerKind};
use crate::word::Word;
use crate::{encoding, padding, training, vocabulary};

/// Shared tokenizer implementation backing every tokenizer class.
pub(crate) struct TokenizerCore {
    /// Maps pairs of token IDs to their merged token ID.
    pub merges: StdHashMap<Pair, u32>,
    /// Maps token strings to token IDs.
    pub atom_to_id: AHashMap<CompactString, u32>,
    /// Reverse mapping: token ID to token string.
    pub id_to_atom: Vec<CompactString>,
    /// Pre-tokenizer (atom-level or character-level).
    pub pretokenizer: PreTokenizer,
    /// Whether BPE merges are learned during training and vocab I/O is allowed.
    pub allow_merges: bool,
}

impl TokenizerCore {
    /// Create a new core with special tokens initialized.
    pub(crate) fn new(kind: PreTokenizerKind, allow_merges: bool) -> Self {
        let mut core = Self {
            merges: StdHashMap::new(),
            atom_to_id: AHashMap::new(),
            id_to_atom: Vec::new(),
            pretokenizer: PreTokenizer::new(kind),
            allow_merges,
        };
        vocabulary::init_special_tokens(&mut core.atom_to_id, &mut core.id_to_atom);
        core
    }

    // --- Special tokens -----------------------------------------------------

    pub(crate) fn pad_token_id(&self) -> u32 {
        0
    }
    pub(crate) fn unk_token_id(&self) -> u32 {
        1
    }
    pub(crate) fn bos_token_id(&self) -> u32 {
        2
    }
    pub(crate) fn eos_token_id(&self) -> u32 {
        3
    }
    pub(crate) fn pad_token(&self) -> &'static str {
        crate::constants::PAD_TOKEN
    }
    pub(crate) fn unk_token(&self) -> &'static str {
        crate::constants::UNK_TOKEN
    }
    pub(crate) fn bos_token(&self) -> &'static str {
        crate::constants::BOS_TOKEN
    }
    pub(crate) fn eos_token(&self) -> &'static str {
        crate::constants::EOS_TOKEN
    }

    // --- Vocabulary queries -------------------------------------------------

    pub(crate) fn vocab_size(&self) -> u32 {
        self.id_to_atom.len() as u32
    }

    pub(crate) fn base_vocab_size(&self) -> u32 {
        (self.id_to_atom.len() - self.merges.len()) as u32
    }

    pub(crate) fn num_merges(&self) -> u32 {
        self.merges.len() as u32
    }

    /// Whether the tokenizer has learned BPE merges. Always `false` for the
    /// no-merge (character-level / atom-level) tokenizers.
    pub(crate) fn is_trained(&self) -> bool {
        !self.merges.is_empty()
    }

    /// Whether a base vocabulary has been built (beyond the 4 special tokens).
    pub(crate) fn has_vocabulary(&self) -> bool {
        self.id_to_atom.len() > crate::constants::NUM_SPECIAL_TOKENS as usize
    }

    pub(crate) fn get_merges(&self) -> Vec<(String, String, String)> {
        vocabulary::get_merges(&self.merges, &self.id_to_atom)
    }

    pub(crate) fn get_vocabulary(&self) -> Vec<(String, u32)> {
        vocabulary::get_vocabulary(&self.id_to_atom)
    }

    pub(crate) fn id_to_token(&self, id: u32) -> PyResult<String> {
        vocabulary::id_to_token(id, &self.id_to_atom)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub(crate) fn token_to_id(&self, token: &str) -> PyResult<u32> {
        vocabulary::token_to_id(token, &self.atom_to_id)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    // --- Vocabulary file I/O ------------------------------------------------

    /// Load a SMILESPE-format vocabulary. Only valid for BPE tokenizers.
    pub(crate) fn load_vocabulary(&mut self, path: &str) -> PyResult<()> {
        self.ensure_merges_supported("load_vocabulary")?;
        vocabulary::load_vocabulary(
            path,
            &mut self.merges,
            &mut self.atom_to_id,
            &mut self.id_to_atom,
        )
        .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Save a SMILESPE-format vocabulary. Only valid for BPE tokenizers.
    pub(crate) fn save_vocabulary(&self, path: &str) -> PyResult<()> {
        self.ensure_merges_supported("save_vocabulary")?;
        vocabulary::save_vocabulary(path, &self.merges, &self.id_to_atom)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// The SMILESPE vocabulary format only stores merge rules, so a no-merge
    /// tokenizer cannot meaningfully use it. Reject the call explicitly rather
    /// than silently writing/reading an empty file.
    fn ensure_merges_supported(&self, method: &str) -> PyResult<()> {
        if self.allow_merges {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "{method} is not supported for non-BPE tokenizers \
                 (the SMILESPE vocabulary format only stores merge rules)"
            )))
        }
    }

    // --- Encoding / decoding ------------------------------------------------

    pub(crate) fn encode(&self, smiles: &str, add_special_tokens: bool) -> Vec<u32> {
        encoding::encode(
            smiles,
            add_special_tokens,
            &self.pretokenizer,
            &self.atom_to_id,
            self.bos_token_id(),
            self.eos_token_id(),
            self.unk_token_id(),
        )
    }

    pub(crate) fn decode(&self, ids: &[u32]) -> PyResult<String> {
        encoding::decode(ids, &self.id_to_atom).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub(crate) fn batch_encode(
        &self,
        py: Python<'_>,
        smiles_list: &[String],
        add_special_tokens: bool,
    ) -> Vec<Vec<u32>> {
        encoding::batch_encode(
            py,
            smiles_list,
            add_special_tokens,
            &self.pretokenizer,
            &self.atom_to_id,
            self.bos_token_id(),
            self.eos_token_id(),
            self.unk_token_id(),
        )
    }

    pub(crate) fn batch_decode(
        &self,
        py: Python<'_>,
        ids_list: &[Vec<u32>],
    ) -> PyResult<Vec<String>> {
        encoding::batch_decode(py, ids_list, &self.id_to_atom)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub(crate) fn pad(
        &self,
        sequences: Vec<Vec<u32>>,
        max_length: Option<usize>,
        padding: &str,
        truncation: bool,
        return_attention_mask: bool,
    ) -> StdHashMap<String, Vec<Vec<u32>>> {
        padding::pad(
            sequences,
            max_length,
            padding,
            truncation,
            return_attention_mask,
            self.pad_token_id(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_batch_padded(
        &self,
        py: Python<'_>,
        smiles_list: &[String],
        max_length: Option<usize>,
        padding: &str,
        truncation: bool,
        add_special_tokens: bool,
        return_attention_mask: bool,
    ) -> StdHashMap<String, Vec<Vec<u32>>> {
        let sequences = self.batch_encode(py, smiles_list, add_special_tokens);
        self.pad(
            sequences,
            max_length,
            padding,
            truncation,
            return_attention_mask,
        )
    }

    /// HuggingFace-style callable interface. Accepts a single SMILES string or
    /// a list of SMILES strings.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn call<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        padding: bool,
        truncation: bool,
        max_length: Option<usize>,
        add_special_tokens: bool,
        return_attention_mask: bool,
    ) -> PyResult<Py<PyDict>> {
        let is_single = text.extract::<String>().is_ok();

        let smiles_list: Vec<String> = if is_single {
            vec![text.extract::<String>()?]
        } else {
            text.extract::<Vec<String>>()?
        };

        let mut sequences = self.batch_encode(py, &smiles_list, add_special_tokens);

        // Apply truncation when not padding (pad() handles it otherwise).
        if truncation {
            if let Some(max_len) = max_length {
                for seq in &mut sequences {
                    seq.truncate(max_len);
                }
            }
        }

        let dict = PyDict::new(py);

        if padding {
            let padded = padding::pad(
                sequences,
                max_length,
                "right",
                truncation,
                return_attention_mask,
                self.pad_token_id(),
            );

            if is_single {
                dict.set_item("input_ids", &padded["input_ids"][0])?;
                if let Some(mask) = padded.get("attention_mask") {
                    dict.set_item("attention_mask", &mask[0])?;
                }
            } else {
                dict.set_item("input_ids", &padded["input_ids"])?;
                if let Some(mask) = padded.get("attention_mask") {
                    dict.set_item("attention_mask", mask)?;
                }
            }
        } else if is_single {
            dict.set_item("input_ids", &sequences[0])?;
            if return_attention_mask {
                let mask: Vec<u32> = vec![1; sequences[0].len()];
                dict.set_item("attention_mask", mask)?;
            }
        } else {
            dict.set_item("input_ids", &sequences)?;
            if return_attention_mask {
                let masks: Vec<Vec<u32>> = sequences.iter().map(|s| vec![1u32; s.len()]).collect();
                dict.set_item("attention_mask", masks)?;
            }
        }

        Ok(dict.into())
    }

    // --- Training -----------------------------------------------------------

    /// Train from a streaming iterator of SMILES strings.
    ///
    /// Builds the base vocabulary from observed pre-tokenization units, then
    /// (only when `allow_merges` is set) learns up to `vocab_size - base` BPE
    /// merges. For no-merge tokenizers this is purely a base-vocabulary pass.
    pub(crate) fn train_from_iterator(
        &mut self,
        py: Python<'_>,
        iterator: &Bound<'_, PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        min_frequency: u32,
    ) -> PyResult<()> {
        // Clear existing state and reinitialize special tokens.
        self.merges.clear();
        self.atom_to_id.clear();
        self.id_to_atom.clear();
        vocabulary::init_special_tokens(&mut self.atom_to_id, &mut self.id_to_atom);

        // Prepare Python iterator.
        let py_iter: Py<PyAny> = unsafe {
            Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        let mut atom_counts: AHashMap<CompactString, u64> = AHashMap::new();
        let mut smiles_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();

        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "Processing SMILES from iterator (buffer_size: {})",
            buffer_size
        );
        let mut total_smiles = 0u64;

        // Helper: refill buffer from the Python iterator.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            Python::attach(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if PyErr::occurred(py) {
                                return Err(PyErr::fetch(py));
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            })
        };

        let pretokenizer = self.pretokenizer.clone();
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_smiles += buf.len() as u64;

            // Pre-tokenize the buffer in parallel.
            let local: Vec<(AHashMap<CompactString, u64>, Vec<CompactString>)> = py.detach(|| {
                use rayon::prelude::*;
                buf.par_iter()
                    .map(|smi| {
                        let atoms = pretokenizer.split(smi);
                        let mut atom_map: AHashMap<CompactString, u64> = AHashMap::new();
                        for atom in &atoms {
                            *atom_map.entry(atom.clone()).or_default() += 1;
                        }
                        (atom_map, atoms)
                    })
                    .collect()
            });

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
            "Processed {} SMILES total, {} unique sequences, {} unique units",
            total_smiles,
            smiles_counts.len(),
            atom_counts.len()
        );

        // Build base vocabulary, sorted by frequency then alphabetically.
        let mut atoms_sorted: Vec<_> = atom_counts.into_iter().collect();
        atoms_sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for (atom, _count) in atoms_sorted {
            let id = self.id_to_atom.len() as u32;
            self.atom_to_id.insert(atom.clone(), id);
            self.id_to_atom.push(atom);
        }

        let base_vocab_size = self.id_to_atom.len() as u32;
        log::info!("Built base vocabulary with {} units", base_vocab_size);

        // No-merge tokenizers (character-level / atom-level) stop here.
        if !self.allow_merges {
            return Ok(());
        }

        if vocab_size <= base_vocab_size {
            log::info!(
                "vocab_size ({}) <= base vocab size ({}), no merges needed",
                vocab_size,
                base_vocab_size
            );
            return Ok(());
        }

        let num_merges = vocab_size - base_vocab_size;

        // Convert SMILES sequences to Words for BPE training.
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
}

// ============================================================================
// RUST TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN};

    fn atom_core() -> TokenizerCore {
        TokenizerCore::new(PreTokenizerKind::Atom, true)
    }

    fn char_core() -> TokenizerCore {
        TokenizerCore::new(PreTokenizerKind::Char, false)
    }

    #[test]
    fn test_core_new() {
        let core = atom_core();
        assert!(core.merges.is_empty());
        assert_eq!(core.atom_to_id.len(), 4);
        assert_eq!(core.id_to_atom.len(), 4);
        assert_eq!(core.id_to_atom[0].as_str(), PAD_TOKEN);
        assert_eq!(core.id_to_atom[1].as_str(), UNK_TOKEN);
        assert_eq!(core.id_to_atom[2].as_str(), BOS_TOKEN);
        assert_eq!(core.id_to_atom[3].as_str(), EOS_TOKEN);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut core = atom_core();
        core.id_to_atom.push(CompactString::from("C")); // ID 4
        core.id_to_atom.push(CompactString::from("O")); // ID 5
        core.atom_to_id.insert(CompactString::from("C"), 4);
        core.atom_to_id.insert(CompactString::from("O"), 5);

        let ids = core.encode("CCO", false);
        assert_eq!(ids, vec![4, 4, 5]);
        assert_eq!(core.decode(&ids).unwrap(), "CCO");
    }

    #[test]
    fn test_encode_with_merge() {
        let mut core = atom_core();
        core.id_to_atom.push(CompactString::from("C")); // ID 4
        core.id_to_atom.push(CompactString::from("O")); // ID 5
        core.id_to_atom.push(CompactString::from("CC")); // ID 6
        core.atom_to_id.insert(CompactString::from("C"), 4);
        core.atom_to_id.insert(CompactString::from("O"), 5);
        core.atom_to_id.insert(CompactString::from("CC"), 6);
        core.merges.insert((4, 4), 6);

        let ids = core.encode("CCO", false);
        assert_eq!(ids, vec![6, 5]);
        assert_eq!(core.decode(&ids).unwrap(), "CCO");
    }

    #[test]
    fn test_is_trained() {
        let mut core = atom_core();
        assert!(!core.is_trained());
        core.merges.insert((4, 5), 6);
        assert!(core.is_trained());
    }

    #[test]
    fn test_has_vocabulary() {
        let mut core = atom_core();
        assert!(!core.has_vocabulary());
        core.id_to_atom.push(CompactString::from("C"));
        assert!(core.has_vocabulary());
    }

    #[test]
    fn test_get_merges() {
        let mut core = atom_core();
        for (i, tok) in ["C", "O", "CC", "CO"].iter().enumerate() {
            core.id_to_atom.push(CompactString::from(*tok));
            core.atom_to_id
                .insert(CompactString::from(*tok), 4 + i as u32);
        }
        core.merges.insert((4, 4), 6);
        core.merges.insert((4, 5), 7);

        let merges = core.get_merges();
        assert_eq!(merges.len(), 2);
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
        let core = atom_core();
        assert_eq!(core.pad_token_id(), 0);
        assert_eq!(core.unk_token_id(), 1);
        assert_eq!(core.bos_token_id(), 2);
        assert_eq!(core.eos_token_id(), 3);
    }

    #[test]
    fn test_special_token_strings() {
        let core = atom_core();
        assert_eq!(core.pad_token(), "<pad>");
        assert_eq!(core.unk_token(), "<unk>");
        assert_eq!(core.bos_token(), "<bos>");
        assert_eq!(core.eos_token(), "<eos>");
    }

    #[test]
    fn test_encode_empty_smiles() {
        let core = atom_core();
        assert!(core.encode("", false).is_empty());
        assert_eq!(core.encode("", true), vec![2, 3]);
    }

    #[test]
    fn test_encode_unknown_atoms() {
        let mut core = atom_core();
        core.id_to_atom.push(CompactString::from("C")); // ID 4
        core.atom_to_id.insert(CompactString::from("C"), 4);
        assert_eq!(core.encode("CO", false), vec![4, 1]); // C, UNK
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let mut core = atom_core();
        core.id_to_atom.push(CompactString::from("C")); // ID 4
        core.id_to_atom.push(CompactString::from("O")); // ID 5
        core.atom_to_id.insert(CompactString::from("C"), 4);
        core.atom_to_id.insert(CompactString::from("O"), 5);
        assert_eq!(core.encode("CO", true), vec![2, 4, 5, 3]);
    }

    #[test]
    fn test_get_vocabulary() {
        let mut core = atom_core();
        core.id_to_atom.push(CompactString::from("C"));
        core.id_to_atom.push(CompactString::from("O"));
        core.atom_to_id.insert(CompactString::from("C"), 4);
        core.atom_to_id.insert(CompactString::from("O"), 5);

        let vocab = core.get_vocabulary();
        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab[0], ("<pad>".to_string(), 0));
        assert_eq!(vocab[4], ("C".to_string(), 4));
        assert_eq!(vocab[5], ("O".to_string(), 5));
    }

    #[test]
    fn test_vocab_size_and_base_vocab() {
        let mut core = atom_core();
        core.id_to_atom.push(CompactString::from("C"));
        core.id_to_atom.push(CompactString::from("O"));
        core.id_to_atom.push(CompactString::from("CC"));
        core.atom_to_id.insert(CompactString::from("C"), 4);
        core.atom_to_id.insert(CompactString::from("O"), 5);
        core.atom_to_id.insert(CompactString::from("CC"), 6);
        core.merges.insert((4, 4), 6);

        assert_eq!(core.vocab_size(), 7);
        assert_eq!(core.base_vocab_size(), 6);
        assert_eq!(core.num_merges(), 1);
    }

    #[test]
    fn test_char_core_encode_splits_characters() {
        let mut core = char_core();
        // Single-character vocabulary entries for C, l, O.
        for (i, tok) in ["C", "l", "O"].iter().enumerate() {
            core.id_to_atom.push(CompactString::from(*tok));
            core.atom_to_id
                .insert(CompactString::from(*tok), 4 + i as u32);
        }
        // "Cl" must split into two character tokens, not one chlorine atom.
        let ids = core.encode("Cl", false);
        assert_eq!(ids, vec![4, 5]);
        assert_eq!(core.decode(&ids).unwrap(), "Cl");
    }

    #[test]
    fn test_no_merge_core_rejects_vocab_io() {
        let mut core = char_core();
        assert!(core
            .save_vocabulary("/tmp/should_not_be_written.txt")
            .is_err());
        assert!(core.load_vocabulary("/tmp/nonexistent.txt").is_err());
    }
}
