//! Python-visible tokenizer classes.
//!
//! All four tokenizers share the same API and the same [`TokenizerCore`]
//! implementation; they differ only in pre-tokenization granularity and whether
//! BPE merges are learned. To avoid repeating ~25 identical `#[pymethods]` four
//! times, the [`define_tokenizer!`] declarative macro emits the `#[pyclass]`
//! struct and its method block; each method is a one-line delegation to the
//! core. A declarative macro expands before the `#[pymethods]` proc-macro runs,
//! so PyO3 sees an ordinary `impl` block.
//!
//! | Class              | Granularity | Learns merges |
//! |--------------------|-------------|---------------|
//! | `CharTokenizer`    | character   | no            |
//! | `AtomTokenizer`    | atom        | no            |
//! | `CharBPETokenizer` | character   | yes           |
//! | `SmilesTokenizer`  | atom        | yes (SPE)     |
//! | `ByteBPETokenizer` | byte        | yes           |

use std::collections::HashMap as StdHashMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple, PyType};

use crate::core::TokenizerCore;
use crate::pretokenizer::PreTokenizerKind;
use crate::serialization;

/// Generate a Python-visible tokenizer class wrapping a [`TokenizerCore`].
macro_rules! define_tokenizer {
    ($Name:ident, $kind:expr, $allow_merges:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(module = "rustmolbpe")]
        pub struct $Name {
            core: TokenizerCore,
        }

        impl Default for $Name {
            fn default() -> Self {
                Self::new()
            }
        }

        #[pymethods]
        impl $Name {
            /// Create a new tokenizer with special tokens initialized.
            #[new]
            pub fn new() -> Self {
                Self {
                    core: TokenizerCore::new($kind, $allow_merges),
                }
            }

            // --- Special tokens ---------------------------------------------

            /// The PAD token ID (always 0).
            #[getter]
            pub fn pad_token_id(&self) -> u32 {
                self.core.pad_token_id()
            }

            /// The UNK token ID (always 1).
            #[getter]
            pub fn unk_token_id(&self) -> u32 {
                self.core.unk_token_id()
            }

            /// The BOS token ID (always 2).
            #[getter]
            pub fn bos_token_id(&self) -> u32 {
                self.core.bos_token_id()
            }

            /// The EOS token ID (always 3).
            #[getter]
            pub fn eos_token_id(&self) -> u32 {
                self.core.eos_token_id()
            }

            /// The PAD token string.
            #[getter]
            pub fn pad_token(&self) -> &'static str {
                self.core.pad_token()
            }

            /// The UNK token string.
            #[getter]
            pub fn unk_token(&self) -> &'static str {
                self.core.unk_token()
            }

            /// The BOS token string.
            #[getter]
            pub fn bos_token(&self) -> &'static str {
                self.core.bos_token()
            }

            /// The EOS token string.
            #[getter]
            pub fn eos_token(&self) -> &'static str {
                self.core.eos_token()
            }

            // --- Training ---------------------------------------------------

            /// Train from a streaming iterator of SMILES strings.
            ///
            /// For BPE tokenizers this learns up to `vocab_size - base` merges.
            /// For non-BPE tokenizers (`CharTokenizer`, `AtomTokenizer`) it only
            /// builds the base vocabulary and `vocab_size` is ignored.
            #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, min_frequency=2))]
            #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, min_frequency=2)")]
            pub fn train_from_iterator(
                &mut self,
                py: Python<'_>,
                iterator: &Bound<'_, PyAny>,
                vocab_size: u32,
                buffer_size: usize,
                min_frequency: u32,
            ) -> PyResult<()> {
                self.core
                    .train_from_iterator(py, iterator, vocab_size, buffer_size, min_frequency)
            }

            // --- Vocabulary file I/O ----------------------------------------

            /// Load a SMILESPE-format vocabulary file.
            ///
            /// Raises `NotImplementedError` for non-BPE tokenizers.
            #[pyo3(signature = (path))]
            pub fn load_vocabulary(&mut self, path: &str) -> PyResult<()> {
                self.core.load_vocabulary(path)
            }

            /// Save the vocabulary to a SMILESPE-format file.
            ///
            /// Raises `NotImplementedError` for non-BPE tokenizers.
            #[pyo3(signature = (path))]
            pub fn save_vocabulary(&self, path: &str) -> PyResult<()> {
                self.core.save_vocabulary(path)
            }

            // --- HuggingFace JSON interop -----------------------------------

            /// Export to a HuggingFace `tokenizers` JSON file.
            ///
            /// The emitted `tokenizer.json` is loadable by
            /// `transformers.PreTrainedTokenizerFast`. Raises
            /// `NotImplementedError` for `SmilesTokenizer` (atom-level BPE
            /// cannot be expressed as a stock HuggingFace fast tokenizer).
            #[pyo3(signature = (path))]
            pub fn save_huggingface(&self, path: &str) -> PyResult<()> {
                self.core.save_huggingface(path)
            }

            /// Load a tokenizer from a HuggingFace `tokenizers` JSON file.
            ///
            /// Raises `ValueError` if the file's granularity / merge profile
            /// does not match this tokenizer class.
            #[classmethod]
            #[pyo3(signature = (path))]
            pub fn from_huggingface(
                _cls: &Bound<'_, PyType>,
                path: &str,
            ) -> PyResult<Self> {
                let mut core = TokenizerCore::new($kind, $allow_merges);
                core.restore_from_huggingface(path)?;
                Ok(Self { core })
            }

            // --- Vocabulary queries -----------------------------------------

            /// The vocabulary size (special tokens + base units + merges).
            #[getter]
            pub fn vocab_size(&self) -> u32 {
                self.core.vocab_size()
            }

            /// The number of base units (vocabulary minus learned merges).
            #[getter]
            pub fn base_vocab_size(&self) -> u32 {
                self.core.base_vocab_size()
            }

            /// The number of learned BPE merges (always 0 for non-BPE tokenizers).
            #[getter]
            pub fn num_merges(&self) -> u32 {
                self.core.num_merges()
            }

            /// Whether the tokenizer has learned BPE merges.
            ///
            /// Always `False` for non-BPE tokenizers; use `has_vocabulary()` to
            /// check whether a base vocabulary has been built.
            pub fn is_trained(&self) -> bool {
                self.core.is_trained()
            }

            /// Whether a base vocabulary has been built beyond the special tokens.
            pub fn has_vocabulary(&self) -> bool {
                self.core.has_vocabulary()
            }

            /// The learned merges as (left, right, merged) string tuples.
            pub fn get_merges(&self) -> Vec<(String, String, String)> {
                self.core.get_merges()
            }

            /// The vocabulary as a list of (token_string, token_id) tuples.
            pub fn get_vocabulary(&self) -> Vec<(String, u32)> {
                self.core.get_vocabulary()
            }

            /// The token string for a token ID.
            pub fn id_to_token(&self, id: u32) -> PyResult<String> {
                self.core.id_to_token(id)
            }

            /// The token ID for a token string.
            pub fn token_to_id(&self, token: &str) -> PyResult<u32> {
                self.core.token_to_id(token)
            }

            // --- Encoding / decoding ----------------------------------------

            /// Encode a SMILES string into token IDs using greedy longest-match.
            #[pyo3(signature = (smiles, add_special_tokens=false))]
            pub fn encode(&self, smiles: &str, add_special_tokens: bool) -> Vec<u32> {
                self.core.encode(smiles, add_special_tokens)
            }

            /// Decode token IDs back into a SMILES string.
            pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
                self.core.decode(&ids)
            }

            /// Encode multiple SMILES strings in parallel.
            #[pyo3(signature = (smiles_list, add_special_tokens=false))]
            #[pyo3(text_signature = "(self, smiles_list, add_special_tokens=False)")]
            pub fn batch_encode(
                &self,
                py: Python<'_>,
                smiles_list: Vec<String>,
                add_special_tokens: bool,
            ) -> Vec<Vec<u32>> {
                self.core.batch_encode(py, &smiles_list, add_special_tokens)
            }

            /// Decode multiple token ID sequences in parallel.
            #[pyo3(signature = (ids_list))]
            pub fn batch_decode(
                &self,
                py: Python<'_>,
                ids_list: Vec<Vec<u32>>,
            ) -> PyResult<Vec<String>> {
                self.core.batch_decode(py, &ids_list)
            }

            /// Pad a batch of token sequences to the same length.
            #[pyo3(signature = (sequences, max_length=None, padding="right", truncation=false, return_attention_mask=true))]
            pub fn pad(
                &self,
                sequences: Vec<Vec<u32>>,
                max_length: Option<usize>,
                padding: &str,
                truncation: bool,
                return_attention_mask: bool,
            ) -> StdHashMap<String, Vec<Vec<u32>>> {
                self.core
                    .pad(sequences, max_length, padding, truncation, return_attention_mask)
            }

            /// Encode and pad a batch of SMILES strings in one call.
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
            ) -> StdHashMap<String, Vec<Vec<u32>>> {
                self.core.encode_batch_padded(
                    py,
                    &smiles_list,
                    max_length,
                    padding,
                    truncation,
                    add_special_tokens,
                    return_attention_mask,
                )
            }

            /// HuggingFace-style callable interface.
            ///
            /// Accepts a single SMILES string or a list of SMILES strings and
            /// returns a dict with `input_ids` and optionally `attention_mask`.
            #[allow(clippy::too_many_arguments)]
            #[pyo3(signature = (text, padding=false, truncation=false, max_length=None, add_special_tokens=false, return_attention_mask=true))]
            pub fn __call__<'py>(
                &self,
                py: Python<'py>,
                text: &Bound<'py, PyAny>,
                padding: bool,
                truncation: bool,
                max_length: Option<usize>,
                add_special_tokens: bool,
                return_attention_mask: bool,
            ) -> PyResult<Py<PyDict>> {
                self.core.call(
                    py,
                    text,
                    padding,
                    truncation,
                    max_length,
                    add_special_tokens,
                    return_attention_mask,
                )
            }

            // --- Pickle support ---------------------------------------------

            /// Pickle support: return (cls, args, state) for serialization.
            pub fn __reduce__<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<(
                Bound<'py, PyType>,
                Bound<'py, PyTuple>,
                Bound<'py, PyDict>,
            )> {
                serialization::create_pickle_state(py, py.get_type::<Self>(), &self.core)
            }

            /// Pickle support: restore state from serialization.
            pub fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
                serialization::restore_into_core(&mut self.core, &state)
            }
        }
    };
}

define_tokenizer!(
    CharTokenizer,
    PreTokenizerKind::Char,
    false,
    "Character-level SMILES tokenizer.\n\n\
     Splits a SMILES string into individual Unicode characters with no BPE\n\
     merges. `vocab_size` passed to `train_from_iterator` is ignored and\n\
     `num_merges` is always 0; vocabulary file I/O is not supported."
);

define_tokenizer!(
    AtomTokenizer,
    PreTokenizerKind::Atom,
    false,
    "Atom-level SMILES tokenizer.\n\n\
     Splits a SMILES string into atoms and structural tokens using the SMILES\n\
     atom regex (multi-character atoms such as `Br`, `Cl`, `[C@@H]` stay\n\
     together) with no BPE merges. `vocab_size` is ignored, `num_merges` is\n\
     always 0, and vocabulary file I/O is not supported."
);

define_tokenizer!(
    CharBPETokenizer,
    PreTokenizerKind::Char,
    true,
    "Character-level BPE tokenizer.\n\n\
     Pre-tokenizes a SMILES string into characters, then applies BPE merges\n\
     learned by frequency during training."
);

define_tokenizer!(
    SmilesTokenizer,
    PreTokenizerKind::Atom,
    true,
    "Atom-level BPE tokenizer (SMILES Pair Encoding, \"SPE\").\n\n\
     Pre-tokenizes a SMILES string into atoms, then applies BPE merges learned\n\
     by frequency during training. This is the original rustmolbpe tokenizer."
);

define_tokenizer!(
    ByteBPETokenizer,
    PreTokenizerKind::Byte,
    true,
    "Byte-level BPE tokenizer.\n\n\
     Pre-tokenizes a SMILES string into raw UTF-8 bytes, then applies BPE\n\
     merges learned by frequency during training. The base alphabet is always\n\
     the 256 byte values, so every input is representable and no `<unk>` token\n\
     is ever produced. SMILESPE and HuggingFace file I/O are not supported;\n\
     use pickle to persist a byte-level tokenizer."
);
