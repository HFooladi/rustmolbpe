//! rustmolbpe - High-performance tokenizers for molecular SMILES strings.
//!
//! This crate provides a ladder of four SMILES tokenizers, all sharing one
//! granularity-agnostic core ([`core::TokenizerCore`]):
//!
//! - [`CharTokenizer`] - character-level splitting, no merges.
//! - [`AtomTokenizer`] - atom-level regex splitting, no merges.
//! - [`CharBPETokenizer`] - BPE trained on characters.
//! - [`SmilesTokenizer`] - BPE trained on atoms (SMILES Pair Encoding, "SPE").
//!
//! Unlike byte-level BPE tokenizers, the atom-level tokenizers treat
//! multi-character atoms (like `Br`, `Cl`, `[C@@H]`) as single units.

mod bytelevel;
mod constants;
mod core;
mod encoding;
mod huggingface;
mod padding;
mod pretokenizer;
mod python;
mod serialization;
mod training;
mod utils;
mod vocabulary;
mod word;

use pyo3::prelude::*;

// Re-export public constants.
pub use constants::{
    Pair, BOS_TOKEN, EOS_TOKEN, NUM_SPECIAL_TOKENS, PAD_TOKEN, SMILES_ATOM_PATTERN, UNK_TOKEN,
};

// Re-export the Python-visible tokenizer classes and function.
pub use python::{
    AtomTokenizer, ByteBPETokenizer, CharBPETokenizer, CharTokenizer, SmilesTokenizer,
};
pub use utils::atomwise_tokenize_py;

/// rustmolbpe - tokenizers for molecular SMILES with Python bindings.
#[pymodule]
fn rustmolbpe(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<SmilesTokenizer>()?;
    m.add_class::<AtomTokenizer>()?;
    m.add_class::<CharTokenizer>()?;
    m.add_class::<CharBPETokenizer>()?;
    m.add_class::<ByteBPETokenizer>()?;
    m.add_function(wrap_pyfunction!(utils::atomwise_tokenize_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__all__",
        vec![
            "SmilesTokenizer",
            "AtomTokenizer",
            "CharTokenizer",
            "CharBPETokenizer",
            "ByteBPETokenizer",
            "atomwise_tokenize",
            "__version__",
        ],
    )?;
    Ok(())
}
