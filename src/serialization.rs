//! Serialization support (pickle) for the tokenizers.
//!
//! Pickle state is versioned:
//! - `version = 1` - legacy `SmilesTokenizer` pickles (pre-0.3.0). They have no
//!   `pretokenizer` key; they are always atom-level, so it defaults to `"atom"`.
//! - `version = 2` - current format. Carries a `pretokenizer` tag (`"atom"` or
//!   `"char"`) so a pickle can only be restored into a matching tokenizer class.

use std::collections::HashMap as StdHashMap;

use ahash::AHashMap;
use compact_str::CompactString;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use crate::core::TokenizerCore;
use crate::pretokenizer::PreTokenizerKind;

/// Current pickle state version.
const PICKLE_VERSION: u32 = 2;

/// Create pickle state from a tokenizer core.
///
/// Returns `(class, args, state)` for `__reduce__`.
pub(crate) fn create_pickle_state<'py>(
    py: Python<'py>,
    cls: Bound<'py, PyType>,
    core: &TokenizerCore,
) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
    let state = PyDict::new(py);

    // Serialize id_to_atom as a list of strings.
    let id_to_atom_list: Vec<&str> = core.id_to_atom.iter().map(|s| s.as_str()).collect();
    state.set_item("id_to_atom", PyList::new(py, id_to_atom_list)?)?;

    // Serialize merges as a list of ((left_id, right_id), merged_id) tuples.
    let merges_list: Vec<((u32, u32), u32)> = core
        .merges
        .iter()
        .map(|(&(l, r), &m)| ((l, r), m))
        .collect();
    state.set_item("merges", merges_list)?;

    // Pre-tokenizer granularity tag - guards against cross-class unpickling.
    state.set_item("pretokenizer", core.pretokenizer.kind().as_tag())?;

    state.set_item("version", PICKLE_VERSION)?;

    // pickle calls cls(*args).__setstate__(state); new() takes no arguments.
    let args = PyTuple::empty(py);
    Ok((cls, args, state))
}

/// Restore tokenizer state from pickle into an existing core.
///
/// `core.pretokenizer` was already set by `new()`; this function validates that
/// the pickled granularity matches and rejects cross-class unpickling.
pub(crate) fn restore_into_core(
    core: &mut TokenizerCore,
    state: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let version: u32 = state
        .get_item("version")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Missing 'version' in pickle state")
        })?
        .extract()?;

    if version != 1 && version != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported pickle version: {version}. Expected version 1 or 2."
        )));
    }

    // version 1 pickles predate the pretokenizer tag and are always atom-level.
    let pretokenizer_tag: String = match state.get_item("pretokenizer")? {
        Some(value) => value.extract()?,
        None => "atom".to_string(),
    };
    let pickled_kind = PreTokenizerKind::from_tag(&pretokenizer_tag).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown pretokenizer tag in pickle state: '{pretokenizer_tag}'"
        ))
    })?;
    if pickled_kind != core.pretokenizer.kind() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Pickle granularity mismatch: state is '{}' but this tokenizer is '{}'. \
             A pickle can only be restored into a matching tokenizer class.",
            pickled_kind.as_tag(),
            core.pretokenizer.kind().as_tag()
        )));
    }

    // Restore id_to_atom / atom_to_id.
    let id_to_atom_list: Vec<String> = state
        .get_item("id_to_atom")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Missing 'id_to_atom' in pickle state")
        })?
        .extract()?;

    let mut id_to_atom = Vec::with_capacity(id_to_atom_list.len());
    let mut atom_to_id = AHashMap::with_capacity(id_to_atom_list.len());
    for (id, atom) in id_to_atom_list.into_iter().enumerate() {
        let compact_atom = CompactString::from(atom);
        atom_to_id.insert(compact_atom.clone(), id as u32);
        id_to_atom.push(compact_atom);
    }

    // Restore merges.
    let merges_list: Vec<((u32, u32), u32)> = state
        .get_item("merges")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'merges' in pickle state"))?
        .extract()?;

    let mut merges = StdHashMap::with_capacity(merges_list.len());
    for ((left_id, right_id), merged_id) in merges_list {
        merges.insert((left_id, right_id), merged_id);
    }

    core.id_to_atom = id_to_atom;
    core.atom_to_id = atom_to_id;
    core.merges = merges;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pickle_roundtrip_atom() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut core = TokenizerCore::new(PreTokenizerKind::Atom, true);
            core.id_to_atom.push(CompactString::from("C"));
            core.atom_to_id.insert(CompactString::from("C"), 4);
            core.merges.insert((4, 4), 5);
            core.id_to_atom.push(CompactString::from("CC"));
            core.atom_to_id.insert(CompactString::from("CC"), 5);

            let cls = py.get_type::<PyDict>(); // any type works for the test
            let (_, _, state) = create_pickle_state(py, cls, &core).unwrap();

            let mut restored = TokenizerCore::new(PreTokenizerKind::Atom, true);
            restore_into_core(&mut restored, &state).unwrap();

            assert_eq!(restored.id_to_atom.len(), 6);
            assert_eq!(restored.merges.get(&(4, 4)), Some(&5));
            assert_eq!(restored.pretokenizer.kind(), PreTokenizerKind::Atom);
        });
    }

    #[test]
    fn test_pickle_roundtrip_char() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut core = TokenizerCore::new(PreTokenizerKind::Char, false);
            core.id_to_atom.push(CompactString::from("C"));
            core.atom_to_id.insert(CompactString::from("C"), 4);

            let cls = py.get_type::<PyDict>();
            let (_, _, state) = create_pickle_state(py, cls, &core).unwrap();

            let mut restored = TokenizerCore::new(PreTokenizerKind::Char, false);
            restore_into_core(&mut restored, &state).unwrap();
            assert_eq!(restored.id_to_atom.len(), 5);
        });
    }

    #[test]
    fn test_pickle_cross_class_rejected() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let char_core = TokenizerCore::new(PreTokenizerKind::Char, false);
            let cls = py.get_type::<PyDict>();
            let (_, _, state) = create_pickle_state(py, cls, &char_core).unwrap();

            // Restoring a char pickle into an atom tokenizer must fail.
            let mut atom_core = TokenizerCore::new(PreTokenizerKind::Atom, true);
            let result = restore_into_core(&mut atom_core, &state);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("granularity mismatch"));
        });
    }

    #[test]
    fn test_legacy_v1_pickle_defaults_to_atom() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            // A version-1 state has no 'pretokenizer' key.
            let state = PyDict::new(py);
            state
                .set_item("id_to_atom", PyList::new(py, ["<pad>", "C"]).unwrap())
                .unwrap();
            state
                .set_item("merges", Vec::<((u32, u32), u32)>::new())
                .unwrap();
            state.set_item("version", 1u32).unwrap();

            let mut atom_core = TokenizerCore::new(PreTokenizerKind::Atom, true);
            restore_into_core(&mut atom_core, &state).unwrap();
            assert_eq!(atom_core.id_to_atom.len(), 2);

            // The same legacy pickle must NOT load into a char tokenizer.
            let mut char_core = TokenizerCore::new(PreTokenizerKind::Char, false);
            assert!(restore_into_core(&mut char_core, &state).is_err());
        });
    }

    #[test]
    fn test_restore_invalid_version() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let state = PyDict::new(py);
            state.set_item("version", 99u32).unwrap();
            state.set_item("id_to_atom", PyList::empty(py)).unwrap();
            state
                .set_item("merges", Vec::<((u32, u32), u32)>::new())
                .unwrap();

            let mut core = TokenizerCore::new(PreTokenizerKind::Atom, true);
            let result = restore_into_core(&mut core, &state);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Unsupported pickle version"));
        });
    }
}
