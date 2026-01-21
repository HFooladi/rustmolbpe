//! Serialization support (pickle) for the SMILES tokenizer.

use std::collections::HashMap as StdHashMap;

use ahash::AHashMap;
use compact_str::CompactString;
use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use crate::constants::{Pair, SMILES_ATOM_PATTERN};

/// Create pickle state from tokenizer data.
///
/// Returns (class, args, state) for __reduce__.
pub(crate) fn create_pickle_state<'py>(
    py: Python<'py>,
    cls: Bound<'py, PyType>,
    id_to_atom: &[CompactString],
    merges: &StdHashMap<Pair, u32>,
) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
    // Create state dict
    let state = PyDict::new(py);

    // Serialize id_to_atom as list of strings
    let id_to_atom_list: Vec<&str> = id_to_atom.iter().map(|s| s.as_str()).collect();
    state.set_item("id_to_atom", PyList::new(py, id_to_atom_list)?)?;

    // Serialize merges as list of ((left_id, right_id), merged_id) tuples
    let merges_list: Vec<((u32, u32), u32)> =
        merges.iter().map(|(&(l, r), &m)| ((l, r), m)).collect();
    state.set_item("merges", merges_list)?;

    // Version for future compatibility
    state.set_item("version", 1u32)?;

    // Return (cls, args, state) - pickle will call cls(*args).__setstate__(state)
    // Use empty tuple for args since SmilesTokenizer::new() takes no arguments
    let args = PyTuple::empty(py);
    Ok((cls, args, state))
}

/// Restore tokenizer state from pickle.
///
/// Returns (id_to_atom, atom_to_id, merges, compiled_pattern).
#[allow(clippy::type_complexity)]
pub(crate) fn restore_from_pickle_state(
    state: &Bound<'_, PyDict>,
) -> PyResult<(
    Vec<CompactString>,
    AHashMap<CompactString, u32>,
    StdHashMap<Pair, u32>,
    Regex,
)> {
    // Check version
    let version: u32 = state
        .get_item("version")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Missing 'version' in pickle state")
        })?
        .extract()?;

    if version != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported pickle version: {}. Expected version 1.",
            version
        )));
    }

    // Restore id_to_atom
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

    // Restore merges
    let merges_list: Vec<((u32, u32), u32)> = state
        .get_item("merges")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'merges' in pickle state"))?
        .extract()?;

    let mut merges = StdHashMap::with_capacity(merges_list.len());
    for ((left_id, right_id), merged_id) in merges_list {
        merges.insert((left_id, right_id), merged_id);
    }

    // Recreate compiled pattern (always the same constant pattern)
    let compiled_pattern = Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern");

    Ok((id_to_atom, atom_to_id, merges, compiled_pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_restore_pickle_state() {
        pyo3::Python::initialize();

        Python::attach(|py| {
            // Create test data
            let mut id_to_atom = Vec::new();
            id_to_atom.push(CompactString::from("<pad>"));
            id_to_atom.push(CompactString::from("C"));
            id_to_atom.push(CompactString::from("O"));
            id_to_atom.push(CompactString::from("CC"));

            let mut merges = StdHashMap::new();
            merges.insert((1, 1), 3); // C + C -> CC

            // We need a dummy class for testing - skip the actual class test
            // Just test that we can create the state dict properly
            let state = PyDict::new(py);
            let id_to_atom_list: Vec<&str> = id_to_atom.iter().map(|s| s.as_str()).collect();
            state
                .set_item("id_to_atom", PyList::new(py, id_to_atom_list).unwrap())
                .unwrap();

            let merges_list: Vec<((u32, u32), u32)> =
                merges.iter().map(|(&(l, r), &m)| ((l, r), m)).collect();
            state.set_item("merges", merges_list).unwrap();
            state.set_item("version", 1u32).unwrap();

            // Restore from state
            let (restored_id_to_atom, restored_atom_to_id, restored_merges, _pattern) =
                restore_from_pickle_state(&state).unwrap();

            assert_eq!(restored_id_to_atom.len(), 4);
            assert_eq!(restored_id_to_atom[0].as_str(), "<pad>");
            assert_eq!(restored_id_to_atom[1].as_str(), "C");
            assert_eq!(restored_atom_to_id.get(&CompactString::from("C")), Some(&1));
            assert_eq!(restored_merges.get(&(1, 1)), Some(&3));
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

            let result = restore_from_pickle_state(&state);
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Unsupported pickle version"));
        });
    }
}
