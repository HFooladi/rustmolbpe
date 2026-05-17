//! HuggingFace `tokenizers` JSON interop.
//!
//! Exports a tokenizer to / imports it from the `tokenizer.json` format used by
//! the HuggingFace `tokenizers` library, so a `rustmolbpe`-trained vocabulary
//! can be consumed by `transformers.PreTrainedTokenizerFast`.
//!
//! HuggingFace's fast `BPE` model only ever merges single Unicode characters
//! within one pre-token, so the granularity ladder maps as follows:
//!
//! | Class              | HF model              | pre-tokenizer        |
//! |--------------------|-----------------------|----------------------|
//! | `CharTokenizer`    | `BPE` (no merges)     | none                 |
//! | `CharBPETokenizer` | `BPE`                 | none                 |
//! | `AtomTokenizer`    | `WordLevel`           | `Split` (atom regex) |
//! | `SmilesTokenizer`  | not representable (atom-level BPE)           |
//!
//! `SmilesTokenizer` export is rejected upstream in [`crate::core`]; importing
//! any HuggingFace file into it fails the granularity checks below.

use std::collections::HashMap as StdHashMap;

use ahash::AHashMap;
use compact_str::CompactString;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::Deserialize;
use serde_json::json;

use crate::constants::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SMILES_ATOM_PATTERN, UNK_TOKEN};
use crate::core::TokenizerCore;
use crate::pretokenizer::PreTokenizerKind;

/// Serialize a tokenizer core to a HuggingFace `tokenizer.json` string.
///
/// Callers must reject `SmilesTokenizer` (atom-level BPE) before this point;
/// for an atom-level core any merges are silently absent from the `WordLevel`
/// model, which is only correct when there are none.
pub(crate) fn to_hf_json(core: &TokenizerCore) -> PyResult<String> {
    // Vocab: token -> id. A BTreeMap keeps the JSON object order deterministic.
    let vocab: std::collections::BTreeMap<&str, u32> = core
        .id_to_atom
        .iter()
        .enumerate()
        .map(|(id, tok)| (tok.as_str(), id as u32))
        .collect();

    // The 4 special tokens, also present in the model vocab at ids 0-3.
    let added_tokens: Vec<_> = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        .iter()
        .enumerate()
        .map(|(id, content)| {
            json!({
                "id": id,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();

    let (pre_tokenizer, model) = match core.pretokenizer.kind() {
        PreTokenizerKind::Char => {
            // BPE model. Merges in learning-priority order as "left right".
            let merges: Vec<String> = crate::vocabulary::get_merges(&core.merges, &core.id_to_atom)
                .into_iter()
                .map(|(left, right, _merged)| format!("{left} {right}"))
                .collect();
            let model = json!({
                "type": "BPE",
                "dropout": null,
                "unk_token": UNK_TOKEN,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": vocab,
                "merges": merges,
            });
            (json!(null), model)
        }
        PreTokenizerKind::Atom => {
            // WordLevel maps each whole atom (incl. "[C@@H]") straight to an id.
            let pre = json!({
                "type": "Split",
                "pattern": { "Regex": SMILES_ATOM_PATTERN },
                "behavior": "Isolated",
                "invert": false,
            });
            let model = json!({
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": UNK_TOKEN,
            });
            (pre, model)
        }
        PreTokenizerKind::Byte => {
            // Byte-level export is a planned roadmap item; `core::save_huggingface`
            // already rejects `ByteBPETokenizer` before reaching this point.
            return Err(PyValueError::new_err(
                "Byte-level BPE cannot be exported to HuggingFace JSON yet.",
            ));
        }
    };

    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        // Fuse concatenates token strings, matching rustmolbpe's plain-concat decode.
        "decoder": { "type": "Fuse" },
        "model": model,
    });

    serde_json::to_string_pretty(&tokenizer)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize HuggingFace JSON: {e}")))
}

/// A subset of the HuggingFace `tokenizer.json` schema needed to rebuild a core.
#[derive(Deserialize)]
struct HfTokenizerFile {
    model: HfModel,
}

#[derive(Deserialize)]
struct HfModel {
    #[serde(rename = "type")]
    model_type: String,
    vocab: StdHashMap<String, u32>,
    #[serde(default)]
    merges: Vec<HfMerge>,
}

/// A merge rule, accepted as either a `["left", "right"]` array (newer
/// `tokenizers`) or a `"left right"` string (older `tokenizers`).
#[derive(Deserialize)]
#[serde(untagged)]
enum HfMerge {
    Pair([String; 2]),
    Joined(String),
}

impl HfMerge {
    fn parts(&self) -> PyResult<(&str, &str)> {
        match self {
            HfMerge::Pair([left, right]) => Ok((left, right)),
            HfMerge::Joined(s) => {
                let mut it = s.splitn(2, ' ');
                match (it.next(), it.next()) {
                    (Some(left), Some(right)) => Ok((left, right)),
                    _ => Err(PyValueError::new_err(format!(
                        "Malformed merge rule in HuggingFace JSON: '{s}'"
                    ))),
                }
            }
        }
    }
}

/// Restore a tokenizer core from a HuggingFace `tokenizer.json` string.
///
/// `core` was constructed by the calling class, so its granularity and
/// merge-support fix which files are acceptable; mismatches raise `ValueError`,
/// mirroring the cross-class guard in [`crate::serialization`].
pub(crate) fn restore_from_hf_json(core: &mut TokenizerCore, json: &str) -> PyResult<()> {
    let file: HfTokenizerFile = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("Invalid HuggingFace JSON: {e}")))?;
    let model = file.model;

    // The model type fixes the granularity; validate it matches this class.
    let core_kind = core.pretokenizer.kind();
    match model.model_type.as_str() {
        "BPE" => {
            if core_kind != PreTokenizerKind::Char {
                return Err(PyValueError::new_err(
                    "HuggingFace granularity mismatch: file is a 'BPE' (character-level) \
                     tokenizer and can only load into CharTokenizer or CharBPETokenizer.",
                ));
            }
            if !model.merges.is_empty() && !core.allow_merges {
                return Err(PyValueError::new_err(
                    "This HuggingFace file has BPE merges; load it into CharBPETokenizer, \
                     not CharTokenizer.",
                ));
            }
        }
        "WordLevel" => {
            if core_kind != PreTokenizerKind::Atom {
                return Err(PyValueError::new_err(
                    "HuggingFace granularity mismatch: file is a 'WordLevel' (atom-level) \
                     tokenizer and can only load into AtomTokenizer.",
                ));
            }
            if core.allow_merges {
                return Err(PyValueError::new_err(
                    "WordLevel HuggingFace files carry no merges and cannot load into \
                     SmilesTokenizer; load into AtomTokenizer instead.",
                ));
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "Unsupported HuggingFace model type '{other}'. Only 'BPE' (character-level) \
                 and 'WordLevel' (atom-level) tokenizers can be imported."
            )));
        }
    }

    // Rebuild id_to_atom; vocab ids must be contiguous starting from 0.
    let vocab_size = model.vocab.len();
    let mut slots: Vec<Option<CompactString>> = vec![None; vocab_size];
    for (token, &id) in &model.vocab {
        let idx = id as usize;
        if idx >= vocab_size {
            return Err(PyValueError::new_err(format!(
                "HuggingFace vocab id {id} is out of range (vocab size {vocab_size}); \
                 ids must be contiguous from 0."
            )));
        }
        if slots[idx].is_some() {
            return Err(PyValueError::new_err(format!(
                "Duplicate HuggingFace vocab id {id}."
            )));
        }
        slots[idx] = Some(CompactString::from(token.as_str()));
    }
    let id_to_atom: Vec<CompactString> = slots
        .into_iter()
        .enumerate()
        .map(|(idx, slot)| {
            slot.ok_or_else(|| {
                PyValueError::new_err(format!("HuggingFace vocab is missing id {idx}."))
            })
        })
        .collect::<PyResult<_>>()?;

    // Special tokens must occupy ids 0-3 with the exact rustmolbpe strings.
    let expected = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN];
    if id_to_atom.len() < expected.len() {
        return Err(PyValueError::new_err(
            "HuggingFace vocab is too small to contain the 4 special tokens.",
        ));
    }
    for (idx, tok) in expected.iter().enumerate() {
        if id_to_atom[idx].as_str() != *tok {
            return Err(PyValueError::new_err(format!(
                "HuggingFace vocab id {idx} is '{}', expected special token '{tok}'. \
                 rustmolbpe requires <pad>/<unk>/<bos>/<eos> at ids 0-3.",
                id_to_atom[idx]
            )));
        }
    }

    let atom_to_id: AHashMap<CompactString, u32> = id_to_atom
        .iter()
        .enumerate()
        .map(|(id, tok)| (tok.clone(), id as u32))
        .collect();

    // Rebuild merges: (left_id, right_id) -> id of the concatenated token.
    let mut merges = StdHashMap::with_capacity(model.merges.len());
    for merge in &model.merges {
        let (left, right) = merge.parts()?;
        let left_id = lookup(&atom_to_id, left)?;
        let right_id = lookup(&atom_to_id, right)?;
        let merged = format!("{left}{right}");
        let merged_id = lookup(&atom_to_id, &merged)?;
        merges.insert((left_id, right_id), merged_id);
    }

    core.id_to_atom = id_to_atom;
    core.atom_to_id = atom_to_id;
    core.merges = merges;
    Ok(())
}

/// Look up a token id, erroring if the merge rule references an unknown token.
fn lookup(atom_to_id: &AHashMap<CompactString, u32>, token: &str) -> PyResult<u32> {
    atom_to_id
        .get(&CompactString::from(token))
        .copied()
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "HuggingFace merge references token '{token}' that is not in the vocab."
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small character-level BPE core: chars C, O and one merge CC.
    fn char_bpe_core() -> TokenizerCore {
        let mut core = TokenizerCore::new(PreTokenizerKind::Char, true);
        for tok in ["C", "O", "CC"] {
            let id = core.id_to_atom.len() as u32;
            core.id_to_atom.push(CompactString::from(tok));
            core.atom_to_id.insert(CompactString::from(tok), id);
        }
        core.merges.insert((4, 4), 6); // C + C -> CC
        core
    }

    /// Build a small atom-level core: atoms C, O, Cl, no merges.
    fn atom_core() -> TokenizerCore {
        let mut core = TokenizerCore::new(PreTokenizerKind::Atom, false);
        for tok in ["C", "O", "Cl"] {
            let id = core.id_to_atom.len() as u32;
            core.id_to_atom.push(CompactString::from(tok));
            core.atom_to_id.insert(CompactString::from(tok), id);
        }
        core
    }

    #[test]
    fn test_char_bpe_roundtrip() {
        let core = char_bpe_core();
        let json = to_hf_json(&core).unwrap();
        assert!(json.contains("\"type\": \"BPE\""));
        assert!(json.contains("\"C C\"")); // the merge rule

        let mut restored = TokenizerCore::new(PreTokenizerKind::Char, true);
        restore_from_hf_json(&mut restored, &json).unwrap();
        assert_eq!(restored.id_to_atom, core.id_to_atom);
        assert_eq!(restored.merges, core.merges);
    }

    #[test]
    fn test_atom_wordlevel_roundtrip() {
        let core = atom_core();
        let json = to_hf_json(&core).unwrap();
        assert!(json.contains("\"type\": \"WordLevel\""));
        assert!(json.contains("\"type\": \"Split\""));

        let mut restored = TokenizerCore::new(PreTokenizerKind::Atom, false);
        restore_from_hf_json(&mut restored, &json).unwrap();
        assert_eq!(restored.id_to_atom, core.id_to_atom);
        assert!(restored.merges.is_empty());
    }

    #[test]
    fn test_cross_granularity_rejected() {
        pyo3::Python::initialize();
        let json = to_hf_json(&char_bpe_core()).unwrap();
        // A BPE (char) file must not load into an atom-level core.
        let mut atom = TokenizerCore::new(PreTokenizerKind::Atom, false);
        let err = restore_from_hf_json(&mut atom, &json).unwrap_err();
        assert!(err.to_string().contains("granularity mismatch"));
    }

    #[test]
    fn test_bpe_with_merges_rejected_by_no_merge_core() {
        pyo3::Python::initialize();
        let json = to_hf_json(&char_bpe_core()).unwrap();
        // CharTokenizer (allow_merges = false) must reject a file with merges.
        let mut char_no_merge = TokenizerCore::new(PreTokenizerKind::Char, false);
        let err = restore_from_hf_json(&mut char_no_merge, &json).unwrap_err();
        assert!(err.to_string().contains("CharBPETokenizer"));
    }

    #[test]
    fn test_wordlevel_rejected_by_smiles_core() {
        pyo3::Python::initialize();
        let json = to_hf_json(&atom_core()).unwrap();
        // SmilesTokenizer is atom-level + allow_merges; a WordLevel file is invalid.
        let mut smiles = TokenizerCore::new(PreTokenizerKind::Atom, true);
        let err = restore_from_hf_json(&mut smiles, &json).unwrap_err();
        assert!(err.to_string().contains("SmilesTokenizer"));
    }

    #[test]
    fn test_invalid_json_rejected() {
        let mut core = TokenizerCore::new(PreTokenizerKind::Char, true);
        assert!(restore_from_hf_json(&mut core, "not json").is_err());
    }

    #[test]
    fn test_missing_special_tokens_rejected() {
        pyo3::Python::initialize();
        // A BPE model whose id 0 is not <pad>.
        let json = r#"{"model":{"type":"BPE","vocab":{"C":0,"O":1},"merges":[]}}"#;
        let mut core = TokenizerCore::new(PreTokenizerKind::Char, true);
        let err = restore_from_hf_json(&mut core, json).unwrap_err();
        assert!(err.to_string().contains("special token"));
    }
}
