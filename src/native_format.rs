//! Lossless native tokenizer file (`save` / `from_file`).
//!
//! A JSON document holding everything needed to rebuild a tokenizer exactly:
//! the full ID-ordered vocabulary, the merges in priority order, and the
//! pre-tokenizer granularity. Unlike the SMILESPE vocabulary format (merge
//! rules only), a round trip preserves every token ID.
//!
//! ```json
//! {"format": "rustmolbpe", "version": 1, "pretokenizer": "char",
//!  "vocab": ["<pad>", "<unk>", "<bos>", "<eos>", "C", "CC"],
//!  "merges": [[4, 4, 5]]}
//! ```

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde::{Deserialize, Serialize};

use crate::constants::{Pair, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN};
use crate::core::TokenizerCore;
use crate::pretokenizer::PreTokenizerKind;

/// Value of the `format` field that identifies a rustmolbpe tokenizer file.
const FORMAT_NAME: &str = "rustmolbpe";
/// Current tokenizer file version.
const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct NativeFile {
    format: String,
    version: u32,
    pretokenizer: String,
    /// Token strings; the list index is the token ID.
    vocab: Vec<String>,
    /// `[left_id, right_id, merged_id]` in priority order.
    merges: Vec<[u32; 3]>,
}

/// Serialize a tokenizer core to a native JSON string.
pub(crate) fn to_native_json(core: &TokenizerCore) -> PyResult<String> {
    let file = NativeFile {
        format: FORMAT_NAME.to_string(),
        version: FORMAT_VERSION,
        pretokenizer: core.pretokenizer.kind().as_tag().to_string(),
        vocab: core.id_to_atom.iter().map(|tok| tok.to_string()).collect(),
        merges: core.merges.iter().map(|&((l, r), m)| [l, r, m]).collect(),
    };
    serde_json::to_string_pretty(&file)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize tokenizer file: {e}")))
}

/// Restore a tokenizer core from a native JSON string.
///
/// `core` was constructed by the calling class, which fixes the acceptable
/// granularity and whether merges are allowed. Everything is validated before
/// `core` is modified; any problem raises `ValueError`.
pub(crate) fn restore_from_native_json(core: &mut TokenizerCore, json: &str) -> PyResult<()> {
    // Parse to a generic `Value` first so `format` / `version` can be checked
    // before the full typed parse. Otherwise a HuggingFace `tokenizer.json` (a
    // different but structurally-JSON schema) or a future version-2 file with
    // a changed schema would fail with a cryptic serde error instead of the
    // dedicated messages below.
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("Invalid rustmolbpe tokenizer file: {e}")))?;

    match value.get("format") {
        Some(serde_json::Value::String(found)) if found == FORMAT_NAME => {}
        Some(serde_json::Value::String(found)) => {
            return Err(PyValueError::new_err(format!(
                "Not a rustmolbpe tokenizer file: format is \"{found}\", expected \
                 \"{FORMAT_NAME}\"."
            )));
        }
        Some(found) => {
            return Err(PyValueError::new_err(format!(
                "Not a rustmolbpe tokenizer file: format is {found}, expected \"{FORMAT_NAME}\"."
            )));
        }
        None => {
            return Err(PyValueError::new_err(format!(
                "Not a rustmolbpe tokenizer file: format is missing, expected \"{FORMAT_NAME}\"."
            )));
        }
    }

    match value.get("version") {
        Some(found) if found.as_u64() == Some(FORMAT_VERSION as u64) => {}
        Some(found) => {
            return Err(PyValueError::new_err(format!(
                "Unsupported rustmolbpe tokenizer file version {found}; this build reads \
                 version {FORMAT_VERSION}."
            )));
        }
        None => {
            return Err(PyValueError::new_err(format!(
                "Unsupported rustmolbpe tokenizer file version missing; this build reads \
                 version {FORMAT_VERSION}."
            )));
        }
    }

    let file: NativeFile = serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(format!("Invalid rustmolbpe tokenizer file: {e}")))?;

    let kind = PreTokenizerKind::from_tag(&file.pretokenizer).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown pretokenizer '{}' in tokenizer file.",
            file.pretokenizer
        ))
    })?;
    if kind != core.pretokenizer.kind() {
        return Err(PyValueError::new_err(format!(
            "Tokenizer file granularity mismatch: file is '{}' but this tokenizer is '{}'.",
            kind.as_tag(),
            core.pretokenizer.kind().as_tag()
        )));
    }
    if !file.merges.is_empty() && !core.allow_merges {
        return Err(PyValueError::new_err(
            "This tokenizer file has BPE merges and cannot load into a tokenizer without \
             merges (CharTokenizer / AtomTokenizer).",
        ));
    }

    let specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN];
    if file.vocab.len() < specials.len()
        || file
            .vocab
            .iter()
            .zip(specials)
            .any(|(token, special)| token.as_str() != special)
    {
        return Err(PyValueError::new_err(
            "Tokenizer file vocab must start with the special tokens <pad>, <unk>, <bos>, <eos>.",
        ));
    }

    // Byte-level tokenizers guarantee no `<unk>` is ever emitted because their
    // base vocab is always all 256 byte values, in byte order, at IDs 4-259
    // (see `TokenizerCore::train_from_iterator`). A file claiming to be
    // byte-level must carry that same fixed alphabet (or be untrained, i.e.
    // just the 4 specials) or that guarantee would silently break on load.
    if kind == PreTokenizerKind::Byte {
        let has_byte_alphabet = file.vocab.len() == specials.len()
            || (file.vocab.len() >= specials.len() + 256
                && (0u16..256).all(|b| {
                    file.vocab[specials.len() + b as usize]
                        == crate::bytelevel::byte_to_token(b as u8).as_str()
                }));
        if !has_byte_alphabet {
            return Err(PyValueError::new_err(
                "Byte-level tokenizer file must contain the 256 byte tokens at IDs 4-259 in \
                 byte order.",
            ));
        }
    }

    let vocab_size = file.vocab.len() as u32;
    let mut merges: Vec<(Pair, u32)> = Vec::with_capacity(file.merges.len());
    let mut seen_pairs: AHashSet<Pair> = AHashSet::new();
    for [left, right, merged] in file.merges {
        if left >= vocab_size || right >= vocab_size || merged >= vocab_size {
            return Err(PyValueError::new_err(format!(
                "Tokenizer file merge [{left}, {right}, {merged}] references an ID outside \
                 the vocab (size {vocab_size})."
            )));
        }
        let (l, r, m) = (
            &file.vocab[left as usize],
            &file.vocab[right as usize],
            &file.vocab[merged as usize],
        );
        if *m != format!("{l}{r}") {
            return Err(PyValueError::new_err(format!(
                "Tokenizer file merge [{left}, {right}, {merged}] is inconsistent: \
                 '{l}' + '{r}' != '{m}'."
            )));
        }
        if !seen_pairs.insert((left, right)) {
            return Err(PyValueError::new_err(format!(
                "Tokenizer file has a duplicate merge [{left}, {right}, {merged}]."
            )));
        }
        merges.push(((left, right), merged));
    }

    let id_to_atom: Vec<CompactString> = file
        .vocab
        .iter()
        .map(|tok| CompactString::from(tok.as_str()))
        .collect();
    let atom_to_id: AHashMap<CompactString, u32> = id_to_atom
        .iter()
        .enumerate()
        .map(|(id, tok)| (tok.clone(), id as u32))
        .collect();

    core.id_to_atom = id_to_atom;
    core.atom_to_id = atom_to_id;
    core.merges = merges;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Character-level BPE core whose merge priority differs from merged-ID
    /// order: "CC" (id 7) was learned before "CO" (id 6).
    fn char_bpe_core() -> TokenizerCore {
        let mut core = TokenizerCore::new(PreTokenizerKind::Char, true);
        for tok in ["C", "O", "CO", "CC"] {
            let id = core.id_to_atom.len() as u32;
            core.id_to_atom.push(CompactString::from(tok));
            core.atom_to_id.insert(CompactString::from(tok), id);
        }
        core.merges.push(((4, 4), 7)); // C + C -> CC
        core.merges.push(((4, 5), 6)); // C + O -> CO
        core
    }

    fn empty_char_bpe() -> TokenizerCore {
        TokenizerCore::new(PreTokenizerKind::Char, true)
    }

    #[test]
    fn test_roundtrip_preserves_ids_and_merge_order() {
        let core = char_bpe_core();
        let json = to_native_json(&core).unwrap();

        let mut restored = empty_char_bpe();
        restore_from_native_json(&mut restored, &json).unwrap();

        assert_eq!(restored.id_to_atom, core.id_to_atom);
        assert_eq!(restored.merges, vec![((4, 4), 7), ((4, 5), 6)]);
        assert_eq!(
            restored.atom_to_id.get(&CompactString::from("CC")),
            Some(&7)
        );
    }

    /// Serialize `char_bpe_core()`, apply `edit` to the JSON, restore it into
    /// `target`, and return the error message.
    fn restore_error(
        edit: impl FnOnce(&mut serde_json::Value),
        mut target: TokenizerCore,
    ) -> String {
        pyo3::Python::initialize();
        let json = to_native_json(&char_bpe_core()).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        edit(&mut value);
        restore_from_native_json(&mut target, &value.to_string())
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn test_invalid_files_rejected() {
        type Edit = Box<dyn FnOnce(&mut serde_json::Value)>;
        let cases: Vec<(Edit, TokenizerCore, &str)> = vec![
            (
                Box::new(|_: &mut serde_json::Value| {}),
                TokenizerCore::new(PreTokenizerKind::Atom, true),
                "granularity mismatch",
            ),
            (
                Box::new(|_: &mut serde_json::Value| {}),
                TokenizerCore::new(PreTokenizerKind::Char, false),
                "has BPE merges",
            ),
            (
                Box::new(|v: &mut serde_json::Value| v["format"] = "other".into()),
                empty_char_bpe(),
                "Not a rustmolbpe tokenizer file",
            ),
            (
                Box::new(|v: &mut serde_json::Value| v["version"] = 2.into()),
                empty_char_bpe(),
                "Unsupported rustmolbpe tokenizer file version",
            ),
            (
                Box::new(|v: &mut serde_json::Value| v["pretokenizer"] = "bogus".into()),
                empty_char_bpe(),
                "Unknown pretokenizer",
            ),
            (
                Box::new(|v: &mut serde_json::Value| v["vocab"][0] = "C".into()),
                empty_char_bpe(),
                "special tokens",
            ),
            (
                Box::new(|v: &mut serde_json::Value| v["merges"][0][2] = 99.into()),
                empty_char_bpe(),
                "outside the vocab",
            ),
            (
                Box::new(|v: &mut serde_json::Value| v["merges"][0][2] = 5.into()),
                empty_char_bpe(),
                "inconsistent",
            ),
            (
                Box::new(|v: &mut serde_json::Value| {
                    let first = v["merges"][0].clone();
                    v["merges"].as_array_mut().unwrap().push(first);
                }),
                empty_char_bpe(),
                "duplicate merge",
            ),
        ];
        for (edit, target, expected) in cases {
            let message = restore_error(edit, target);
            assert!(
                message.contains(expected),
                "expected '{expected}' in '{message}'"
            );
        }
    }

    #[test]
    fn test_invalid_json_rejected() {
        pyo3::Python::initialize();
        let mut core = empty_char_bpe();
        let err = restore_from_native_json(&mut core, "not json").unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid rustmolbpe tokenizer file"));
    }

    #[test]
    fn test_missing_format_key_rejected_before_typed_parse() {
        // A structurally-valid JSON document missing "format" entirely (e.g. a
        // HuggingFace tokenizer.json, whose top-level "version" is "1.0", a
        // string, not our integer) must fail the format/version pre-check with
        // a clear message, not a cryptic serde type error.
        pyo3::Python::initialize();
        let mut core = empty_char_bpe();
        let err = restore_from_native_json(
            &mut core,
            r#"{"version": "1.0", "truncation": null, "model": {"type": "BPE"}}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("Not a rustmolbpe tokenizer file"),
            "got: {err}"
        );
    }

    #[test]
    fn test_byte_level_file_without_256_byte_tokens_rejected() {
        // A file claiming to be byte-level but whose vocab is just the 4
        // specials plus an ordinary readable token ("C") is not a valid
        // byte-level tokenizer file: it lacks the fixed 256-byte alphabet that
        // guarantees ByteBPETokenizer never emits `<unk>`.
        pyo3::Python::initialize();
        let mut byte_core = TokenizerCore::new(PreTokenizerKind::Byte, true);
        let id = byte_core.id_to_atom.len() as u32;
        byte_core.id_to_atom.push(CompactString::from("C"));
        byte_core.atom_to_id.insert(CompactString::from("C"), id);
        let json = to_native_json(&byte_core).unwrap();

        let mut target = TokenizerCore::new(PreTokenizerKind::Byte, true);
        let err = restore_from_native_json(&mut target, &json).unwrap_err();
        assert!(err.to_string().contains("256 byte tokens"), "got: {err}");
    }
}
