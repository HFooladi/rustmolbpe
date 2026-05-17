//! Byte-level pre-tokenization helpers.
//!
//! Byte-level BPE pre-tokenizes on raw UTF-8 *bytes*, giving a fixed 256-symbol
//! base alphabet and a lossless round-trip: every input is representable, so no
//! `<unk>` token is ever produced.
//!
//! Tokens are stored as UTF-8 [`CompactString`]s everywhere in this crate, but
//! a raw byte `0x80`-`0xFF` is not valid UTF-8 on its own. We therefore use
//! GPT-2's `bytes_to_unicode` bijection: each of the 256 byte values maps to a
//! distinct *printable* Unicode scalar, so the rest of the pipeline (BPE
//! training, greedy encoding) keeps operating on ordinary strings unchanged.
//! Only pre-tokenization ([`split_bytes`]) and decoding ([`decode_byte_string`])
//! are byte-level-specific.

use std::collections::HashMap;
use std::sync::LazyLock;

use compact_str::{CompactString, ToCompactString};

/// Maps each byte value to its representative Unicode scalar (GPT-2 scheme).
///
/// Bytes in the printable ranges `0x21..=0x7E`, `0xA1..=0xAC`, `0xAE..=0xFF`
/// map to the scalar with that codepoint; every other byte maps to a scalar in
/// `U+0100..` so the representative is always printable and never whitespace.
static BYTE_TO_CHAR: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut table = ['\0'; 256];
    let mut next_extra = 0u32;
    for byte in 0u16..256 {
        let byte = byte as u8;
        let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        table[byte as usize] = if printable {
            char::from_u32(byte as u32).expect("byte in Latin-1 range is a valid scalar")
        } else {
            let ch = char::from_u32(256 + next_extra).expect("U+0100.. is a valid scalar");
            next_extra += 1;
            ch
        };
    }
    table
});

/// Reverse of [`BYTE_TO_CHAR`]: representative scalar back to its byte value.
static CHAR_TO_BYTE: LazyLock<HashMap<char, u8>> = LazyLock::new(|| {
    BYTE_TO_CHAR
        .iter()
        .enumerate()
        .map(|(byte, &ch)| (ch, byte as u8))
        .collect()
});

/// The single-character token representing one raw byte.
pub(crate) fn byte_to_token(byte: u8) -> CompactString {
    BYTE_TO_CHAR[byte as usize].to_compact_string()
}

/// Split a string into one single-character token per raw UTF-8 byte.
pub(crate) fn split_bytes(s: &str) -> Vec<CompactString> {
    s.bytes().map(byte_to_token).collect()
}

/// Reassemble the original string from a concatenation of byte-char tokens.
///
/// Each scalar in `s` is mapped back to its byte; the resulting byte sequence
/// is then decoded as UTF-8. Errors if `s` contains a scalar that is not a
/// byte representative, or if the bytes are not valid UTF-8.
pub(crate) fn decode_byte_string(s: &str) -> Result<String, String> {
    let mut bytes = Vec::with_capacity(s.len());
    for ch in s.chars() {
        match CHAR_TO_BYTE.get(&ch) {
            Some(&byte) => bytes.push(byte),
            None => {
                return Err(format!(
                    "Cannot decode byte-level token: character '{ch}' (U+{:04X}) \
                     is not a valid byte representative.",
                    ch as u32
                ))
            }
        }
    }
    String::from_utf8(bytes).map_err(|e| format!("Byte-level decode produced invalid UTF-8: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bijection_is_total_and_invertible() {
        // All 256 byte values map to distinct scalars, and the reverse map
        // recovers every one of them.
        assert_eq!(CHAR_TO_BYTE.len(), 256);
        for byte in 0u16..256 {
            let byte = byte as u8;
            let ch = BYTE_TO_CHAR[byte as usize];
            assert_eq!(CHAR_TO_BYTE.get(&ch), Some(&byte));
        }
    }

    #[test]
    fn test_no_byte_char_is_whitespace() {
        // Merge files are space-separated, so no representative may be a space.
        for &ch in BYTE_TO_CHAR.iter() {
            assert!(
                !ch.is_whitespace(),
                "byte char {ch:?} must not be whitespace"
            );
        }
    }

    #[test]
    fn test_printable_ascii_maps_to_itself() {
        // All of SMILES is printable ASCII, which maps to its own codepoint.
        for byte in b'!'..=b'~' {
            assert_eq!(BYTE_TO_CHAR[byte as usize], byte as char);
        }
    }

    #[test]
    fn test_split_then_decode_roundtrip() {
        // Including a multi-byte character ('é') and an emoji to exercise the
        // multi-byte reassembly path.
        for s in [
            "CCO",
            "CC(=O)O",
            "Cl/C=C\\Br",
            "C\u{00e9}O",
            "",
            "\u{1F600}",
        ] {
            let units = split_bytes(s);
            let joined: String = units.iter().map(|u| u.as_str()).collect();
            assert_eq!(decode_byte_string(&joined).unwrap(), s);
        }
    }

    #[test]
    fn test_multibyte_char_splits_into_its_bytes() {
        // 'é' is U+00E9 = UTF-8 bytes 0xC3 0xA9 -> two byte-char tokens.
        let units = split_bytes("\u{00e9}");
        assert_eq!(units.len(), 2);
        assert_eq!(units[0], byte_to_token(0xC3));
        assert_eq!(units[1], byte_to_token(0xA9));
    }

    #[test]
    fn test_decode_rejects_non_byte_char() {
        // A scalar with no byte representative cannot be decoded.
        assert!(decode_byte_string("\u{4E2D}").is_err());
    }
}
