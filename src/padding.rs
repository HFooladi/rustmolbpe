//! Padding and batch processing methods for the SMILES tokenizer.

use std::collections::HashMap;

/// Pad a batch of token sequences to the same length.
///
/// # Arguments
/// * `sequences` - List of token ID sequences to pad
/// * `max_length` - Maximum length to pad/truncate to (None = use longest sequence)
/// * `padding` - Padding side: "right" (default) or "left"
/// * `truncation` - Whether to truncate sequences longer than max_length
/// * `return_attention_mask` - Whether to return attention masks
/// * `pad_token_id` - The ID of the padding token
///
/// # Returns
/// A HashMap with 'input_ids' and optionally 'attention_mask'
pub(crate) fn pad(
    sequences: Vec<Vec<u32>>,
    max_length: Option<usize>,
    padding: &str,
    truncation: bool,
    return_attention_mask: bool,
    pad_token_id: u32,
) -> HashMap<String, Vec<Vec<u32>>> {
    if sequences.is_empty() {
        let mut result = HashMap::new();
        result.insert("input_ids".to_string(), Vec::new());
        if return_attention_mask {
            result.insert("attention_mask".to_string(), Vec::new());
        }
        return result;
    }

    // Determine the target length
    let longest = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let target_len = match max_length {
        Some(max) => {
            if truncation {
                max
            } else {
                std::cmp::max(max, longest)
            }
        }
        None => longest,
    };

    let pad_left = padding == "left";

    let mut padded_sequences = Vec::with_capacity(sequences.len());
    let mut attention_masks = if return_attention_mask {
        Vec::with_capacity(sequences.len())
    } else {
        Vec::new()
    };

    for seq in sequences {
        let seq_len = seq.len();

        // Truncate if needed
        let truncated: Vec<u32> = if truncation && seq_len > target_len {
            seq[..target_len].to_vec()
        } else {
            seq
        };

        let current_len = truncated.len();
        let pad_count = target_len.saturating_sub(current_len);

        // Create padded sequence
        let padded = if pad_left {
            let mut p = vec![pad_token_id; pad_count];
            p.extend(truncated);
            p
        } else {
            let mut p = truncated;
            p.extend(vec![pad_token_id; pad_count]);
            p
        };

        // Create attention mask (1 for real tokens, 0 for padding)
        if return_attention_mask {
            let mask = if pad_left {
                let mut m = vec![0u32; pad_count];
                m.extend(vec![1u32; current_len]);
                m
            } else {
                let mut m = vec![1u32; current_len];
                m.extend(vec![0u32; pad_count]);
                m
            };
            attention_masks.push(mask);
        }

        padded_sequences.push(padded);
    }

    let mut result = HashMap::new();
    result.insert("input_ids".to_string(), padded_sequences);
    if return_attention_mask {
        result.insert("attention_mask".to_string(), attention_masks);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_empty() {
        let result = pad(vec![], None, "right", false, true, 0);
        assert!(result.get("input_ids").unwrap().is_empty());
        assert!(result.get("attention_mask").unwrap().is_empty());
    }

    #[test]
    fn test_pad_right() {
        let sequences = vec![vec![1, 2], vec![1, 2, 3, 4]];
        let result = pad(sequences, None, "right", false, true, 0);

        let input_ids = result.get("input_ids").unwrap();
        assert_eq!(input_ids[0], vec![1, 2, 0, 0]);
        assert_eq!(input_ids[1], vec![1, 2, 3, 4]);

        let attention_mask = result.get("attention_mask").unwrap();
        assert_eq!(attention_mask[0], vec![1, 1, 0, 0]);
        assert_eq!(attention_mask[1], vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_pad_left() {
        let sequences = vec![vec![1, 2], vec![1, 2, 3, 4]];
        let result = pad(sequences, None, "left", false, true, 0);

        let input_ids = result.get("input_ids").unwrap();
        assert_eq!(input_ids[0], vec![0, 0, 1, 2]);
        assert_eq!(input_ids[1], vec![1, 2, 3, 4]);

        let attention_mask = result.get("attention_mask").unwrap();
        assert_eq!(attention_mask[0], vec![0, 0, 1, 1]);
        assert_eq!(attention_mask[1], vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_pad_with_truncation() {
        let sequences = vec![vec![1, 2, 3, 4, 5]];
        let result = pad(sequences, Some(3), "right", true, true, 0);

        let input_ids = result.get("input_ids").unwrap();
        assert_eq!(input_ids[0], vec![1, 2, 3]);

        let attention_mask = result.get("attention_mask").unwrap();
        assert_eq!(attention_mask[0], vec![1, 1, 1]);
    }

    #[test]
    fn test_pad_max_length_no_truncation() {
        let sequences = vec![vec![1, 2]];
        let result = pad(sequences, Some(5), "right", false, true, 0);

        let input_ids = result.get("input_ids").unwrap();
        assert_eq!(input_ids[0], vec![1, 2, 0, 0, 0]);
    }

    #[test]
    fn test_pad_without_attention_mask() {
        let sequences = vec![vec![1, 2], vec![1, 2, 3]];
        let result = pad(sequences, None, "right", false, false, 0);

        assert!(result.get("input_ids").is_some());
        assert!(result.get("attention_mask").is_none());
    }
}
