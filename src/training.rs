//! BPE training algorithm implementation.

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;

use crate::constants::Pair;
use crate::word::{count_pairs_parallel, MergeJob, Word};

/// Core incremental BPE training given unique words and their counts.
///
/// # Arguments
/// * `merges` - Mutable reference to the ordered merge list to populate
/// * `atom_to_id` - Mutable reference to the atom-to-ID mapping
/// * `id_to_atom` - Mutable reference to the ID-to-atom vector
/// * `words` - Vector of Word structs representing unique token sequences
/// * `counts` - Corresponding frequency counts for each word
/// * `num_merges` - Maximum number of merge operations to perform
/// * `min_frequency` - Minimum count a pair must have to be merged; training
///   stops early once the most frequent remaining pair falls below it
pub(crate) fn train_core_incremental(
    merges: &mut Vec<(Pair, u32)>,
    atom_to_id: &mut AHashMap<CompactString, u32>,
    id_to_atom: &mut Vec<CompactString>,
    mut words: Vec<Word>,
    counts: Vec<i32>,
    num_merges: u32,
    min_frequency: u32,
) {
    log::info!("Starting BPE training: {} merges to compute", num_merges);

    // ---- Initial pair_counts and where_to_update (parallel) ----
    log::info!(
        "Computing initial pair counts from {} unique sequences",
        words.len()
    );
    let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

    // ---- Build heap ----
    log::info!("Building heap with {} unique pairs", pair_counts.len());
    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, pos) in where_to_update.drain() {
        let c = *pair_counts.get(&pair).unwrap_or(&0);
        if c > 0 {
            heap.push(MergeJob {
                pair,
                count: c as u64,
                pos,
            });
        }
    }

    // ---- Merge loop ----
    log::info!("Starting merge loop");
    let mut merges_done = 0u32;
    let mut last_log_percent = 0u32;
    let base_vocab_size = id_to_atom.len() as u32;

    while merges_done < num_merges {
        let Some(mut top) = heap.pop() else {
            break;
        };

        // Lazy refresh: if the count changed since we queued this job, update and requeue
        let current = *pair_counts.get(&top.pair).unwrap_or(&0);
        if current <= 0 {
            continue;
        }
        if top.count != current as u64 {
            top.count = current as u64;
            heap.push(top);
            continue;
        }

        // Merging only ever raises counts of pairs involving the new token,
        // which are queued fresh, so queued counts never underestimate. The
        // first up-to-date job is therefore the most frequent remaining pair:
        // if it is below min_frequency, no remaining pair qualifies.
        if top.count < u64::from(min_frequency) {
            log::info!(
                "Stopping: most frequent remaining pair occurs {} times (min_frequency: {})",
                top.count,
                min_frequency
            );
            break;
        }

        // Record merge
        let new_id = base_vocab_size + merges_done;
        merges.push((top.pair, new_id));

        // Build merged token string
        let left_str = &id_to_atom[top.pair.0 as usize];
        let right_str = &id_to_atom[top.pair.1 as usize];
        let merged_str = CompactString::from(format!("{}{}", left_str, right_str));
        id_to_atom.push(merged_str.clone());
        atom_to_id.insert(merged_str, new_id);

        // Merge this pair in all words where it occurs
        let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
        for &word_idx in &top.pos {
            let changes = words[word_idx].merge_pair(top.pair, new_id);
            for (pair, delta) in changes {
                let delta_total = delta * counts[word_idx];
                if delta_total != 0 {
                    *pair_counts.entry(pair).or_default() += delta_total;
                    if delta > 0 {
                        local_pos_updates.entry(pair).or_default().insert(word_idx);
                    }
                }
            }
        }

        // Add the updated pair counts back to the heap
        for (pair, pos) in local_pos_updates {
            let cnt = *pair_counts.get(&pair).unwrap_or(&0);
            if cnt > 0 {
                heap.push(MergeJob {
                    pair,
                    count: cnt as u64,
                    pos,
                });
            }
        }

        merges_done += 1;

        // Log progress every 1%
        if let Some(current_percent) = (merges_done * 100).checked_div(num_merges) {
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                    current_percent,
                    merges_done,
                    num_merges,
                    top.pair,
                    new_id,
                    top.count
                );
                last_log_percent = current_percent;
            }
        }
    }

    log::info!("Finished training: {} merges completed", merges_done);
}

// ============================================================================
// RUST TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Train over the base vocabulary {C=0, O=1, N=2} and return the merged
    /// token strings in merge order.
    fn train(words: Vec<Vec<u32>>, counts: Vec<i32>, min_frequency: u32) -> Vec<String> {
        let mut id_to_atom: Vec<CompactString> = ["C", "O", "N"]
            .iter()
            .map(|s| CompactString::from(*s))
            .collect();
        let mut atom_to_id: AHashMap<CompactString, u32> = id_to_atom
            .iter()
            .enumerate()
            .map(|(i, a)| (a.clone(), i as u32))
            .collect();
        let mut merges = Vec::new();
        let base = id_to_atom.len();

        train_core_incremental(
            &mut merges,
            &mut atom_to_id,
            &mut id_to_atom,
            words.into_iter().map(Word::new).collect(),
            counts,
            100,
            min_frequency,
        );

        id_to_atom[base..].iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_min_frequency_stops_at_first_pair_below_threshold() {
        // "CCC" x2: (C, C) occurs 4 times; after merging, (CC, C) occurs 2 times.
        let ccc = || vec![vec![0, 0, 0]];
        assert_eq!(train(ccc(), vec![2], 2), vec!["CC", "CCC"]);
        assert_eq!(train(ccc(), vec![2], 4), vec!["CC"]);
        assert!(train(ccc(), vec![2], 5).is_empty());
    }

    #[test]
    fn test_min_frequency_sums_pair_counts_across_words() {
        // "CCO" and "CCN" occur once each, but (C, C) occurs twice in total.
        assert_eq!(
            train(vec![vec![0, 0, 1], vec![0, 0, 2]], vec![1, 1], 2),
            vec!["CC"]
        );
    }
}
