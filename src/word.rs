//! Internal data structures for BPE training.

use std::cmp::Ordering;

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;

use crate::constants::Pair;

/// A sequence of token IDs representing a word during BPE training.
#[derive(Clone, Debug)]
pub(crate) struct Word {
    pub ids: Vec<u32>,
}

impl Word {
    #[inline]
    pub fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    pub fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    pub fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n {
                    Some(self.ids[i + 2])
                } else {
                    None
                };

                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

/// A job in the merge priority queue during BPE training.
#[derive(Debug, Eq)]
pub(crate) struct MergeJob {
    pub pair: Pair,
    pub count: u64,
    /// Set of word indices where this pair may occur and needs processing.
    pub pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count; tie-break to ascending pair order (deterministic)
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // ascending order on the pair when counts tie
            other.pair.cmp(&self.pair)
        }
    }
}

/// Count pairs in parallel across all words.
#[inline]
pub(crate) fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_pairs() {
        let word = Word::new(vec![1, 2, 3, 4]);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert_eq!(pairs, vec![(1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_word_merge_pair() {
        let mut word = Word::new(vec![1, 2, 3, 1, 2]);
        let _deltas = word.merge_pair((1, 2), 99);
        assert_eq!(word.ids, vec![99, 3, 99]);
    }

    #[test]
    fn test_count_pairs_parallel() {
        let words = vec![Word::new(vec![1, 2, 3]), Word::new(vec![1, 2, 4])];
        let counts = vec![1, 2];

        let (pair_counts, positions) = count_pairs_parallel(&words, &counts);

        assert_eq!(pair_counts.get(&(1, 2)), Some(&3));
        assert_eq!(pair_counts.get(&(2, 3)), Some(&1));
        assert_eq!(pair_counts.get(&(2, 4)), Some(&2));

        assert!(positions.get(&(1, 2)).unwrap().contains(&0));
        assert!(positions.get(&(1, 2)).unwrap().contains(&1));
    }
}
