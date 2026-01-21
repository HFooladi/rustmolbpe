use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

/// SMILES atom-level tokenization regex pattern
/// Matches:
/// - Bracketed atoms: [C@@H], [nH], [O-], etc.
/// - Two-char elements: Br, Cl (must come before B, C)
/// - Single-char elements: C, N, O, S, P, F, I, B
/// - Aromatic atoms: b, c, n, o, s, p
/// - Bonds: =, #, -, :, ~
/// - Stereochemistry: @, /, \
/// - Branches: (, )
/// - Disconnected: .
/// - Ring numbers: single digit or %XX
/// - Other: +, ?, >, *, $
const SMILES_ATOM_PATTERN: &str = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])";

/// Special tokens for sequence modeling
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

/// Number of special tokens (always reserved at IDs 0-3)
pub const NUM_SPECIAL_TOKENS: u32 = 4;

type Pair = (u32, u32);

/// A BPE tokenizer specifically designed for molecular SMILES strings.
///
/// Unlike byte-level BPE tokenizers, this tokenizer uses atom-level pre-tokenization
/// where multi-character atoms (like Br, Cl, [C@@H]) are treated as single tokens.
#[pyclass(module = "rustmolbpe")]
pub struct SmilesTokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<Pair, u32>,
    /// Maps atom strings to token IDs
    pub atom_to_id: AHashMap<CompactString, u32>,
    /// Reverse mapping: token ID to atom string
    pub id_to_atom: Vec<CompactString>,
    /// Compiled SMILES regex pattern
    compiled_pattern: Regex,
}

impl Default for SmilesTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------ Internal helpers ------------------------

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
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

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    /// set of word indices where this pair may occur and needs processing
    pos: AHashSet<usize>,
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

#[inline]
fn count_pairs_parallel(
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

/// Tokenize a SMILES string into atom-level tokens
fn atomwise_tokenize(smiles: &str, pattern: &Regex) -> Vec<CompactString> {
    let mut tokens = Vec::new();
    for m in pattern.find_iter(smiles).flatten() {
        tokens.push(CompactString::from(m.as_str()));
    }
    tokens
}

// ------------------------ END helpers ------------------------

impl SmilesTokenizer {
    /// Core incremental BPE training given unique words and their counts.
    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, num_merges: u32) {
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
        let base_vocab_size = self.id_to_atom.len() as u32;

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

            // Record merge
            let new_id = base_vocab_size + merges_done;
            self.merges.insert(top.pair, new_id);

            // Build merged token string
            let left_str = &self.id_to_atom[top.pair.0 as usize];
            let right_str = &self.id_to_atom[top.pair.1 as usize];
            let merged_str = CompactString::from(format!("{}{}", left_str, right_str));
            self.id_to_atom.push(merged_str.clone());
            self.atom_to_id.insert(merged_str, new_id);

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
            if num_merges > 0 {
                let current_percent = (merges_done * 100) / num_merges;
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
}

/// Public methods for the SmilesTokenizer class that will be exposed to Python.
#[pymethods]
impl SmilesTokenizer {
    /// Create a new SmilesTokenizer with special tokens initialized
    #[new]
    pub fn new() -> Self {
        let mut tokenizer = Self {
            merges: StdHashMap::new(),
            atom_to_id: AHashMap::new(),
            id_to_atom: Vec::new(),
            compiled_pattern: Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern"),
        };
        tokenizer.init_special_tokens();
        tokenizer
    }

    /// Initialize special tokens (PAD=0, UNK=1, BOS=2, EOS=3)
    fn init_special_tokens(&mut self) {
        // Only add if not already present
        if self.id_to_atom.is_empty() {
            for (id, token) in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
                .iter()
                .enumerate()
            {
                let token_str = CompactString::from(*token);
                self.atom_to_id.insert(token_str.clone(), id as u32);
                self.id_to_atom.push(token_str);
            }
        }
    }

    /// Get the PAD token ID (always 0)
    #[getter]
    pub fn pad_token_id(&self) -> u32 {
        0
    }

    /// Get the UNK token ID (always 1)
    #[getter]
    pub fn unk_token_id(&self) -> u32 {
        1
    }

    /// Get the BOS token ID (always 2)
    #[getter]
    pub fn bos_token_id(&self) -> u32 {
        2
    }

    /// Get the EOS token ID (always 3)
    #[getter]
    pub fn eos_token_id(&self) -> u32 {
        3
    }

    /// Get the PAD token string
    #[getter]
    pub fn pad_token(&self) -> &'static str {
        PAD_TOKEN
    }

    /// Get the UNK token string
    #[getter]
    pub fn unk_token(&self) -> &'static str {
        UNK_TOKEN
    }

    /// Get the BOS token string
    #[getter]
    pub fn bos_token(&self) -> &'static str {
        BOS_TOKEN
    }

    /// Get the EOS token string
    #[getter]
    pub fn eos_token(&self) -> &'static str {
        EOS_TOKEN
    }

    /// Train from a streaming iterator of SMILES strings.
    ///
    /// # Arguments
    /// * `iterator` - Python iterator yielding SMILES strings
    /// * `vocab_size` - Target vocabulary size (base atoms + merges)
    /// * `buffer_size` - Number of SMILES to process per batch (default: 8192)
    /// * `min_frequency` - Minimum frequency for a merge to be learned (default: 2)
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, min_frequency=2))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, min_frequency=2)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        min_frequency: u32,
    ) -> PyResult<()> {
        // Clear existing state and reinitialize special tokens
        self.merges.clear();
        self.atom_to_id.clear();
        self.id_to_atom.clear();
        self.init_special_tokens();

        // Prepare Python iterator
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Global atom counts and SMILES counts
        let mut atom_counts: AHashMap<CompactString, u64> = AHashMap::new();
        let mut smiles_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();

        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "Processing SMILES from iterator (buffer_size: {})",
            buffer_size
        );
        let mut total_smiles = 0u64;

        // Helper: refill buffer from Python iterator
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::attach(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            })
        };

        // Stream ingestion loop
        let pattern = self.compiled_pattern.clone();
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_smiles += buf.len() as u64;

            // Process buffer in parallel
            let local: Vec<(AHashMap<CompactString, u64>, Vec<CompactString>)> = py.detach(|| {
                buf.par_iter()
                    .map(|smi| {
                        let atoms = atomwise_tokenize(smi, &pattern);
                        let mut atom_map: AHashMap<CompactString, u64> = AHashMap::new();
                        for atom in &atoms {
                            *atom_map.entry(atom.clone()).or_default() += 1;
                        }
                        (atom_map, atoms)
                    })
                    .collect()
            });

            // Merge into global counts
            for (local_atom_counts, atoms) in local {
                for (k, v) in local_atom_counts {
                    *atom_counts.entry(k).or_default() += v;
                }
                *smiles_counts.entry(atoms).or_default() += 1;
            }

            if exhausted {
                break;
            }
        }

        log::info!(
            "Processed {} SMILES total, {} unique atom sequences, {} unique atoms",
            total_smiles,
            smiles_counts.len(),
            atom_counts.len()
        );

        // Build base vocabulary from atoms (sorted by frequency, then alphabetically)
        let mut atoms_sorted: Vec<_> = atom_counts.into_iter().collect();
        atoms_sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for (atom, _count) in atoms_sorted {
            let id = self.id_to_atom.len() as u32;
            self.atom_to_id.insert(atom.clone(), id);
            self.id_to_atom.push(atom);
        }

        let base_vocab_size = self.id_to_atom.len() as u32;
        log::info!("Built base vocabulary with {} atoms", base_vocab_size);

        if vocab_size <= base_vocab_size {
            log::info!(
                "vocab_size ({}) <= base vocab size ({}), no merges needed",
                vocab_size,
                base_vocab_size
            );
            return Ok(());
        }

        let num_merges = vocab_size - base_vocab_size;

        // Convert SMILES to Words
        let mut words = Vec::with_capacity(smiles_counts.len());
        let mut cvec = Vec::with_capacity(smiles_counts.len());

        for (atoms, count) in smiles_counts.into_iter() {
            if count >= min_frequency as i32 {
                let ids: Vec<u32> = atoms
                    .iter()
                    .map(|a| *self.atom_to_id.get(a).unwrap())
                    .collect();
                words.push(Word::new(ids));
                cvec.push(count);
            }
        }

        self.train_core_incremental(words, cvec, num_merges);
        Ok(())
    }

    /// Load vocabulary from a SMILESPE-format file.
    /// Format: one merge per line, `token1 token2` (space-separated)
    /// Special tokens (PAD, UNK, BOS, EOS) are always added at IDs 0-3.
    #[pyo3(signature = (path))]
    pub fn load_vocabulary(&mut self, path: &str) -> PyResult<()> {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Cannot open file: {}", e))
        })?;
        let reader = BufReader::new(file);

        // Clear existing state and initialize special tokens first
        self.merges.clear();
        self.atom_to_id.clear();
        self.id_to_atom.clear();
        self.init_special_tokens();

        // First pass: collect all unique atoms from merge rules
        let mut all_atoms: AHashSet<CompactString> = AHashSet::new();
        let mut merge_pairs: Vec<(CompactString, CompactString)> = Vec::new();

        for line in reader.lines() {
            let line = line
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read error: {}", e)))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Split by space - first space separates the two tokens
            let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid line in vocabulary file: '{}'",
                    trimmed
                )));
            }

            let left = CompactString::from(parts[0]);
            let right = CompactString::from(parts[1]);
            all_atoms.insert(left.clone());
            all_atoms.insert(right.clone());
            merge_pairs.push((left, right));
        }

        // Build base vocabulary from atoms (sorted for determinism)
        let mut atoms_sorted: Vec<_> = all_atoms.into_iter().collect();
        atoms_sorted.sort();

        for atom in atoms_sorted {
            let id = self.id_to_atom.len() as u32;
            self.atom_to_id.insert(atom.clone(), id);
            self.id_to_atom.push(atom);
        }

        // Build merges, adding merged tokens to vocabulary as we go
        // We need to process merges in order and assign IDs incrementally
        for (left, right) in merge_pairs.iter() {
            let left_id = *self.atom_to_id.get(left).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown token: {}", left))
            })?;
            let right_id = *self.atom_to_id.get(right).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown token: {}", right))
            })?;

            let merged_str = CompactString::from(format!("{}{}", left, right));

            // Get or create the merged token ID
            let new_id = if let Some(&existing_id) = self.atom_to_id.get(&merged_str) {
                // Token already exists (it appeared in merge rules), use existing ID
                existing_id
            } else {
                // Create new token
                let id = self.id_to_atom.len() as u32;
                self.atom_to_id.insert(merged_str.clone(), id);
                self.id_to_atom.push(merged_str);
                id
            };

            self.merges.insert((left_id, right_id), new_id);
        }

        let base_vocab_size = (self.id_to_atom.len() - self.merges.len()) as u32;

        log::info!(
            "Loaded vocabulary: {} base atoms, {} merges, {} total vocab",
            base_vocab_size,
            merge_pairs.len(),
            self.vocab_size()
        );

        Ok(())
    }

    /// Save vocabulary to a SMILESPE-format file.
    /// Format: one merge per line, `token1 token2` (space-separated)
    #[pyo3(signature = (path))]
    pub fn save_vocabulary(&self, path: &str) -> PyResult<()> {
        let mut file = File::create(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Cannot create file: {}", e))
        })?;

        // Sort merges by their resulting token ID (order learned)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &new_id)| new_id);

        for (&(left_id, right_id), _) in sorted_merges {
            let left_str = &self.id_to_atom[left_id as usize];
            let right_str = &self.id_to_atom[right_id as usize];
            writeln!(file, "{} {}", left_str, right_str)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Write error: {}", e)))?;
        }

        Ok(())
    }

    /// Return the vocabulary size (base atoms + merges)
    #[getter]
    pub fn vocab_size(&self) -> u32 {
        self.id_to_atom.len() as u32
    }

    /// Return the number of base atoms in the vocabulary
    #[getter]
    pub fn base_vocab_size(&self) -> u32 {
        (self.id_to_atom.len() - self.merges.len()) as u32
    }

    /// Return the number of learned merges
    #[getter]
    pub fn num_merges(&self) -> u32 {
        self.merges.len() as u32
    }

    /// Check if the tokenizer has been trained or has a vocabulary loaded.
    /// Returns true if there are merge rules, false otherwise.
    pub fn is_trained(&self) -> bool {
        !self.merges.is_empty()
    }

    /// Return the learned merge rules as (left, right, merged) string tuples.
    /// Merges are returned in order of learning priority (by merged token ID).
    pub fn get_merges(&self) -> Vec<(String, String, String)> {
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &new_id)| new_id);

        sorted_merges
            .into_iter()
            .map(|(&(left_id, right_id), &merged_id)| {
                let left_str = self.id_to_atom[left_id as usize].to_string();
                let right_str = self.id_to_atom[right_id as usize].to_string();
                let merged_str = self.id_to_atom[merged_id as usize].to_string();
                (left_str, right_str, merged_str)
            })
            .collect()
    }

    /// Return the vocabulary as a list of (token_string, token_id) tuples
    pub fn get_vocabulary(&self) -> Vec<(String, u32)> {
        self.id_to_atom
            .iter()
            .enumerate()
            .map(|(id, token)| (token.to_string(), id as u32))
            .collect()
    }

    /// Encode a SMILES string into token IDs using greedy longest-match.
    /// This produces better compression than the standard BPE merge-order approach.
    /// Unknown atoms are encoded as UNK token.
    #[pyo3(signature = (smiles, add_special_tokens=false))]
    pub fn encode(&self, smiles: &str, add_special_tokens: bool) -> Vec<u32> {
        // Step 1: Atomwise tokenization
        let atoms = atomwise_tokenize(smiles, &self.compiled_pattern);

        if atoms.is_empty() {
            if add_special_tokens {
                return vec![self.bos_token_id(), self.eos_token_id()];
            }
            return Vec::new();
        }

        // Step 2: Greedy longest-match encoding
        // At each position, find the longest sequence of atoms that matches a token
        let mut result = Vec::new();

        // Add BOS token if requested
        if add_special_tokens {
            result.push(self.bos_token_id());
        }

        let mut pos = 0;

        while pos < atoms.len() {
            let mut best_len = 1;
            let mut best_id = self.atom_to_id.get(&atoms[pos]).copied();

            // Try progressively longer matches (up to reasonable limit)
            let max_len = std::cmp::min(atoms.len() - pos, 32); // Cap at 32 atoms
            let mut concat = atoms[pos].clone();

            for len in 2..=max_len {
                concat.push_str(&atoms[pos + len - 1]);
                if let Some(&id) = self.atom_to_id.get(&concat) {
                    best_len = len;
                    best_id = Some(id);
                }
            }

            // Use UNK token for unknown atoms instead of skipping
            result.push(best_id.unwrap_or(self.unk_token_id()));
            pos += best_len;
        }

        // Add EOS token if requested
        if add_special_tokens {
            result.push(self.eos_token_id());
        }

        result
    }

    /// Decode token IDs back to a SMILES string
    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let mut result = String::new();

        for id in ids {
            if (id as usize) < self.id_to_atom.len() {
                result.push_str(&self.id_to_atom[id as usize]);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown token id: {}",
                    id
                )));
            }
        }

        Ok(result)
    }

    /// Encode multiple SMILES strings in parallel using rayon.
    #[pyo3(signature = (smiles_list, add_special_tokens=false))]
    #[pyo3(text_signature = "(self, smiles_list, add_special_tokens=False)")]
    pub fn batch_encode(
        &self,
        py: Python<'_>,
        smiles_list: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let results = py.detach(|| {
            smiles_list
                .par_iter()
                .map(|smi| self.encode(smi, add_special_tokens))
                .collect::<Vec<Vec<u32>>>()
        });

        Ok(results)
    }

    /// Decode multiple token ID sequences in parallel.
    #[pyo3(signature = (ids_list))]
    pub fn batch_decode(&self, py: Python<'_>, ids_list: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        let id_to_atom = &self.id_to_atom;
        let results: Result<Vec<String>, _> = py.detach(|| {
            ids_list
                .par_iter()
                .map(|ids| {
                    let mut result = String::new();
                    for &id in ids {
                        if (id as usize) < id_to_atom.len() {
                            result.push_str(&id_to_atom[id as usize]);
                        } else {
                            return Err(format!("Unknown token id: {}", id));
                        }
                    }
                    Ok(result)
                })
                .collect()
        });

        results.map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Pad a batch of token sequences to the same length.
    ///
    /// # Arguments
    /// * `sequences` - List of token ID sequences to pad
    /// * `max_length` - Maximum length to pad/truncate to (None = use longest sequence)
    /// * `padding` - Padding side: "right" (default) or "left"
    /// * `truncation` - Whether to truncate sequences longer than max_length
    /// * `return_attention_mask` - Whether to return attention masks
    ///
    /// # Returns
    /// A dict with 'input_ids' and optionally 'attention_mask'
    #[pyo3(signature = (sequences, max_length=None, padding="right", truncation=false, return_attention_mask=true))]
    pub fn pad(
        &self,
        sequences: Vec<Vec<u32>>,
        max_length: Option<usize>,
        padding: &str,
        truncation: bool,
        return_attention_mask: bool,
    ) -> PyResult<std::collections::HashMap<String, Vec<Vec<u32>>>> {
        if sequences.is_empty() {
            let mut result = std::collections::HashMap::new();
            result.insert("input_ids".to_string(), Vec::new());
            if return_attention_mask {
                result.insert("attention_mask".to_string(), Vec::new());
            }
            return Ok(result);
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
        let pad_id = self.pad_token_id();

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
                let mut p = vec![pad_id; pad_count];
                p.extend(truncated);
                p
            } else {
                let mut p = truncated;
                p.extend(vec![pad_id; pad_count]);
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

        let mut result = std::collections::HashMap::new();
        result.insert("input_ids".to_string(), padded_sequences);
        if return_attention_mask {
            result.insert("attention_mask".to_string(), attention_masks);
        }

        Ok(result)
    }

    /// Encode and pad a batch of SMILES strings in one call.
    ///
    /// Convenience method that combines batch_encode and pad.
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
    ) -> PyResult<std::collections::HashMap<String, Vec<Vec<u32>>>> {
        let sequences = self.batch_encode(py, smiles_list, add_special_tokens)?;
        self.pad(
            sequences,
            max_length,
            padding,
            truncation,
            return_attention_mask,
        )
    }

    /// Get token string for a given token ID
    pub fn id_to_token(&self, id: u32) -> PyResult<String> {
        if (id as usize) < self.id_to_atom.len() {
            Ok(self.id_to_atom[id as usize].to_string())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown token id: {}",
                id
            )))
        }
    }

    /// Get token ID for a given token string
    pub fn token_to_id(&self, token: &str) -> PyResult<u32> {
        self.atom_to_id
            .get(&CompactString::from(token))
            .copied()
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown token: {}", token))
            })
    }

    /// Pickle support: return (cls, args, state) for serialization
    pub fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PyType>,
        Bound<'py, PyTuple>,
        Bound<'py, PyDict>,
    )> {
        // Get the class type
        let cls = py.get_type::<Self>();

        // Create state dict
        let state = PyDict::new(py);

        // Serialize id_to_atom as list of strings
        let id_to_atom_list: Vec<&str> = self.id_to_atom.iter().map(|s| s.as_str()).collect();
        state.set_item("id_to_atom", PyList::new(py, id_to_atom_list)?)?;

        // Serialize merges as list of ((left_id, right_id), merged_id) tuples
        let merges_list: Vec<((u32, u32), u32)> = self
            .merges
            .iter()
            .map(|(&(l, r), &m)| ((l, r), m))
            .collect();
        state.set_item("merges", merges_list)?;

        // Version for future compatibility
        state.set_item("version", 1u32)?;

        // Return (cls, args, state) - pickle will call cls(*args).__setstate__(state)
        // Use empty tuple for args since SmilesTokenizer::new() takes no arguments
        let args = PyTuple::empty(py);
        Ok((cls, args, state))
    }

    /// Pickle support: restore state from serialization
    pub fn __setstate__(&mut self, state: Bound<'_, PyDict>) -> PyResult<()> {
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

        self.id_to_atom.clear();
        self.atom_to_id.clear();

        for (id, atom) in id_to_atom_list.into_iter().enumerate() {
            let compact_atom = CompactString::from(atom);
            self.atom_to_id.insert(compact_atom.clone(), id as u32);
            self.id_to_atom.push(compact_atom);
        }

        // Restore merges
        let merges_list: Vec<((u32, u32), u32)> = state
            .get_item("merges")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'merges' in pickle state")
            })?
            .extract()?;

        self.merges.clear();
        for ((left_id, right_id), merged_id) in merges_list {
            self.merges.insert((left_id, right_id), merged_id);
        }

        // Recreate compiled pattern (always the same constant pattern)
        self.compiled_pattern = Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern");

        Ok(())
    }
}

/// Tokenize a SMILES string into atom-level tokens (exposed to Python)
#[pyfunction]
#[pyo3(name = "atomwise_tokenize")]
fn atomwise_tokenize_py(smiles: &str) -> Vec<String> {
    let pattern = Regex::new(SMILES_ATOM_PATTERN).expect("Invalid SMILES pattern");
    atomwise_tokenize(smiles, &pattern)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// A BPE tokenizer for molecular SMILES with Python bindings
#[pymodule]
fn rustmolbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<SmilesTokenizer>()?;
    m.add_function(wrap_pyfunction!(atomwise_tokenize_py, m)?)?;
    Ok(())
}

// ============================================================================
// RUST TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomwise_tokenize_simple() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("CCO", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("C"),
                CompactString::from("C"),
                CompactString::from("O")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_halogen() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("CBr", &pattern);
        assert_eq!(
            tokens,
            vec![CompactString::from("C"), CompactString::from("Br")]
        );

        let tokens = atomwise_tokenize("CCl", &pattern);
        assert_eq!(
            tokens,
            vec![CompactString::from("C"), CompactString::from("Cl")]
        );
    }

    #[test]
    fn test_atomwise_tokenize_bracket() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("[C@@H](O)C", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("[C@@H]"),
                CompactString::from("("),
                CompactString::from("O"),
                CompactString::from(")"),
                CompactString::from("C")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_aromatic() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("c1ccccc1", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("c"),
                CompactString::from("1"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("c"),
                CompactString::from("1")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_ring_closure() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        // Two-digit ring closure with %
        let tokens = atomwise_tokenize("C%12CC%12", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("C"),
                CompactString::from("%12"),
                CompactString::from("C"),
                CompactString::from("C"),
                CompactString::from("%12")
            ]
        );
    }

    #[test]
    fn test_atomwise_tokenize_bonds() {
        let pattern = Regex::new(SMILES_ATOM_PATTERN).unwrap();
        let tokens = atomwise_tokenize("C=C#N", &pattern);
        assert_eq!(
            tokens,
            vec![
                CompactString::from("C"),
                CompactString::from("="),
                CompactString::from("C"),
                CompactString::from("#"),
                CompactString::from("N")
            ]
        );
    }

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
    fn test_tokenizer_new() {
        let tok = SmilesTokenizer::new();
        assert!(tok.merges.is_empty());
        // Special tokens are initialized: PAD, UNK, BOS, EOS
        assert_eq!(tok.atom_to_id.len(), 4);
        assert_eq!(tok.id_to_atom.len(), 4);
        assert_eq!(tok.id_to_atom[0].as_str(), PAD_TOKEN);
        assert_eq!(tok.id_to_atom[1].as_str(), UNK_TOKEN);
        assert_eq!(tok.id_to_atom[2].as_str(), BOS_TOKEN);
        assert_eq!(tok.id_to_atom[3].as_str(), EOS_TOKEN);
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

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut tok = SmilesTokenizer::new();

        // Manually add atoms after special tokens (IDs 0-3)
        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);

        let smiles = "CCO";
        let ids = tok.encode(smiles, false);
        assert_eq!(ids, vec![4, 4, 5]);

        let decoded = tok.decode(ids).unwrap();
        assert_eq!(decoded, smiles);
    }

    #[test]
    fn test_encode_with_merge() {
        let mut tok = SmilesTokenizer::new();

        // Set up vocabulary after special tokens (IDs 0-3)
        tok.id_to_atom.push(CompactString::from("C")); // ID 4
        tok.id_to_atom.push(CompactString::from("O")); // ID 5
        tok.id_to_atom.push(CompactString::from("CC")); // ID 6, merged token
        tok.atom_to_id.insert(CompactString::from("C"), 4);
        tok.atom_to_id.insert(CompactString::from("O"), 5);
        tok.atom_to_id.insert(CompactString::from("CC"), 6);

        // Add merge rule: (4, 4) -> 6  (C + C -> CC)
        tok.merges.insert((4, 4), 6);

        let ids = tok.encode("CCO", false);
        assert_eq!(ids, vec![6, 5]); // CC, O

        let decoded = tok.decode(ids).unwrap();
        assert_eq!(decoded, "CCO");
    }
}
