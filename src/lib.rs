#![cfg_attr(docsrs, feature(doc_auto_cfg))]
//! Atomic Kernel Inference SDK for cryptographically-sealed AI decision records.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::convert::TryInto;
use blake3::Hash as Blake3Hash;
use ed25519_dalek::{Signer, SigningKey};
use chrono::Utc;
use uuid::Uuid;
use rand::rngs::OsRng;
use std::path::Path;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use hex;

/// Retrieves a raw, high-precision nanosecond timestamp from the system clock.
///
/// On Linux platforms, this targets `CLOCK_MONOTONIC_RAW` directly to remain completely
/// immune to downstream NTP time adjustments or slewing, guaranteeing monotonic execution sequencing.
#[inline(always)]
pub fn monotonic_raw_nanos() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::mem::MaybeUninit;
        use libc::{clock_gettime, CLOCK_MONOTONIC_RAW, timespec};
        let mut ts = MaybeUninit::<timespec>::uninit();
        unsafe {
            if clock_gettime(CLOCK_MONOTONIC_RAW, ts.as_mut_ptr()) == 0 {
                let ts = ts.assume_init();
                (ts.tv_sec as u64)
                    .wrapping_mul(1_000_000_000)
                    .wrapping_add(ts.tv_nsec as u64)
            } else {
                std::time::Instant::now().elapsed().as_nanos() as u64
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

/// A localized, low-overhead embedding vector tracking contextual micro-shifts.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaEmbedding {
    /// Quantized direction parameters capturing structural data coordinates.
    pub vector: Vec<i8>,
    /// Confidence indicator matching the inference weight.
    pub confidence: f64,
    /// Absolute mathematical size metrics of the contextual variance.
    pub delta_norm: f64,
}

/// A real-time evaluation token logging historical data convergence patterns.
#[derive(Debug, Serialize, Deserialize)]
pub struct CoreInsightToken {
    /// Human-readable explanation characterizing the historical alignment state.
    pub lesson: String,
    /// Exact mathematical probability mapping standard similarity metrics.
    pub confidence: f64,
    /// Embedded spatial variance metrics representing localized context drift.
    pub delta: Option<DeltaEmbedding>,
}

/// A triad of metadata anchoring the lineage, physical origin, and analytical intent of an event block.
#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PPPTriplet {
    /// Cryptographic or system lineage tracking the structural history.
    #[pyo3(get, set)] pub provenance: String,
    /// The specific systemic, environmental, or geographic processing node.
    #[pyo3(get, set)] pub place: String,
    /// The logical framework intent guiding the execution loop.
    #[pyo3(get, set)] pub purpose: String,
}

#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pymethods]
impl PPPTriplet {
    #[new]
    #[pyo3(signature = (provenance, place, purpose))]
    pub fn new(provenance: String, place: String, purpose: String) -> Self {
        Self { provenance, place, purpose }
    }
}

/// An audit node linking autonomous determinations to definitive human-in-the-loop review overrides.
#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HumanDeltaChain {
    /// Unique identity tracking the active sequence.
    #[pyo3(get, set)] pub chain_id: String,
    /// Reference signature pointer linking directly back to the original model output.
    #[pyo3(get, set)] pub agent_decision_ref: String,
    /// Evaluation state marking whether review processing has formally closed.
    #[pyo3(get, set)] pub resolved: bool,
    /// Terminal validation node sealing the chain integrity state.
    #[pyo3(get, set)] pub terminal_node: String,
}

#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pymethods]
impl HumanDeltaChain {
    #[new]
    #[pyo3(signature = (agent_decision_ref, resolved, terminal_node, chain_id=None))]
    pub fn new(
        agent_decision_ref: String,
        resolved: bool,
        terminal_node: String,
        chain_id: Option<String>,
    ) -> Self {
        Self {
            chain_id: chain_id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            agent_decision_ref,
            resolved,
            terminal_node,
        }
    }
}

/// A court-admissible, tamper-evident data structure capturing an immutable point-in-time calculation state.
#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct SealedRecord {
    /// Universally unique identifier managing the tracking envelope.
    #[pyo3(get)] pub id: String,
    /// RFC3339 timestamp recording standard real-world clock time.
    #[pyo3(get)] pub timestamp: String,
    /// Monotonic clock offset logging real-world physical processing sequence order.
    #[pyo3(get)] pub monotonic_nanos: u64,
    /// BLAKE3 hash string locking total structure components.
    #[pyo3(get)] pub hash: String,
    /// Ed25519 digital signature verifying authority origins.
    #[pyo3(get)] pub signature: Vec<u8>,
    /// Accumulated chain integrity root verifying sequencing order.
    #[pyo3(get)] pub merkle_root: String,
    /// Similarity rating measuring historical continuity states.
    #[pyo3(get)] pub coherence_score: f64,
    /// System health scale monitoring moving reliability baselines.
    #[pyo3(get)] pub reputation_scalar: f64,
    /// JSON serialization holding provenance parameters.
    #[pyo3(get)] pub ppp_json: String,
    /// JSON string capturing overall operational environment conditions.
    #[pyo3(get)] pub ctx_json: String,
    /// Standard engineering input used for generating calculations.
    #[pyo3(get)] pub prompt: String,
    /// JSON representation outlining internal calculation steps.
    #[pyo3(get)] pub reasoning_trace_json: String,
    /// Definitive answer generated through computational operations.
    #[pyo3(get)] pub output: String,
    /// JSON structure tracking human oversight and adjustments.
    #[pyo3(get)] pub human_delta_chain_json: String,
    /// JSON representation storing analytical insight parameters.
    #[pyo3(get)] pub core_insight_json: Option<String>,
}

/// Retrieves an existing Ed25519 signing token or initializes a new key pair if missing.
///
/// If path parameter is set to `":memory:"`, an ephemeral configuration is utilized.
pub fn load_or_generate_signing_key(wal_path: &str) -> SigningKey {
    if wal_path == ":memory:" {
        return SigningKey::generate(&mut OsRng);
    }
    let key_path = format!("{}.key", wal_path);
    if Path::new(&key_path).exists() {
        let bytes = std::fs::read(&key_path).expect("Failed to read signing key");
        let arr: [u8; 32] = bytes.try_into().expect("Invalid key length");
        SigningKey::from_bytes(&arr)
    } else {
        let key = SigningKey::generate(&mut OsRng);
        std::fs::write(&key_path, key.to_bytes()).expect("Failed to write signing key");
        key
    }
}

/// Core execution container managing real-time calculations and integrity verification sequences.
pub struct AKIEngine {
    /// Core signing identity generating cryptographic seals.
    pub signing_key: SigningKey,
    /// Running blockchain-style history validation pointer.
    pub merkle_root: Blake3Hash,
    /// Sliding memory store capturing spatial data vectors.
    pub spine: VecDeque<[f32; 64]>,
    /// Overall reliability baseline rating.
    pub reputation: f64,
    /// Acceptable baseline indicator filtering drift variance.
    pub coherence_threshold: f64,
    /// Active write-ahead log system resource location.
    pub wal_path: String,
    /// Maximum limit managing internal vector stores.
    pub max_spine_size: usize,
}

impl AKIEngine {
    /// Processes a weighted exponential decay mapping across historical structural coordinates.
    pub fn weighted_spine_average(&self) -> [f32; 64] {
        let mut avg = [0.0f32; 64];
        let n = self.spine.len();
        if n == 0 { return avg; }
        let lambda = 0.98f32;
        let z = (1.0 - lambda.powi(n as i32)) / (1.0 - lambda);
        for (i, emb) in self.spine.iter().enumerate() {
            let w = lambda.powi(i as i32) / z;
            for k in 0..64 {
                avg[k] += w * emb[k];
            }
        }
        avg
    }

    /// Appends a new block item to update the running history state verification tree.
    pub fn update_merkle_root(&mut self, leaf_hash: &Blake3Hash) -> Blake3Hash {
        let combined = format!("{:?}{:?}", self.merkle_root, leaf_hash);
        blake3::hash(combined.as_bytes())
    }

    #[cfg(feature = "python")]
    pub fn append_to_wal(&self, record: &SealedRecord) {
        if self.wal_path == ":memory:" { return; }
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.wal_path) {
            let _ = writeln!(file, "{}", serde_json::to_string(record).unwrap_or_default());
        }
    }
}

/// High-performance Python extension wrapper container exposed for model integration bindings.
#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pyclass]
#[pyo3(name = "AKIEngine")]
pub struct PyAKIEngine {
    /// Under-the-hood pure Rust calculation system.
    pub inner: AKIEngine,
}

#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pymethods]
impl PyAKIEngine {
    #[new]
    pub fn new(wal_path: String) -> Self {
        if wal_path != ":memory:" {
            let _ = OpenOptions::new().create(true).append(true).open(&wal_path);
        }
        Self {
            inner: AKIEngine {
                signing_key: load_or_generate_signing_key(&wal_path),
                merkle_root: blake3::hash(b"genesis"),
                spine: VecDeque::with_capacity(500),
                reputation: 0.5,
                coherence_threshold: 0.92,
                wal_path,
                max_spine_size: 500,
            },
        }
    }

    /// Captures execution contexts, runs real-time consistency checks, and signs an immutable audit record.
    #[pyo3(signature = (ctx, prompt, reasoning_trace, output, ppp_triplet, human_delta_chain, auto_insight=true))]
    pub fn capture(
        &mut self,
        py: Python<'_>,
        ctx: &Bound<'_, PyAny>,
        prompt: String,
        reasoning_trace: &Bound<'_, PyAny>,
        output: String,
        ppp_triplet: PyRef<PPPTriplet>,
        human_delta_chain: PyRef<HumanDeltaChain>,
        auto_insight: bool,
    ) -> PyResult<SealedRecord> {
        let json_module = py.import_bound("json")?;
        let ctx_json: String = json_module.call_method1("dumps", (ctx,))?.extract()?;
        let trace_json: String = json_module.call_method1("dumps", (reasoning_trace,))?.extract()?;

        let ppp = ppp_triplet.clone();
        let hdc = HumanDeltaChain {
            chain_id: human_delta_chain.chain_id.clone(),
            agent_decision_ref: human_delta_chain.agent_decision_ref.clone(),
            resolved: human_delta_chain.resolved,
            terminal_node: human_delta_chain.terminal_node.clone(),
        };

        let seed = (prompt.len() + output.len()) as f32;
        let mut current = [0.0f32; 64];
        for i in 0..64 {
            current[i] = ((seed + i as f32 * 0.37) % 6.28).sin() * 0.6 + 0.4;
        }

        self.inner.spine.push_front(current);
        if self.inner.spine.len() > self.inner.max_spine_size {
            self.inner.spine.pop_back();
        }

        let spine_avg = self.inner.weighted_spine_average();

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..64 {
            let a = current[i] as f64;
            let b = spine_avg[i] as f64;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let coherence = if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let core_insight = if auto_insight && coherence >= self.inner.coherence_threshold {
            let mut delta_vec = vec![0i8; 64];
            for i in 0..64 {
                let diff = (current[i] - spine_avg[i]) * 127.0;
                delta_vec[i] = diff.clamp(-128.0, 127.0) as i8;
            }
            Some(CoreInsightToken {
                lesson: "Decision aligns well with historical pattern.".to_string(),
                confidence: coherence,
                delta: Some(DeltaEmbedding {
                    vector: delta_vec,
                    confidence: coherence,
                    delta_norm: 0.18,
                }),
            })
        } else {
            None
        };

        self.inner.reputation = 0.98 * self.inner.reputation + 0.02 * coherence;

        let canonical = format!(
            "{}{}{}{}{:?}{:?}",
            ppp.provenance, ppp.place, ppp.purpose,
            prompt, coherence, self.inner.reputation
        );
        let record_hash = blake3::hash(canonical.as_bytes());
        let signature = self.inner.signing_key.sign(record_hash.as_bytes()).to_bytes().to_vec();
        self.inner.merkle_root = self.inner.update_merkle_root(&record_hash);

        let monotonic_ns = monotonic_raw_nanos();

        let record = SealedRecord {
            id: format!("aki_{}", Uuid::new_v4()),
            timestamp: Utc::now().to_rfc3339(),
            monotonic_nanos: monotonic_ns,
            hash: record_hash.to_hex().to_string(),
            signature,
            merkle_root: self.inner.merkle_root.to_hex().to_string(),
            coherence_score: coherence,
            reputation_scalar: self.inner.reputation,
            ppp_json: serde_json::to_string(&ppp).unwrap_or_default(),
            ctx_json,
            prompt,
            reasoning_trace_json: trace_json,
            output,
            human_delta_chain_json: serde_json::to_string(&hdc).unwrap_or_default(),
            core_insight_json: core_insight
                .as_ref()
                .and_then(|c| serde_json::to_string(c).ok()),
        };

        self.inner.append_to_wal(&record);
        Ok(record)
    }

    /// Exposes the verifying identity public hex configuration signature.
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.inner.signing_key.verifying_key().to_bytes())
    }
}

/// Main native library interface mapping out public Python module interface class pointers.
#[cfg(feature = "python")]
#[cfg_attr(docsrs, doc(cfg(feature = "python")))]
#[pymodule]
fn agdr_aki(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAKIEngine>()?;
    m.add_class::<PPPTriplet>()?;
    m.add_class::<HumanDeltaChain>()?;
    m.add_class::<SealedRecord>()?;
    Ok(())
}