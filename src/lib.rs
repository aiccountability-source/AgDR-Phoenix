use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use blake3::Hash as Blake3Hash;
use ed25519_dalek::{Signer, SigningKey};
use chrono::Utc;
use uuid::Uuid;
use rand::rngs::OsRng;

#[derive(Serialize, Deserialize, Clone)]
pub struct PPPTriplet {
    pub provenance: String,
    pub place: String,
    pub purpose: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HumanDeltaChain {
    pub chain_id: String,
    pub agent_decision_ref: String,
    pub resolved: bool,
    pub terminal_node: String,
}

#[derive(Serialize, Deserialize)]
pub struct CaptureRequest {
    pub ctx: serde_json::Value,
    pub prompt: String,
    pub reasoning_trace: serde_json::Value,
    pub output: String,
    pub ppp: PPPTriplet,
    pub human_delta_chain: Option<HumanDeltaChain>,
}

#[derive(Serialize, Deserialize)]
pub struct DeltaEmbedding {
    pub vector: [i8; 64],
    pub confidence: f64,
    pub delta_norm: f64,
}

#[derive(Serialize, Deserialize)]
pub struct CoreInsightToken {
    pub lesson: String,
    pub confidence: f64,
    pub delta: Option<DeltaEmbedding>,
}

#[derive(Serialize, Deserialize)]
pub struct SealedRecord {
    pub id: String,
    pub timestamp: String,
    pub hash: String,
    pub signature: Vec<u8>,
    pub ppp: PPPTriplet,
    pub ctx: serde_json::Value,
    pub prompt: String,
    pub reasoning_trace: serde_json::Value,
    pub output: String,
    pub merkle_root: String,
    pub coherence_score: f64,
    pub reputation_scalar: f64,
    pub core_insight: Option<CoreInsightToken>,
    pub human_delta_chain: Option<HumanDeltaChain>,
}

#[pyclass]
pub struct PhoenixKernel {
    signing_key: SigningKey,
    merkle_root: Blake3Hash,
    spine: VecDeque<[f32; 64]>,
    reputation: f64,
    coherence_threshold: f64,
    wal_path: String,
    max_spine_size: usize,
}
use std::path::Path;

fn load_or_generate_signing_key(wal_path: &str) -> SigningKey {
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

#[pymethods]
impl PhoenixKernel {
    #[new]
    fn new(wal_path: String, _enable_gpu: bool) -> Self {
        let _ = OpenOptions::new().create(true).append(true).open(&wal_path);
        Self {
            signing_key: load_or_generate_signing_key(&wal_path),
            merkle_root: blake3::hash(b"genesis"),
            spine: VecDeque::with_capacity(500),
            reputation: 0.5,
            coherence_threshold: 0.92,
            wal_path,
            max_spine_size: 500,
        }
    }

    fn capture(&mut self, request_json: String, auto_insight: bool) -> String {
        let req: CaptureRequest = match serde_json::from_str(&request_json) {
            Ok(r) => r,
            Err(_) => return r#"{"error":"invalid request"}"#.to_string(),
        };

        // Simple, deterministic, input-driven embedding (no external dependencies)
        // Production: replace with real sentence-transformer or ONNX model
        let seed = (req.prompt.len() + req.output.len()) as f32;
        let mut current = [0.0f32; 64];
        for i in 0..64 {
            current[i] = ((seed + i as f32 * 0.37) % 6.28).sin() * 0.6 + 0.4;
        }

        self.spine.push_front(current);
        if self.spine.len() > self.max_spine_size {
            self.spine.pop_back();
        }

        let spine_avg = self.weighted_spine_average();

        // Real normalized dot-product coherence (proxy for cosine similarity)
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

        let core_insight = if auto_insight && coherence >= self.coherence_threshold {
            let mut delta_vec = [0i8; 64];
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

        self.reputation = 0.98 * self.reputation + 0.02 * coherence;

        let canonical = format!("{:?}{:?}{:?}", req, coherence, self.reputation);
        let record_hash = blake3::hash(canonical.as_bytes());
        let signature = self.signing_key.sign(record_hash.as_bytes()).to_bytes().to_vec();

        self.merkle_root = self.update_merkle_root(&record_hash);

        let record = SealedRecord {
            id: format!("aki_{}", Uuid::new_v4()),
            timestamp: Utc::now().to_rfc3339(),
            hash: record_hash.to_hex().to_string(),
            signature,
            ppp: req.ppp,
            ctx: req.ctx,
            prompt: req.prompt,
            reasoning_trace: req.reasoning_trace,
            output: req.output,
            merkle_root: self.merkle_root.to_hex().to_string(),
            coherence_score: coherence,
            reputation_scalar: self.reputation,
            core_insight,
            human_delta_chain: req.human_delta_chain,
        };

        self.append_to_wal(&record);

        serde_json::to_string(&record).unwrap()
    }

    fn public_key_hex(&self) -> String {
        hex::encode(self.signing_key.verifying_key().to_bytes())
    }

    fn weighted_spine_average(&self) -> [f32; 64] {
        let mut avg = [0.0f32; 64];
        let n = self.spine.len();
        if n == 0 {
            return avg;
        }
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

    fn update_merkle_root(&mut self, leaf_hash: &Blake3Hash) -> Blake3Hash {
        let combined = format!("{:?}{:?}", self.merkle_root, leaf_hash);
        blake3::hash(combined.as_bytes())
    }

    fn append_to_wal(&self, record: &SealedRecord) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.wal_path) {
            let _ = writeln!(file, "{}", serde_json::to_string(record).unwrap());
        }
    }
}

#[pymodule]
fn agdr_aki(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PhoenixKernel>()?;
    Ok(())
}
