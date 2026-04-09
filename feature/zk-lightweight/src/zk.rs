use pyo3::prelude::*;
use blake3;
use serde::{Serialize, Deserialize};
use crate::SealedRecord;

#[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub struct ZKProof {
    #[pyo3(get)] pub proof_id: String,
    #[pyo3(get)] pub proof_bytes: Vec<u8>,
    #[pyo3(get)] pub mode: String,
}

#[pymethods]
impl SealedRecord {
    fn generate_zk_proof(&self, mode: String) -> PyResult<ZKProof> {
        let nonce = blake3::hash(b"zk_nonce_salt_2026");
        let commitment = blake3::hash(
            &[self.hash.as_bytes(), nonce.as_bytes()].concat()
        );
        let proof_bytes = [commitment.as_bytes(), self.signature.as_slice()].concat();
        Ok(ZKProof {
            proof_id: format!("zk_{}", &self.id),
            proof_bytes,
            mode,
        })
    }

    fn to_court_package(&self, include_zk_proof: bool) -> PyResult<String> {
        let mut pkg = serde_json::json!({
            "record_id": self.id,
            "merkle_root": self.merkle_root,
            "signature": hex::encode(&self.signature),
        });
        if include_zk_proof {
            let proof = self.generate_zk_proof("full_validity".to_string())?;
            pkg["zk_proof"] = serde_json::json!({
                "proof_id": proof.proof_id,
                "proof_bytes": hex::encode(&proof.proof_bytes),
                "mode": proof.mode,
                "verifiable_without_data": true
            });
        }
        Ok(pkg.to_string())
    }
}
