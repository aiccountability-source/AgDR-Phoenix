# AgDR-Phoenix [![Version](https://img.shields.io/crates/v/agdr-aki)](https://crates.io/crates/agdr-aki)

[![CI](https://github.com/aiccountability-source/AgDR-Phoenix/actions/workflows/ci.yml/badge.svg)](https://github.com/aiccountability-source/AgDR-Phoenix/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://spdx.org/licenses/Apache-2.0)
[![License: CC0 1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Runtime: Rust + PyO3](https://img.shields.io/badge/Runtime-Rust_%2B_PyO3-orange.svg)](https://pyo3.rs)

[![Stewarded by GGF](https://img.shields.io/badge/Stewarded_by-Genesis_Glass_Foundation-4B0082.svg)](https://accountability.ai)

**Atomic Kernel Inference SDK for Phoenix v1.8.5 Ultra**

Developed by the [Genesis Glass Foundation](https://accountability.ai) (Fondation Genèse Cristal)
A federally incorporated Canadian not-for-profit.
[founding@accountability.ai](mailto:founding@accountability.ai)

---

> "AgDR satisfies a CISO. AgDR satisfies a judge."

---

## What It Does

AgDR-Phoenix seals every AI decision at the exact moment it occurs. Each record carries a **PPP Triplet** (Provenance, Place, Purpose), a coherence score, a reputation scalar, and a **HumanDeltaChain** linking machine output to human oversight or escalation.

The result is a tamper-evident, court-admissible decision record with a clear chain of custody from the instant of inference. Zero-knowledge proofs allow selective disclosure to regulators and courts without exposing the underlying reasoning.

---

## Performance

| Operation | Latency |
|---|---|
| Atomic Kernel Inference (hot path) | 0.6 - 2 µs |
| Full sealed record (WAL + Merkle) | 10 - 50 µs |
| ZK proof generation | < 10 ms |
| ZK verification (Merkle root only) | < 10 ms |

---

## Installation

```bash
pip install maturin[patchelf]
git clone https://github.com/aiccountability-source/AgDR-Phoenix
cd AgDR-Phoenix
maturin develop --release
```

---

## Quick Start

```python
from agdr_aki import AKIEngine, PPPTriplet, HumanDeltaChain

engine = AKIEngine("records.db")

ppp = PPPTriplet(
    provenance="Phoenix v1.8 ACME",
    place="Toronto CA PIPEDA",
    purpose="AgDR"
)

hdc = HumanDeltaChain(
    agent_decision_ref="aki_001",
    resolved=True,
    terminal_node="autonomous"
)

record = engine.capture(
    ctx={"claim": "CLM-12345"},
    prompt="Process claim #12345",
    reasoning_trace={"steps": ["analyze", "decide"]},
    output="Approve",
    ppp_triplet=ppp,
    human_delta_chain=hdc
)

print(record.id)
print("Coherence:", record.coherence_score)
print("Reputation:", record.reputation_scalar)
```

---

## Zero-Knowledge Proofs

AgDR-Phoenix supports lightweight on-demand zero-knowledge proofs on sealed records.

```python
# Generate ZK proof
zk_proof = record.generate_zk_proof(mode="full_validity")

# Export court package with ZK proof attached
court_pkg = record.to_court_package(include_zk_proof=True)
print(court_pkg)
```

**Properties:**
- Proves the record is contemporaneous, tamper-evident, PPP-compliant, and correctly signed
- Reveals nothing about the original context, prompt, reasoning trace, or output
- Verification uses only the public Merkle root
- Designed for regulators and courts under PIPEDA and the Canada Evidence Act

---

## Core Concepts

| Concept | Description |
|---|---|
| **PPP Triplet** | Provenance, Place, Purpose immutable legal anchor recorded at inference time |
| **Atomic Kernel Inference (AKI)** | Sub-microsecond sealed decision capture at the exact moment of inference |
| **HumanDeltaChain** | Required fiduciary link tracking human oversight or escalation and enforced at API level |
| **Coherence Score** | Normalized cosine similarity against the agent's historical spine (0.0 - 1.0) |
| **Reputation Scalar** | Exponentially weighted moving average of past coherence, provides a longitudinal reliability signal |
| **Sealed Record** | Tamper-evident post-encryption object with BLAKE3 hash and Ed25519 signature |
| **Zero-Knowledge Proof** | Selective disclosure of record integrity without revealing decision content |

---

## Evidentiary Architecture

| Layer | Implementation |
|---|---|
| Hashing | BLAKE3 |
| Signing | Ed25519 (downstream of AKI hot path) |
| Chaining | Forward-secret Merkle tree |
| Persistence | Write-Ahead Log (WAL) |
| Degradation | Limitations Act 2002 (Ontario): 2-year basic, 15-year ultimate |
| Admissibility | Canada Evidence Act — portable to EU AI Act (August 2026) |
| Privacy | Zero-knowledge proof layer for selective disclosure |

---

## Distinction from Structural Monitoring

AgDR captures **semantic decision provenance**: what the agent decided, why, under what authority, and with what confidence, sealed at the moment of inference.

Structural container monitoring (e.g., the AGA standard) records whether the system was running correctly. AgDR records what the system actually decided. Both serve different functions in an AI governance stack.

---

## Specification and Standards

| Resource | Reference |
|---|---| |
| Reasoning Capture Methodology v1.0 | ISBN 978-1-7389042-1-1 |
| Library and Archives Canada | Deposited under CC-BY 4.0 |
| Steward | Genesis Glass Foundation — federally incorporated, royalty-free anchor in articles of incorporation |

---

## License

Dual-licensed under [Apache 2.0](https://spdx.org/licenses/Apache-2.0) and [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

The AgDR open standard is stewarded by the [Genesis Glass Foundation](https://accountability.ai) (Fondation Genèse Cristal), a federally incorporated Canadian not-for-profit. The royalty-free grant is anchored in the Foundation's articles of incorporation.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are welcome. The standard is open.

---

## Contact

**Genesis Glass Foundation**
[founding@accountability.ai](mailto:founding@accountability.ai)
[https://accountability.ai](https://accountability.ai)
 
