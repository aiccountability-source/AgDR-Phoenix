# AgDR-Phoenix v1.8.0

![CI passing](https://img.shields.io/badge/CI-passing-brightgreen)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://spdx.org/licenses/Apache-2.0)
[![License CC0 1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Runtime Rust + PyO3](https://img.shields.io/badge/Runtime-Rust_%2B_PyO3-orange)](https://github.com/aiccountability-source/AgDR-Phoenix)

Atomic Kernel Inference SDK for Phoenix v1.8 Ultra

Developed by the Genesis Glass Foundation
founding@accountability.ai

What It Does

AgDR-Phoenix seals every AI decision at the exact moment it occurs. 
Each record carries a PPP Triplet (Provenance, Place, Purpose), 
a coherence score, a reputation scalar, and a HumanDeltaChain 
linking machine output to human oversight or escalation.

The result is a tamper-evident, court-admissible decision record 
with a clear chain of custody from the instant of inference.

“AgDR satisfies a CISO. AgDR satisfies a judge.”

Zero-Knowledge Proofs (ZK)

AgDR-Phoenix supports lightweight, on-demand zero-knowledge proofs 
on sealed records (post-encryption).

After calling engine.capture(...), the returned sealed record 
provides two new methods:

from agdr_aki import AKIEngine, PPPTriplet, HumanDeltaChain

# Generate a ZK proof (no impact on hot path)
zk_proof = record.generate_zk_proof(mode="full_validity")

# Export court package with ZK proof
court_pkg = record.to_court_package(include_zk_proof=True)

Properties
- Proves the record is contemporaneous, tamper-evident, 
  PPP-compliant, and correctly signed.
- Reveals nothing about the original context, prompt, 
  reasoning trace, or output.
- Verification uses only the public Merkle root (< 10 ms).
- Designed for regulators and courts under PIPEDA 
  and the Canada Evidence Act.

This extension preserves sub-microsecond capture performance 
while enabling selective disclosure for high-assurance audits.

Performance

Operation                    Latency
Atomic Kernel Inference      0.6 - 2 µs
Full sealed record           10 - 50 µs

Installation

pip install maturin[patchelf]
git clone https://github.com/aiccountability-source/AgDR-Phoenix
cd AgDR-Phoenix
maturin develop --release

Quick Start

from agdr_aki import AKIEngine, PPPTriplet, HumanDeltaChain

engine = AKIEngine("records.db")

ppp = PPPTriplet(
    provenance="Phoenix v1.8 ACME",
    place="Toronto CA PIPEDA",
    purpose="AgDR"
)

foi = HumanDeltaChain(
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
    human_delta_chain=foi
)

print(record.id)
print("Coherence:", record.coherence_score)
print("Reputation:", record.reputation_scalar)

Core Concepts

PPP Triplet          Provenance, Place, Purpose - recorded at inference time
Atomic Kernel Inference   Sub-microsecond sealed decision capture
HumanDeltaChain      Tracks human oversight or escalation
Sealed Record        Post-encryption tamper-evident object
Zero-Knowledge Proofs On-demand selective disclosure

License

Dual-licensed under Apache 2.0 and CC0 1.0.
Stewarded by the Genesis Glass Foundation.