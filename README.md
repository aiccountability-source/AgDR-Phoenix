# AgDR-Phoenix v1.8.0

**Atomic Kernel Inference SDK for Phoenix v1.8 Ultra**  
Developed by the [Genesis Glass Foundation](https://accountability.ai) | founding@accountability.ai

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://spdx.org/licenses/Apache-2.0)
[![License: CC0 1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Runtime: Rust + PyO3](https://img.shields.io/badge/Runtime-Rust_%2B_PyO3-orange.svg)](https://pyo3.rs)

-----

## What It Does

AgDR-Phoenix seals every AI decision at the exact moment it occurs. Each record carries a **PPP Triplet** (Provenance, Place, Purpose), a coherence score, a reputation scalar, and a **HumanDeltaChain** linking machine output to human oversight or escalation.

The result is a tamper-evident, court-admissible decision record with a clear chain of custody from the instant of inference.

> “AgDR satisfies a CISO. AgDR satisfies a judge.”

-----

## Performance

|Operation                         |Latency   |
|----------------------------------|----------|
|Atomic Kernel Inference (hot path)|0.6 - 2 µs|
|Full sealed record (WAL + Merkle) |10 - 50 µs|

Sub-microsecond capture records the decision contemporaneously with inference. Contemporaneous sealing minimises alteration risk and establishes chain of custody at the precise instant the decision was made. Regulators and courts place high evidentiary weight on immediate, tamper-evident records.

-----

## Installation

```bash
pip install maturin[patchelf]
git clone https://github.com/aiccountability-source/AgDR-Phoenix
cd AgDR-Phoenix
maturin develop --release
```

-----

## Quick Start

```python
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
```

-----

## Core Concepts

### PPP Triplet

**Provenance, Place, Purpose.** Immutable anchor recorded at inference time. Establishes the legal and regulatory context of the decision before any downstream processing occurs.

### Atomic Kernel Inference (AKI)

The “i” moment: the exact point at which the decision is sealed with cryptographic protection. Ed25519 signing runs downstream of the hot path; AKI capture itself operates at sub-microsecond latency.

### HumanDeltaChain (FOI)

**Fiduciary Office Intervener.** Links machine decisions to human oversight or escalation. Required for meaningful human-in-the-loop and strong evidentiary weight. The `terminal_node` field records whether the decision resolved autonomously or escalated to a human agent.

### Coherence Score

Measures alignment with the agent’s historical sensory spine. Range: 0.0 to 1.0. Drives band routing for automatic degradation logic.

### Reputation Scalar

Exponentially weighted moving average of past coherence scores. Provides a longitudinal signal on agent reliability without exposing raw decision history.

### Delta Embedding

64-byte quantized vector generated on high-coherence decisions. Used for self-coaching and pattern reinforcement in the agent’s reasoning spine.

-----

## Evidentiary Architecture

AgDR-Phoenix is built to satisfy two audiences simultaneously: the security team and the courtroom.

- **BLAKE3 + Ed25519 signing** for cryptographic integrity
- **Forward-secret Merkle tree chaining** across the record sequence
- **WAL (Write-Ahead Log)** for persistence guarantees before acknowledgment
- **Automatic degradation** governed by the Limitations Act 2002 (Ontario): 2-year basic period, 15-year ultimate period
- **Canada Evidence Act** alignment for domestic admissibility; designed for portability to EU AI Act (August 2026) and common law jurisdictions

-----

## Distinction from Structural Monitoring

AgDR captures **semantic decision provenance**: what the agent decided, why, under what authority, and with what confidence, sealed at the moment of inference.

Structural container monitoring (e.g., the AGA standard) records whether the system was running correctly. AgDR records what the system actually decided. Both serve different functions in an AI governance stack.

-----

## Specification and Standards

|Resource                          |Location                                                                           |
|----------------------------------|-----------------------------------------------------------------------------------|
|AgDR Full Specification v0.2      |ISBN 978-1-7389042-0-4                                                             |
|Reasoning Capture Methodology v1.0|ISBN 978-1-7389042-1-1                                                             |
|License                           |Apache 2.0 OR CC0 1.0 (royalty-free anchor clause in GGF articles of incorporation)|

-----

## License

Dual-licensed under [Apache 2.0](https://spdx.org/licenses/Apache-2.0) and [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

The AgDR open standard is stewarded by the [Genesis Glass Foundation](https://accountability.ai) (Fondation Genese Cristal), a federally incorporated Canadian not-for-profit. The royalty-free grant is anchored in the Foundation’s articles of incorporation.

-----

## Contact

Genesis Glass Foundation  
founding@accountability.ai 
https://accountability.ai
