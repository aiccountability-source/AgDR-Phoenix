# Changelog

##Release 1.8.4 - Python 3.13 & 3.14 Support

This release adds official support for Python 3.13 and 3.14. No code changes were required. 
The update primarily involves updating the Stable ABI target in Cargo.toml to abi3-py39 for forward compatibility and declaring support for the new Python versions in pyproject.toml by widening the requires-python range and adding the appropriate Trove classifiers.


## [1.8.0] - 2026-04-08

### Added
- Full Rust kernel with PyO3 bindings for high-performance atomic capture.
- Sensory spine with exponential weighting for historical decision context.
- Coherence scoring with Kalman-enhanced hysteresis.
- Reputation scalar via EWMA.
- Delta embedding (64-byte quantized) for self-coaching.
- HumanDeltaChain (FOI) for human-in-the-loop and evidentiary standing.
- PPP Triplet as the required immutable anchor.
- Dual licensing: Apache-2.0 OR CC0-1.0.
- Polished documentation emphasizing latency and contemporaneous evidence value.

### Changed
- Transitioned to a lean Rust core for minimal dependencies and sub-microsecond hot path.
- Focused on production readiness while preserving core AgDR principles of tamper-evidence and truth-seeking.

### Notes
Phoenix v1.8.0 delivers a lean, tamper-evident Atomic Kernel Inference SDK suitable for real-world agents (insurance adjudication, grid balancing, autonomous systems, etc.). It builds upon the AgDR vision with a strong emphasis on contemporaneous sealing and human oversight.
