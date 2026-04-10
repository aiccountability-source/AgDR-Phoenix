from agdr_aki import AKIEngine, PPPTriplet, HumanDeltaChain

def test_capture():
    # One argument, not two
    engine = AKIEngine(":memory:")

    # Real objects, not a JSON string
    ppp = PPPTriplet(
        provenance="test",
        place="Toronto",
        purpose="test"
    )

    hdc = HumanDeltaChain(
        agent_decision_ref="aki_001",
        resolved=True,
        terminal_node="autonomous"
    )

    # Returns a SealedRecord object, not JSON
    record = engine.capture(
        ctx={},
        prompt="test",
        reasoning_trace={},
        output="test",
        ppp_triplet=ppp,
        human_delta_chain=hdc
    )

    assert record.id.startswith("aki_")
    assert 0.0 <= record.coherence_score <= 1.0
    assert record.reputation_scalar >= 0.0
    assert len(record.hash) > 0
    assert len(record.merkle_root) > 0
    print("✅ Basic capture test passed")


def test_public_key():
    engine = AKIEngine(":memory:")
    key = engine.public_key_hex()
    assert len(key) == 64  # 32 bytes as hex
    print("✅ Public key test passed")


if __name__ == "__main__":
    test_capture()
    test_public_key()
