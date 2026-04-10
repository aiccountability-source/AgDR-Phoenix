from agdr_aki import AKIEngine, PPPTriplet, HumanDeltaChain

def test_capture():
    engine = AKIEngine(":memory:")
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
    print("✅ Basic capture test passed")
