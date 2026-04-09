from agdr_aki import AKIEngine, PPPTriplet

def test_capture():
    engine = AKIEngine(":memory:")
    ppp = PPPTriplet(
        provenance="test",
        place="Toronto",
        purpose="test"
    )
    record = engine.capture(
        ctx={},
        prompt="test",
        reasoning_trace={},
        output="test",
        ppp_triplet=ppp
    )
    assert record.id.startswith("aki_")
    assert 0.0 <= record.coherence_score <= 1.0
    print("✅ Basic test passed")
