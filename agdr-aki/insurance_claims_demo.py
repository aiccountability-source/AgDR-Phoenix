from agdr_aki import AKIEngine, PPPTriplet, HumanDeltaChain

engine = AKIEngine("insurance_demo.db")

record = engine.capture(
    ctx={"claim_id": "CLM-12345"},
    prompt="Process insurance claim #12345",
    reasoning_trace={"steps": ["analyze", "evaluate", "decide"]},
    output="Approve with standard conditions",
    ppp_triplet=PPPTriplet(
        provenance="Phoenix v1.8 on ACME Insurance Agent",
        place="Toronto, Ontario, CA - PIPEDA",
        purpose="Truth-seeking decision under AgDR standard",
    ),
    human_delta_chain=HumanDeltaChain(
        agent_decision_ref="aki_auto_001",
        resolved=True,
        terminal_node="autonomous"
    )
)

print("Sealed record:", record.id)
print("Coherence:", record.coherence_score)
print("Reputation:", record.reputation_scalar)
