from .agdr_aki import PhoenixKernel
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PPPTriplet:
    provenance: str
    place: str
    purpose: str

@dataclass
class HumanDeltaChain:
    agent_decision_ref: str
    resolved: bool
    terminal_node: str
    chain_id: str = field(default_factory=lambda: __import__('uuid').uuid4().hex)

class SealedRecord:
    def __init__(self, data: dict):
        self.__dict__.update(data)

class AKIEngine:
    def __init__(self, wal_path: str, enable_gpu: bool = False):
        self._kernel = PhoenixKernel(wal_path, enable_gpu)

    def capture(
        self,
        ctx: dict,
        prompt: str,
        reasoning_trace: dict,
        output: str,
        ppp_triplet: PPPTriplet,
        human_delta_chain: Optional[HumanDeltaChain] = None,
        auto_insight: bool = True,
    ) -> SealedRecord:
        payload = {
            "ctx": ctx,
            "prompt": prompt,
            "reasoning_trace": reasoning_trace,
            "output": output,
            "ppp": {
                "provenance": ppp_triplet.provenance,
                "place": ppp_triplet.place,
                "purpose": ppp_triplet.purpose,
            },
            "human_delta_chain": {
                "chain_id": human_delta_chain.chain_id,
                "agent_decision_ref": human_delta_chain.agent_decision_ref,
                "resolved": human_delta_chain.resolved,
                "terminal_node": human_delta_chain.terminal_node,
            } if human_delta_chain else None,
        }
        result = self._kernel.capture(json.dumps(payload), auto_insight)
        return SealedRecord(json.loads(result))
