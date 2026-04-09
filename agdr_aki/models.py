cat > agdr_aki/models.py << 'EOF'
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Optional
import uuid
from datetime import datetime

class PPPTriplet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    provenance: str
    place: str
    purpose: str

class HumanDeltaChain(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chain_id: str = Field(default_factory=lambda: f"hdc_{uuid.uuid4().hex[:12]}")
    agent_decision_ref: str
    resolved: bool = False
    terminal_node: str = Field(..., pattern="^(autonomous|human|escalated|terminal)$")

class CaptureRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    ctx: Dict[str, Any]
    prompt: str
    reasoning_trace: Dict[str, Any]
    output: str
    ppp: PPPTriplet
    human_delta_chain: Optional[HumanDeltaChain] = None

class SealedRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    timestamp: datetime
    hash: str
    signature: bytes
    ppp: PPPTriplet
    ctx: Dict[str, Any]
    prompt: str
    reasoning_trace: Dict[str, Any]
    output: str
    merkle_root: str
    coherence_score: float
    reputation_scalar: float
    human_delta_chain: Optional[HumanDeltaChain] = None
EOF
