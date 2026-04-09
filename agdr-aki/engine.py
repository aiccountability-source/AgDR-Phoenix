cat > agdr_aki/engine.py << 'EOF'
from pathlib import Path
from typing import Any, Dict, Optional

from .models import CaptureRequest, PPPTriplet, HumanDeltaChain, SealedRecord
from .rust_bindings import PhoenixKernel

class AKIEngine:
    def __init__(self, db_path: str | Path = "agdr_aki.db", enable_gpu: bool = False):
        self.db_path = Path(db_path)
        self._rust_kernel = PhoenixKernel(str(self.db_path), enable_gpu)
        self._coherence_threshold = 0.92

    def capture(
        self,
        ctx: Dict[str, Any],
        prompt: str,
        reasoning_trace: Dict[str, Any],
        output: str,
        ppp_triplet: PPPTriplet,
        human_delta_chain: Optional[HumanDeltaChain] = None,
        auto_insight: bool = True,
    ) -> SealedRecord:
        request = CaptureRequest(
            ctx=ctx,
            prompt=prompt,
            reasoning_trace=reasoning_trace,
            output=output,
            ppp=ppp_triplet,
            human_delta_chain=human_delta_chain,
        )

        raw_record = self._rust_kernel.capture(
            request.model_dump_json(),
            auto_insight
        )
        return SealedRecord.model_validate_json(raw_record)

    @property
    def public_key(self) -> str:
        return self._rust_kernel.public_key_hex()
EOF
