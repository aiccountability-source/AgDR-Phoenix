mkdir -p agdr_aki
cat > agdr_aki/__init__.py << 'EOF'
from .models import PPPTriplet, HumanDeltaChain
from .engine import AKIEngine

__version__ = "1.8.0"
__all__ = ["AKIEngine", "PPPTriplet", "HumanDeltaChain"]
EOF
