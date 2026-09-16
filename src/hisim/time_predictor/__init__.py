from hisim.time_predictor.base import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch,
)

try:
    from hisim.time_predictor.aiconfigurator import (
        AIConfiguratorTimePredictor,
    )
except ImportError:
    AIConfiguratorTimePredictor = None

try:
    from hisim.time_predictor.k3_predictor import K3ProfilePredictor
except ImportError:
    K3ProfilePredictor = None


__all__ = (
    "FakeRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "AIConfiguratorTimePredictor",
    "K3ProfilePredictor",
)
