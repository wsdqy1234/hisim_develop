from hisim.time_predictor.base import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch,
)

# AIC is an optional import: cache/sparse-KV validation runs use the
# dependency-free ConstantTimePredictor and only need the AIC *database* for
# the pool-capacity estimate, not for importing hisim.
try:
    from hisim.time_predictor.aiconfigurator import (
        AIConfiguratorTimePredictor,
    )
except ImportError:  # pragma: no cover - exercised in minimal containers
    AIConfiguratorTimePredictor = None

from hisim.time_predictor.constant import ConstantTimePredictor


__all__ = (
    FakeRequest,
    ScheduleBatch,
    InferTimePredictor,
    AIConfiguratorTimePredictor,
    ConstantTimePredictor,
)
