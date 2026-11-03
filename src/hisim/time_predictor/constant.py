"""Small deterministic predictor used by offline validation runs.

Cache/sparse-KV semantics do not depend on a latency database.  A constant
predictor lets a weight-free mock model exercise the complete SGLang scheduler
and cache pipeline when an AIC database is not mounted in the container.  It is
not intended as a performance model for production capacity planning.
"""

from __future__ import annotations

from hisim.time_predictor.base import InferTimePredictor, ScheduleBatch


class ConstantTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model,
        hw,
        config,
        prefill_ms: float = 1.0,
        decode_ms: float = 0.1,
        prefill_us_per_token: float = 0.0,
        decode_us_per_token: float = 0.0,
    ):
        super().__init__(model, hw, config)
        self.prefill_s = max(0.0, float(prefill_ms)) / 1000.0
        self.decode_s = max(0.0, float(decode_ms)) / 1000.0
        self.prefill_per_token_s = max(0.0, float(prefill_us_per_token)) / 1e6
        self.decode_per_token_s = max(0.0, float(decode_us_per_token)) / 1e6

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_decode():
            return self.decode_s + self.decode_per_token_s * batch.batch_size
        return self.prefill_s + self.prefill_per_token_s * batch.num_context_tokens
