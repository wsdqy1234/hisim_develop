"""K3 profile-based inference time predictor (V3).

Uses measured profiling data from 23-layer reduced model on Ascend 910C (TP=8).
All data from e2e_latency (sglang meta_info, no NPU profiler), consistent with
cache bandwidth model.

Prefill: 8-point lookup + linear interpolation
  Sources: supplement_profiling_results.json + v2 + v3 + gap_experiment_results.json
  Points: 1.5K/4K/6K/16K/25K/65K/128K/256K tokens
  Max interpolation gap: 65K tokens (was 105K in V2)

Decode: prefill-corrected pure decode times (output=512)
  - Base per-step at bs=1: 80.5ms (average of 6 measurements, seq=144 to 65K)
  - Batch ratio: nearly flat [1.0, 1.05, 1.02, 1.05, 1.10] for bs=[1,4,8,16,32]
    Old V2 ratios [1.0, 1.23, 1.29, 2.21, 3.92] were 85% prefill contamination artifacts
  - Seq factor: nearly flat [1.0, ..., 1.10, 1.20] for seq=[4K, ..., 32K, 65K]
    Old V2 factors [1.44, 1.0, 0.797, ...] were wrong direction (decreasing vs actual flat)

Data sources:
  v1: supplement_profiling_results.json (2K/8K/32K/256K prefill, OL512 decode)
  v2: supplement_profiling_v2_results.json (short-seq decode, low-token prefill)
  v3: supplement_profiling_v3_results.json (4K/16K/64K prefill, high-seq decode)
  gap: gap_experiment_results.json (128K prefill + prefix cache validation)

Known limitations:
- Only 23-layer reduced model (not full 93-layer)
- Only TP=8, disable-cuda-graph config
- Only W4A8+bf16 quantization
- Layer scaling not implemented (predictor returns 23L times)
"""
from hisim.time_predictor.base import InferTimePredictor, ScheduleBatch
from hisim.utils import get_logger

logger = get_logger("hisim")

import numpy as np


class K3ProfilePredictor(InferTimePredictor):
    """Predict inference time for Kimi-K3-23L (reduced, TP=8) on Ascend 910C.

    Prefill: 8-point lookup + linear interpolation (e2e, no profiler).
    Decode: base × batch_ratio × seq_factor (all from no-profiler, output=512).
    """

    # ===== Prefill (e2e from meta_info/GAP, no profiler, 23L) =====
    # 8 points: max interpolation gap = 65K tokens (was 105K in V2)
    # Sources: v1 (2K/8K/32K/256K), v3 (4K/16K/64K), gap (128K)
    PREFILL_TOKS = [1576, 4096, 6302, 16373, 25207, 65480, 131072, 201654]
    PREFILL_TIMES = [0.439, 0.725, 1.140, 2.716, 4.332, 11.463, 24.808, 39.957]

    # ===== Decode model (all no-profiler, output=512, prefill-corrected) =====
    # Pure decode per-step at bs=1 (average of 6 measurements, seq=144 to 65K):
    #   144 tok: 81.9ms, 2229 tok: 79.3ms, 3151 tok: 80.2ms,
    #   8901 tok: 81.6ms, 16373 tok: 81.8ms, 32771 tok: 80.7ms, 65480 tok: 77.8ms
    # Average: 80.5ms, variance: 4.1ms -> essentially flat
    DECODE_BASE = 0.0805  # seconds

    # Batch scaling ratios (prefill-corrected, output=512, no profiler)
    # V2/V3 data after removing prefill contamination:
    #   bs=1:  80.5ms (ratio 1.00)
    #   bs=4:  84.6ms (ratio 1.05) — V3 BS4_4KT
    #   bs=8:  80.7-88.5ms (ratio 1.00-1.10) — V3 BS8, varies with seq
    #   bs=16: ~85ms  (ratio 1.05) — interpolated (no clean data)
    #   bs=32: 88.5ms (ratio 1.10) — V3 BS32_4KT
    # Old V2 ratios [1.0, 1.23, 1.29, 2.21, 3.92] were 85% prefill contamination
    BATCH_RATIOS = [1.0, 1.05, 1.05, 1.08, 1.10]
    BATCH_SIZES = [1, 4, 8, 16, 32]

    # Seq factor: FLAT at 1.0 for all seq_lens
    # bs=1 data (7 points, 144 to 65K tokens) shows 77.8-81.9ms (flat, variance 4.1ms)
    # bs=8 shows mild increase at 32K (88.5ms vs 80.7ms at 16K) but
    # this is a batch×seq interaction, not a pure seq effect. Kept flat
    # to avoid overestimating at bs=1 (where 90% of simulation happens).
    SEQ_TOKS = [1024, 16384, 32768, 65536, 131072]
    SEQ_FACTORS = [1.0, 1.0, 1.0, 1.0, 1.0]

    # ===== Model metadata =====
    # Layer scaling (23L -> 93L), NOT applied by default
    MLA_SCALE_93 = 24 / 5     # 4.80
    KDA_SCALE_93 = 69 / 18   # 3.83
    MOE_SCALE_93 = 92 / 22   # 4.18
    TP_SCALE_93 = 93 / 23    # 4.04

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        """Return inference time in seconds. Negative = error."""
        if batch.is_empty():
            return 0.0
        if batch.is_prefill():
            return self._predict_prefill(batch)
        else:
            return self._predict_decode(batch)

    def _predict_prefill(self, batch: ScheduleBatch) -> float:
        """Prefill time: 8-point lookup + linear interpolation by token count."""
        total_time = 0.0
        for req in batch.reqs:
            seq_len = req.input_length
            t = self._interp(self.PREFILL_TOKS, self.PREFILL_TIMES, seq_len)
            total_time += t
        return total_time

    def _predict_decode(self, batch: ScheduleBatch) -> float:
        """Decode step time: base × batch_ratio × seq_factor.

        All parameters from no-profiler, output=512, prefill-corrected data.
        """
        bs = batch.batch_size
        max_seq = max((req.past_kv_length for req in batch.reqs), default=4096)
        if max_seq == 0:
            max_seq = 4096

        batch_ratio = self._interp(self.BATCH_SIZES, self.BATCH_RATIOS, bs)
        seq_factor = self._interp(self.SEQ_TOKS, self.SEQ_FACTORS, max_seq)
        return self.DECODE_BASE * batch_ratio * seq_factor

    @staticmethod
    def _interp(x_arr: list, y_arr: list, x: float) -> float:
        """Linear interpolation with clamping."""
        x_arr = list(x_arr)
        y_arr = list(y_arr)
        if x <= x_arr[0]:
            return y_arr[0]
        if x >= x_arr[-1]:
            return y_arr[-1]
        for i in range(len(x_arr) - 1):
            if x_arr[i] <= x <= x_arr[i + 1]:
                x0, x1 = x_arr[i], x_arr[i + 1]
                y0, y1 = y_arr[i], y_arr[i + 1]
                ratio = (x - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return y_arr[-1]
