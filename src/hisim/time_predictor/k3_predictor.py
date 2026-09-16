"""K3 profile-based inference time predictor (V2).

Uses measured profiling data from 23-layer reduced model on Ascend 910C (TP=8).
Implements lookup + linear interpolation across two measurement domains:

Domain 1 (prefill): e2e from sglang meta_info (no profiler) — 5 points
  - 2K(1576tok, 0.439s), 8K(6302tok, 1.140s), 32K(25207tok, 4.332s),
    128K(131072tok, 24.808s GAP), 256K(201654tok, 39.957s)
  - These are wall-clock-like times (no NPU profiler), consistent with cache bandwidth model

Domain 2 (decode): mixed sources
  - Base per-step at seq=4K: 81.4ms (from OL512, prefill amortized, no profiler)
  - Batch scaling: B1-B5 (profiler) + BS8/BS16 (no profiler)
  - Seq scaling: D1-D5 step_trace ratios (profiler, only used as ratios)

Known limitations:
- Only 23-layer reduced model (not full 93-layer)
- Only TP=8, disable-cuda-graph config
- Only W4A8+bf16 quantization
- 5 prefill points (sparse, 256K gap between 128K-256K)
- Batch×seq matrix is sparse (4 new points + 5 B1-B5)
- Layer scaling not implemented (predictor returns 23L times)
"""
from hisim.time_predictor.base import InferTimePredictor, ScheduleBatch
from hisim.utils import get_logger

logger = get_logger("hisim")

import numpy as np


class K3ProfilePredictor(InferTimePredictor):
    """Predict inference time for Kimi-K3-23L (reduced, TP=8) on Ascend 910C.

    Prefill: 5-point lookup + linear interpolation (e2e, no profiler).
    Decode: batch×seq 2D lookup (mixed sources, no profiler domain preferred).
    """

    # ===== Prefill (e2e from meta_info/GAP, no profiler, 23L) =====
    # Sources: supplement_profiling_results.json + gap_experiment_results.json
    # token counts are actual prompt_tokens (not nominal seq_len)
    PREFILL_TOKS = [1576, 6302, 25207, 131072, 201654]
    PREFILL_TIMES = [0.439, 1.140, 4.332, 24.808, 39.957]

    # ===== Decode model =====
    # Strategy: use no-profiler data as primary, profiler B1-B5 ratios as fallback
    #
    # No-profiler data points (actual deployment timing):
    #   bs=1, seq=4K:  81.4ms (OL512, prefill amortized)
    #   bs=8, seq=1K:  179.1ms (BS8_1K)
    #   bs=8, seq=4K:  124.4ms (BS8_4K)
    #   bs=8, seq=16K:  99.1ms (BS8_16K)
    #   bs=16, seq=4K: 155.7ms (BS16_4K)
    #
    # Profiler B1-B5 (disable-cuda-graph, seq=15751) — used as batch RATIO only:
    #   B1:B2:B3:B4:B5 = 1.0 : 1.23 : 1.29 : 2.21 : 3.92

    # Base decode per-step at bs=1, seq=4K (no profiler, prefill amortized)
    DECODE_BASE = 0.0814  # seconds

    # Batch scaling ratios (from B1-B5, domain-independent)
    BATCH_RATIOS = [1.0, 1.23, 1.29, 2.21, 3.92]
    BATCH_SIZES_RATIO = [1, 4, 8, 16, 32]

    # No-profiler 2D data points: (bs, seq) -> per_step (seconds)
    NOPROF_2D = {
        (1, 4096): 0.0814,
        (8, 1024): 0.1791,
        (8, 4096): 0.1244,
        (8, 16384): 0.0991,
        (16, 4096): 0.1557,
    }

    # Seq scaling for bs=8 (from no-profiler BS8 data)
    # seq=1K: 179.1ms, seq=4K: 124.4ms, seq=16K: 99.1ms
    # Trend: shorter seq = higher per-step (prefill overhead not amortized)
    BS8_SEQS = [1024, 4096, 16384]
    BS8_TIMES = [0.1791, 0.1244, 0.0991]
    BS8_REF_SEQ = 4096  # reference seq for batch ratio

    # Seq scaling ratio (relative to seq=4K)
    # From D1-D5 step_trace ratios (domain-independent):
    # seq=1K: 0.976, seq=4K: 1.013, seq=16K: 0.999, seq=64K: 1.104, seq=128K: 1.026
    # But BS8 data shows stronger seq effect: seq=1K/seq=4K = 179.1/124.4 = 1.44
    # Use BS8 data for seq effect (more representative)
    SEQ_FACTORS_TOKS = [1024, 4096, 16384, 65536, 131072]
    SEQ_FACTORS = [1.44, 1.0, 0.797, 0.797, 0.797]  # based on BS8, flat beyond 16K

    # ===== Model metadata =====
    # Layer scaling (23L → 93L), NOT applied by default
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
        """Prefill time: lookup + linear interpolation by actual token count."""
        total_time = 0.0
        for req in batch.reqs:
            # Use input_length as proxy for token count
            # (HiSim ScheduleBatch doesn't have exact prompt_tokens, use input_length)
            seq_len = req.input_length
            t = self._interp(self.PREFILL_TOKS, self.PREFILL_TIMES, seq_len)
            total_time += t
        return total_time

    def _predict_decode(self, batch: ScheduleBatch) -> float:
        """Decode step time: 2D model from no-profiler data + B1-B5 ratios.

        Strategy:
        1. Check no-profiler 2D table for exact (bs, seq) match
        2. Otherwise: base(bs=1, seq=4K) × batch_ratio(bs) × seq_factor(seq)
        """
        bs = batch.batch_size
        max_seq = max((req.past_kv_length for req in batch.reqs), default=4096)
        if max_seq == 0:
            max_seq = 4096

        # 1. Check 2D table for exact match
        key = (bs, max_seq)
        if key in self.NOPROF_2D:
            return self.NOPROF_2D[key]

        # 2. For bs=8, use BS8 seq interpolation directly
        if bs == 8:
            base_8 = self._interp(self.BS8_SEQS, self.BS8_TIMES, max_seq)
            return base_8

        # 3. General: base(81.4ms) × batch_ratio × seq_factor
        batch_ratio = self._interp(self.BATCH_SIZES_RATIO, self.BATCH_RATIOS, bs)
        seq_factor = self._interp(self.SEQ_FACTORS_TOKS, self.SEQ_FACTORS, max_seq)
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
