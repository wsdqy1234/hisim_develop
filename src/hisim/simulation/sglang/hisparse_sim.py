"""HiSparse (hierarchical sparse KV) simulation for hisim.

This module is an abstract decode-side model of hierarchical sparse KV on top
of the hisim replay loop.  It can run with sglang's scheduler hooks, but the
profiles here describe the vLLM-Ascend/official DSV4.1 byte layout, not a
byte-for-byte model of sglang's current C4/C128 pools.  Real indexer Top-K IDs,
CUDA events and ``sgl_kernel.kvcacheio`` traces are still required for
calibration.

The implemented hierarchy is:

* The scheduler-visible KV pool is the *logical* pool.  Its capacity is
  ``physical_pool_tokens * host_to_device_ratio`` because the full sequence
  lives in pinned host DRAM.
* HBM only holds a fixed *hot* pool.  Every running request reserves
  ``padded_buffer_size = device_buffer_size + page_size`` slots, so the number
  of concurrently resident requests is bounded by
  ``physical_pool_tokens // padded_buffer_size``.
* At every decode step the indexer selects ``top_k`` compressed positions.
  Selection cadence is independent of compression and backup cadence.  Misses
  are copied host->device on the critical path; hits can still read index
  payload.  The regular hot slots use hit-to-MRU LRU, and the newest token has
  a separate reserved slot outside that LRU.  On the direct-short path, where
  the whole compressed sequence fits the device buffer, the previous newest
  entry enters the regular LRU without an H2D transfer.  On the long-sequence
  swap path it stays outside the regular LRU and must be copied back if the
  next Top-K selects it again.
* Prefill KV is staged device->host once.  Reused prefix tokens are not staged
  again.  Each newly decoded compressed entry is eagerly backed up device->host
  on a side stream (hidden from the critical path by default, but the bytes and
  duration are accounted).

The real indexer scores are not observable in a trace replay, so the top-k
access stream is a configurable abstraction (see :class:`HiSparseSimConfig`).
``sticky`` keeps a working set whose non-recent part churns by ``churn`` per
step.  This remains a synthetic access model; calibration must use real
hot-buffer hit/miss and swap-in counters.

Per-ratio working sets aggregate the source layers of that ratio.  The profile
gains optional source-layer main/index splits for per-channel IO accounting.
When ``shared_index_freq > 1`` the non-anchor IO is approximated as prefetched
off the critical path; anchor/follower events still need real traces.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from hisim.utils import get_logger

logger = get_logger("hisim")


# ---------------------------------------------------------------------------
# Static KV layout profiles (L0/L1/L2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVProfile:
    """Static KV layout for a model family.

    The legacy HiSparse model assumes a uniform layout
    (``num_layers * bytes_per_token_per_layer``).  Models such as
    DeepSeek-V4.1 store a much smaller KV (FP4 + cross-layer reuse), so that
    formula overestimates capacity and IO by orders of magnitude.

    ``offload_bytes_per_token`` is the **total** per-token KV footprint of the
    offloadable source layers.  ``source_layer_ratios`` splits that total over
    the source layers; a source layer with ``ratio=r`` produces one compressed
    entry every ``r`` tokens, so a compressed entry costs
    ``per_layer_bytes * r``.

    L0 uses only ``offload_bytes_per_token`` for capacity and total IO.  L1
    additionally uses ``source_layer_ratios`` to run one compressed working
    set per ratio group.  L2 uses the optional fields below to model the
    deployed per-source-layer layout, page/allocator geometry, indexer,
    SWA/compressor residency and the admission upper bound.

    ``bytes_scope`` controls whether the byte fields are already **per rank**
    (``"per_rank"``) or a model/global value that is divided by ``tp_size``
    (``"global"``).  This matters for MLA: with DCP=1 and one KV head per rank,
    the eight TP ranks hold replicas and the per-rank bytes must not be divided
    again.
    """

    name: str = "generic"
    # The profiles in this module describe the vLLM-Ascend/official layout.
    # They are not a byte-for-byte model of sglang's current C4/C128 pools.
    runtime_target: str = "vllm_ascend"
    # Deployment concurrency used by the admission/steady-state formulas when
    # the server does not provide an explicit max_running_requests.
    deployment_max_num_reqs: int = 0
    # Initial regular hot-buffer contents: earliest | latest | empty.
    hot_buffer_init: str = "earliest"
    offload_bytes_per_token: float = 0.0
    # Ratios of the KV source layers, e.g. (2, 2, 2, 1) for DSV4.1.
    source_layer_ratios: Tuple[int, ...] = ()
    # (layer_count, compress_ratio) for metadata, e.g. ((2, 0), (18, 2), (20, 1)).
    layer_groups: Tuple[Tuple[int, int], ...] = ()
    # sglang's index_topk: number of compressed positions selected per step.
    index_topk: int = 0
    # L2: HBM-resident indexer pool, bytes per token.  This is *separate* from
    # ``offload_bytes_per_token``; use 0 when the index is embedded in the
    # source-layer payload (as in the .22 BF16 profile).
    indexer_bytes_per_token: float = 0.0
    # L2: SWA ring, per-request window (tokens) and total bytes per token
    # across all SWA layers.
    swa_window_tokens: int = 0
    swa_bytes_per_token: float = 0.0

    # -- L2 topology / layout metadata -------------------------------------
    # "global" divides byte fields by tp_size; "per_rank" does not.
    bytes_scope: str = "global"
    source_layer_ids: Tuple[int, ...] = ()
    # Per original token, per source layer, e.g. (577, 577, 577, 1154).
    source_layer_bytes_per_token: Tuple[float, ...] = ()
    # Optional split of each source-layer payload into main KV and index bytes.
    # BF16 .22: main (512,512,512,1024), index (65,65,65,130).
    source_layer_main_kv_bytes_per_token: Tuple[float, ...] = ()
    source_layer_index_bytes_per_token: Tuple[float, ...] = ()
    # ((source_layer, (consumer_layer, ...)), ...).
    source_consumers: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()
    index_storage_layers: Tuple[int, ...] = ()
    index_compute_layers: Tuple[int, ...] = ()
    # Index payload embedded in offload bytes, for reporting only.
    index_bytes_per_token: float = 0.0
    main_kv_bytes_per_token: float = 0.0
    scheduler_block_tokens: int = 0
    page_bytes: int = 0
    slot_count: int = 0
    allocator_bytes_per_token: float = 0.0
    payload_bytes_per_block: float = 0.0
    candidate_blocks: int = 0
    candidate_block_tokens: int = 0
    candidate_source_layer: int = -1
    candidate_filter_layers: Tuple[int, ...] = ()
    swa_layers: int = 0
    # Token window used by the admission formula; includes in-flight prefill.
    swa_admission_tokens: int = 0
    compressor_layers: Tuple[int, ...] = ()
    compressor_page_bytes: int = 0
    compressor_bytes_per_request: float = 0.0
    admission_swa_groups: int = 0
    admission_extra_groups: int = 0
    admission_bytes_per_block: int = 0

    @property
    def enabled(self) -> bool:
        return self.offload_bytes_per_token > 0

    @property
    def per_rank(self) -> bool:
        return self.bytes_scope.strip().lower() == "per_rank"

    @property
    def source_layer_bytes_sum(self) -> float:
        return float(sum(self.source_layer_bytes_per_token))

    @property
    def capacity_bytes_per_token(self) -> float:
        """Bytes occupied per original token in the HBM long-KV pool.

        The effective payload and the allocator footprint are different
        quantities.  When an allocator slope is known it wins; otherwise fall
        back to payload + separately resident index.  IO/staging still use
        ``offload_bytes_per_token`` and must not use this property.
        """
        if self.allocator_bytes_per_token > 0:
            return float(self.allocator_bytes_per_token)
        return float(self.offload_bytes_per_token + self.indexer_bytes_per_token)

    def io_groups(self) -> List[Tuple[int, float]]:
        """Return ``[(ratio, bytes_per_compressed_entry), ...]``.

        If ``source_layer_bytes_per_token`` is set, the byte weight of each
        source layer is used directly.  Otherwise the legacy behaviour splits
        ``offload_bytes_per_token`` evenly over the source layers.  For the
        FP4/CSA2 profile (890 B/token, ratios ``(2, 2, 2, 1)``) that yields
        ``[(1, 222.5), (2, 1335.0)]``; for the `.22` BF16 profile it yields
        ``[(1, 1154.0), (2, 3462.0)]``.
        """
        if not self.enabled:
            return []
        ratios = tuple(int(r) for r in self.source_layer_ratios if int(r) > 0)
        if not ratios:
            return [(1, float(self.offload_bytes_per_token))]

        if self.source_layer_bytes_per_token:
            if len(self.source_layer_bytes_per_token) == len(ratios):
                grouped: Dict[int, float] = {}
                for ratio, bytes_per_token in zip(
                    ratios, self.source_layer_bytes_per_token
                ):
                    grouped[ratio] = grouped.get(ratio, 0.0) + (
                        float(bytes_per_token) * ratio
                    )
                return [(ratio, grouped[ratio]) for ratio in sorted(grouped)]
            logger.warning(
                "KV profile %s has %d source ratios but %d per-layer byte "
                "values; falling back to an even split.",
                self.name,
                len(ratios),
                len(self.source_layer_bytes_per_token),
            )

        per_layer = float(self.offload_bytes_per_token) / len(ratios)
        groups: List[Tuple[int, float]] = []
        for ratio in sorted(set(ratios)):
            layer_bytes = sum(per_layer for r in ratios if r == ratio)
            groups.append((ratio, float(layer_bytes * ratio)))
        return groups

    def io_channel_groups(self) -> List[Tuple[int, float, float]]:
        """Return ``[(ratio, main_kv_bytes, index_payload_bytes), ...]``.

        The split is only emitted when the profile has explicit per-source
        main/index bytes.  Otherwise the entire entry is treated as main KV and
        index traffic remains an explicitly uncalibrated residual.
        """
        ratios = tuple(int(r) for r in self.source_layer_ratios if int(r) > 0)
        main = tuple(float(x) for x in self.source_layer_main_kv_bytes_per_token)
        index = tuple(float(x) for x in self.source_layer_index_bytes_per_token)
        if ratios and len(main) == len(ratios) and len(index) == len(ratios):
            grouped: Dict[int, Tuple[float, float]] = {}
            for ratio, main_b, index_b in zip(ratios, main, index):
                old_main, old_index = grouped.get(ratio, (0.0, 0.0))
                grouped[ratio] = (
                    old_main + main_b * ratio,
                    old_index + index_b * ratio,
                )
            return [
                (ratio, grouped[ratio][0], grouped[ratio][1])
                for ratio in sorted(grouped)
            ]
        return [(ratio, total, 0.0) for ratio, total in self.io_groups()]

    def admission_blocks(self, tokens: int) -> int:
        """Return the installed vLLM-Ascend admission block count.

        ``B(L) = ceil(L/block) + extra_groups +
                   swa_groups * (ceil(min(L, swa_admission_tokens)/block) + 1)``

        This is an **upper bound** for one request, not steady-state decode
        occupancy.  Profiles without an allocator layout return 0.
        """
        block = int(self.scheduler_block_tokens)
        if block <= 0 or int(self.admission_bytes_per_block) <= 0:
            return 0
        length = max(0, int(tokens))
        long_blocks = (length + block - 1) // block
        swa_tokens = self.swa_admission_tokens or self.swa_window_tokens or length
        swa_tokens = min(length, int(swa_tokens))
        swa_blocks = (swa_tokens + block - 1) // block if swa_tokens > 0 else 0
        return (
            long_blocks
            + int(self.admission_extra_groups)
            + int(self.admission_swa_groups) * (swa_blocks + 1)
        )

    def admission_bytes(self, tokens: int) -> int:
        blocks = self.admission_blocks(tokens)
        return int(blocks * int(self.admission_bytes_per_block))

    def admission_overhead_bytes(self, tokens: int) -> int:
        """Admission bytes not attributable to long-KV blocks."""
        block = int(self.scheduler_block_tokens)
        if block <= 0 or int(self.admission_bytes_per_block) <= 0:
            return 0
        length = max(0, int(tokens))
        long_blocks = (length + block - 1) // block
        blocks = self.admission_blocks(length)
        return int(max(0, blocks - long_blocks) * int(self.admission_bytes_per_block))

    @classmethod
    def from_value(cls, value) -> Optional["KVProfile"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in ("", "none", "legacy", "generic"):
                return None
            if key in ("dsv41", "dsv4.1", "deepseek_v41", "deepseek-v4.1"):
                return DSV41_PROFILE
            if key in (
                "dsv41_fp4_csa2",
                "dsv41-fp4",
                "dsv41_fp4",
                "fp4_csa2",
            ):
                return DSV41_FP4_PROFILE
            if key in (
                "dsv41_vllm_bf16",
                "dsv41-bf16",
                "dsv41_bf16",
                "vllm_bf16",
            ):
                return DSV41_VLLM_BF16_PROFILE
            raise ValueError(f"Unknown KV profile {value!r}")
        if isinstance(value, dict):
            raw = dict(value)
            int_tuple_fields = (
                "source_layer_ratios",
                "source_layer_ids",
                "index_storage_layers",
                "index_compute_layers",
                "candidate_filter_layers",
                "compressor_layers",
            )
            for key in int_tuple_fields:
                if key in raw:
                    raw[key] = tuple(int(x) for x in raw[key])
            for key in (
                "source_layer_bytes_per_token",
                "source_layer_main_kv_bytes_per_token",
                "source_layer_index_bytes_per_token",
            ):
                if key in raw:
                    raw[key] = tuple(float(x) for x in raw[key])
            if "layer_groups" in raw:
                raw["layer_groups"] = tuple(
                    (int(a), int(b)) for a, b in raw["layer_groups"]
                )
            if "source_consumers" in raw:
                consumers = raw["source_consumers"]
                items = consumers.items() if isinstance(consumers, dict) else consumers
                raw["source_consumers"] = tuple(
                    (int(source), tuple(int(x) for x in layers))
                    for source, layers in items
                )
            return cls(**raw)
        raise TypeError(f"Unsupported KV profile value: {type(value)!r}")


# Source -> consumer map from the installed .22 vLLM-Ascend layout.  It is a
# model-level topology, so the FP4/CSA2 profile can reuse it even though its
# byte layout is different.
DSV41_SOURCE_CONSUMERS: Tuple[Tuple[int, Tuple[int, ...]], ...] = (
    (2, (2, 3, 4, 5, 6, 7)),
    (8, (8, 9, 10, 11, 12, 13)),
    (14, (14, 15, 16, 17, 18, 19)),
    (20, tuple(range(20, 40))),
)

# DeepSeek-V4.1-Flash (config.json):
#   40 layers: 2 SWA (ratio 0) + 18 CSA (ratio 2) + 20 HCA (ratio 1)
#   kv_source_layer_ids=[2, 8, 14, 20] -> 3 x ratio 2 + 1 x ratio 1
#   index_topk=512, index_head_dim=128
#   official global KV footprint: 890 B/token (FP4 + CSA2 cross-layer reuse)
#
# This remains the L0/L1-compatible profile: byte layout is the official
# 890 B/token global figure, and L2 byte-only fields stay 0.  The structural
# fields are shared with the FP4/CSA2 L2 profile below.
DSV41_PROFILE = KVProfile(
    name="dsv41",
    runtime_target="vllm_ascend",
    offload_bytes_per_token=890.0,
    source_layer_ratios=(2, 2, 2, 1),
    layer_groups=((2, 0), (18, 2), (20, 1)),
    index_topk=512,
    indexer_bytes_per_token=0.0,  # set to 544 in dsv41_fp4_csa2
    swa_window_tokens=128,
    swa_bytes_per_token=0.0,  # unknown for the FP4/CSA2 target
    source_layer_ids=(2, 8, 14, 20),
    source_consumers=DSV41_SOURCE_CONSUMERS,
    index_storage_layers=(2, 8, 14, 20),
    index_compute_layers=(2, 8, 14, 20, 24, 28, 32, 36),
    candidate_blocks=2048,
    candidate_block_tokens=8,
    candidate_source_layer=20,
    candidate_filter_layers=(24, 28, 32, 36),
    scheduler_block_tokens=128,
    swa_layers=40,
    swa_admission_tokens=4223,
    compressor_layers=(2, 8, 14),
)

# FP4/CSA2 L2 approximation.  The model topology (source layers, consumers,
# index candidate path, SWA/compressor layers, 128-token scheduler block) is
# known; the absolute FP4 page/allocator and SWA/compressor bytes are not.
# ``indexer_bytes_per_token=544`` is the FP4 8-layer index assumption from the
# design notes, kept separate so it can be overridden by a custom dict.
DSV41_FP4_PROFILE = KVProfile(
    name="dsv41_fp4_csa2",
    runtime_target="vllm_ascend",
    offload_bytes_per_token=890.0,
    source_layer_ratios=(2, 2, 2, 1),
    layer_groups=((2, 0), (18, 2), (20, 1)),
    index_topk=512,
    indexer_bytes_per_token=544.0,
    swa_window_tokens=128,
    swa_bytes_per_token=0.0,
    bytes_scope="global",
    source_layer_ids=(2, 8, 14, 20),
    source_consumers=DSV41_SOURCE_CONSUMERS,
    index_storage_layers=(2, 8, 14, 20),
    index_compute_layers=(2, 8, 14, 20, 24, 28, 32, 36),
    candidate_blocks=2048,
    candidate_block_tokens=8,
    candidate_source_layer=20,
    candidate_filter_layers=(24, 28, 32, 36),
    scheduler_block_tokens=128,
    swa_layers=40,
    swa_admission_tokens=4223,
    compressor_layers=(2, 8, 14),
)

# .22 BF16 vLLM-Ascend layout (source/layout-derived, not live tensor
# introspection).  All byte fields are per rank; DCP=1 and each rank holds one
# KV head, so TP8 must not divide them again.
DSV41_VLLM_BF16_PROFILE = KVProfile(
    name="dsv41_vllm_bf16",
    runtime_target="vllm_ascend",
    deployment_max_num_reqs=8,
    hot_buffer_init="earliest",
    offload_bytes_per_token=2885.0,
    source_layer_ratios=(2, 2, 2, 1),
    layer_groups=((2, 0), (18, 2), (20, 1)),
    index_topk=512,
    indexer_bytes_per_token=0.0,  # index is embedded in 2885
    swa_window_tokens=128,
    swa_bytes_per_token=40 * 512 * 2,  # 40 layers, BF16, 512-wide
    bytes_scope="per_rank",
    source_layer_ids=(2, 8, 14, 20),
    source_layer_bytes_per_token=(577.0, 577.0, 577.0, 1154.0),
    source_layer_main_kv_bytes_per_token=(512.0, 512.0, 512.0, 1024.0),
    source_layer_index_bytes_per_token=(65.0, 65.0, 65.0, 130.0),
    source_consumers=DSV41_SOURCE_CONSUMERS,
    index_storage_layers=(2, 8, 14, 20),
    index_compute_layers=(2, 8, 14, 20, 24, 28, 32, 36),
    index_bytes_per_token=325.0,
    main_kv_bytes_per_token=2560.0,
    scheduler_block_tokens=128,
    page_bytes=131072,
    slot_count=4,
    allocator_bytes_per_token=4096.0,
    payload_bytes_per_block=369280.0,
    candidate_blocks=2048,
    candidate_block_tokens=8,
    candidate_source_layer=20,
    candidate_filter_layers=(24, 28, 32, 36),
    swa_layers=40,
    swa_admission_tokens=128 - 1 + 2 * 2048,
    compressor_layers=(2, 8, 14),
    compressor_page_bytes=32 * 1024 * 4,
    compressor_bytes_per_request=3 * 32 * 1024 * 4,
    admission_swa_groups=10,
    admission_extra_groups=1,
    admission_bytes_per_block=524288,
)


def profile_tp_scale(profile: KVProfile, tp_size: int = 1) -> float:
    """Scale a profile's byte fields to one rank."""
    return 1.0 if profile.per_rank else 1.0 / max(1, int(tp_size))


def physical_pool_tokens_for_budget(
    profile: KVProfile,
    budget_bytes: float,
    max_num_reqs: int,
    tp_size: int = 1,
) -> int:
    """Size the HBM hot pool from an available KV-byte budget.

    Long KV uses the allocator/capacity slope, while SWA and compressor state
    are subtracted separately.  This is intentionally the inverse of
    ``HiSparseSimulator.hbm_kv_bytes`` so e2e replay and analytic sweeps cannot
    silently use different denominators.  Return 0 when the budget cannot fit
    even one token for the requested request count.
    """
    scale = profile_tp_scale(profile, tp_size)
    reqs = max(1, int(max_num_reqs))
    per_req_overhead = (
        float(profile.swa_window_tokens) * float(profile.swa_bytes_per_token)
        + float(profile.compressor_bytes_per_request)
    ) * scale
    capacity_per_token = profile.capacity_bytes_per_token * scale
    if capacity_per_token <= 0:
        raise ValueError(
            f"HiSparse profile {profile.name!r} has no positive capacity slope"
        )
    remaining = float(budget_bytes) - reqs * per_req_overhead
    if remaining < capacity_per_token:
        # Do not silently round an infeasible pool up to one token.  The caller
        # needs to distinguish "no request fits" from a valid one-token pool.
        return 0
    return int(remaining // capacity_per_token)


def hot_pool_geometry_for_budget(
    profile: KVProfile,
    budget_bytes: float,
    requested_reqs: int,
    padded_buffer_size: int,
    tp_size: int = 1,
) -> Tuple[int, int]:
    """Return ``(physical_tokens, max_reqs)`` for one HBM budget.

    The number of request-overhead reservations and the physical-token count
    depend on each other: the hot-pool concurrency cap limits how many requests
    can actually run, while each running request consumes SWA/compressor bytes.
    Resolve that fixed point so the final ``max_reqs`` is the one used by the
    simulator and by ``HiSparseSimulator.hbm_kv_bytes``.  Raise ``ValueError``
    when the budget cannot hold one request's overhead plus a padded buffer.
    """
    requested = max(1, int(requested_reqs))
    padded = max(1, int(padded_buffer_size))

    def fits(reqs: int) -> bool:
        tokens = physical_pool_tokens_for_budget(
            profile, budget_bytes, reqs, tp_size=tp_size
        )
        return tokens >= padded and reqs <= tokens // padded

    if not fits(1):
        scale = profile_tp_scale(profile, tp_size)
        per_req_overhead = (
            float(profile.swa_window_tokens) * float(profile.swa_bytes_per_token)
            + float(profile.compressor_bytes_per_request)
        ) * scale
        minimum = (
            per_req_overhead
            + profile.capacity_bytes_per_token * scale * padded
        )
        raise ValueError(
            f"HiSparse profile {profile.name!r} cannot fit one request with "
            f"{float(budget_bytes):.0f} B: need at least {minimum:.0f} B "
            f"for SWA/compressor plus {padded} hot-pool tokens"
        )

    # Lower concurrency frees request overhead and can only increase the number
    # of hot-pool buffers, so feasibility is monotone.  Find the largest
    # concurrency not exceeding the requested cap instead of stopping at the
    # first clamp; otherwise a clamped value can waste budget and understate the
    # final scheduler limit.
    lo, hi = 1, requested
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    physical_tokens = physical_pool_tokens_for_budget(
        profile, budget_bytes, best, tp_size=tp_size
    )
    return physical_tokens, best


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HiSparseSimConfig:
    """Knobs for the HiSparse simulation.

    Defaults mirror sglang's HiSparse defaults where they exist:
    ``top_k=2048``, ``device_buffer_size=2*top_k`` and
    ``host_to_device_ratio=2``.  The access-model knobs have no sglang
    counterpart (the real values come from the indexer) and are documented as
    calibration parameters.
    """

    enable: bool = False
    # Number of KV positions selected by the DSA indexer per decode step.
    top_k: int = 2048
    # Per-request HBM buffer, in tokens.  0 means 2 * top_k.
    device_buffer_size: int = 0
    # host pool tokens = physical (HBM) pool tokens * this ratio.
    host_to_device_ratio: int = 2
    # Page granularity of the device/host pools.  padded_buffer_size is
    # device_buffer_size + page_size, matching hisparse_coordinator.py.
    page_size: int = 1
    # Access model: sticky | recency | uniform.
    access_model: str = "sticky"
    # Fraction of the non-recent working set replaced every decode step.
    churn: float = 0.05
    # Fraction of top_k that is always the most recent positions.
    recent_ratio: float = 0.5
    # Top-k selection cadence.  ``every_step`` models a changing query even
    # when no new compressed entry is produced.  ``compression_boundary`` is
    # the old sensitivity-only assumption that the selected set is frozen
    # between compression boundaries.
    selection_cadence: str = "every_step"
    # Initial regular hot buffer: "" follows profile/legacy default;
    # earliest | latest | empty override it.
    hot_buffer_init: str = ""
    # Read index payload for selected positions on hits as well as misses.
    # Only profiles with an explicit main/index split have index bytes.
    index_read_on_hit: bool = True
    # 1 = every layer does its own indexer/swap-in; N > 1 = one anchor layer
    # per group of N, the followers are prefetched (off the critical path).
    shared_index_freq: int = 1
    # Eager device->host backup of the newest token.  sglang runs it on a side
    # stream, so by default it is accounted but not added to the step latency.
    backup_on_critical: bool = False
    seed: int = 0
    # Static KV layout for models whose global KV is not
    # ``num_layers * bytes_per_token_per_layer`` (e.g. DeepSeek-V4.1).
    # None/"" keeps the legacy geometry exactly as before.
    kv_profile: Optional[KVProfile] = None

    def __post_init__(self) -> None:
        self.kv_profile = KVProfile.from_value(self.kv_profile)
        self.top_k = max(1, int(self.top_k))
        # The indexer physically cannot select more than index_topk positions.
        if self.kv_profile is not None and self.kv_profile.index_topk > 0:
            self.top_k = min(self.top_k, int(self.kv_profile.index_topk))
        if self.device_buffer_size <= 0:
            self.device_buffer_size = 2 * self.top_k
        self.device_buffer_size = max(self.top_k, int(self.device_buffer_size))
        self.host_to_device_ratio = max(1, int(self.host_to_device_ratio))
        self.page_size = max(1, int(self.page_size))
        self.churn = min(max(float(self.churn), 0.0), 1.0)
        self.recent_ratio = min(max(float(self.recent_ratio), 0.0), 1.0)
        self.shared_index_freq = max(1, int(self.shared_index_freq))
        if self.access_model not in ("sticky", "recency", "uniform"):
            logger.warning(
                "Unknown HiSparse access model %r, falling back to 'sticky'.",
                self.access_model,
            )
            self.access_model = "sticky"
        if self.selection_cadence not in ("every_step", "compression_boundary"):
            logger.warning(
                "Unknown selection cadence %r, falling back to 'every_step'.",
                self.selection_cadence,
            )
            self.selection_cadence = "every_step"
        if self.hot_buffer_init not in ("", "earliest", "latest", "empty"):
            logger.warning(
                "Unknown hot buffer init %r, falling back to profile default.",
                self.hot_buffer_init,
            )
            self.hot_buffer_init = ""

    @property
    def padded_buffer_size(self) -> int:
        # hisparse_coordinator.py: an extra page for the newest token.
        return self.device_buffer_size + self.page_size

    @property
    def critical_fraction(self) -> float:
        """Fraction of per-layer swap-in bytes that is on the critical path."""
        return 1.0 / self.shared_index_freq

    @classmethod
    def from_mapping(cls, raw: Optional[dict], env: Optional[dict] = None) -> "HiSparseSimConfig":
        raw = dict(raw or {})
        env = dict(env or {})
        aliases = {
            "enable": "enable",
            "top_k": "top_k",
            "device_buffer_size": "device_buffer_size",
            "host_to_device_ratio": "host_to_device_ratio",
            "page_size": "page_size",
            "access_model": "access_model",
            "churn": "churn",
            "recent_ratio": "recent_ratio",
            "selection_cadence": "selection_cadence",
            "hot_buffer_init": "hot_buffer_init",
            "index_read_on_hit": "index_read_on_hit",
            "shared_index_freq": "shared_index_freq",
            "backup_on_critical": "backup_on_critical",
            "seed": "seed",
            "kv_profile": "kv_profile",
        }
        kwargs = {}
        for key, attr in aliases.items():
            if key in raw and raw[key] is not None:
                kwargs[attr] = raw[key]
        for key, value in env.items():
            if value is not None:
                kwargs[aliases[key]] = value
        # Coerce values coming from JSON / environment strings.
        for key in ("top_k", "device_buffer_size", "host_to_device_ratio", "page_size",
                    "shared_index_freq", "seed"):
            if key in kwargs:
                kwargs[key] = int(kwargs[key])
        for key in ("churn", "recent_ratio"):
            if key in kwargs:
                kwargs[key] = float(kwargs[key])
        if "enable" in kwargs and not isinstance(kwargs["enable"], bool):
            kwargs["enable"] = str(kwargs["enable"]).strip().lower() in (
                "1", "true", "yes", "on",
            )
        for key in ("backup_on_critical", "index_read_on_hit"):
            if key in kwargs and not isinstance(kwargs[key], bool):
                kwargs[key] = str(kwargs[key]).strip().lower() in (
                    "1", "true", "yes", "on",
                )
        return cls(**kwargs)


def load_hisparse_config() -> HiSparseSimConfig:
    """Read the ``hisparse`` section of the hisim config, then apply env
    overrides.  Environment variables win so a single asset can be reused for
    baseline vs. sparse runs."""
    raw = {}
    try:
        from hisim.simulation.manager import Envs

        with open(Envs.config_path()) as f:
            config = json.load(f)
        raw = config.get("hisparse", {}) or {}
    except Exception:
        raw = {}

    env = {
        "enable": os.getenv("HISIM_HISPARSE"),
        "top_k": os.getenv("HISIM_HISPARSE_TOP_K"),
        "device_buffer_size": os.getenv("HISIM_HISPARSE_DEVICE_BUFFER_SIZE"),
        "host_to_device_ratio": os.getenv("HISIM_HISPARSE_HOST_TO_DEVICE_RATIO"),
        "page_size": os.getenv("HISIM_HISPARSE_PAGE_SIZE"),
        "access_model": os.getenv("HISIM_HISPARSE_ACCESS_MODEL"),
        "churn": os.getenv("HISIM_HISPARSE_CHURN"),
        "recent_ratio": os.getenv("HISIM_HISPARSE_RECENT_RATIO"),
        "selection_cadence": os.getenv("HISIM_HISPARSE_SELECTION_CADENCE"),
        "hot_buffer_init": os.getenv("HISIM_HISPARSE_HOT_BUFFER_INIT"),
        "index_read_on_hit": os.getenv("HISIM_HISPARSE_INDEX_READ_ON_HIT"),
        "shared_index_freq": os.getenv("HISIM_HISPARSE_SHARED_INDEX_FREQ"),
        "backup_on_critical": os.getenv("HISIM_HISPARSE_BACKUP_ON_CRITICAL"),
        "seed": os.getenv("HISIM_HISPARSE_SEED"),
        "kv_profile": os.getenv("HISIM_HISPARSE_KV_PROFILE"),
    }
    return HiSparseSimConfig.from_mapping(raw, env)


# ---------------------------------------------------------------------------
# Transfer timing
# ---------------------------------------------------------------------------


def transfer_duration(
    direction: str,
    num_bytes: float,
    read_bw: Optional[float] = None,
    write_bw: Optional[float] = None,
) -> float:
    """Same bandwidth model hisim uses for L2 host transfers:

        duration = t0 + bytes / (raw_bandwidth * 0.85)

    with t0 = 6.67us host->device and 4us device->host.
    """
    if num_bytes <= 0:
        return 0.0
    raw_bw = read_bw if direction == "H2D" else write_bw
    if raw_bw is None:
        try:
            from hisim.simulation.manager import ConfigManager

            pc = ConfigManager.get_platform_config()
            raw_bw = (
                pc.memory_read_bandwidth
                if direction == "H2D"
                else pc.memory_write_bandwidth
            )
        except Exception:
            raw_bw = None
    t0 = 6.67e-6 if direction == "H2D" else 4e-6
    if not raw_bw:
        return 0.0
    return t0 + num_bytes / (raw_bw * 0.85)


# ---------------------------------------------------------------------------
# Top-k access model
# ---------------------------------------------------------------------------


class _WorkingSet:
    """Canonical per-request top-k access stream.

    The selected set is represented as a rolling *recent* window (implicit:
    the last ``n_recent`` positions) plus a *sticky* set of older positions.
    Only the changes are materialised, so a step costs
    ``O(churn * top_k + new_tokens)`` instead of ``O(top_k)``.  The hot buffer
    is an LRU set of token positions with capacity ``padded_buffer_size``;
    positions no longer selected are evicted lazily, which is what gives the
    buffer its slack over a bare working set.
    """

    def __init__(
        self,
        top_k: int,
        capacity: int,
        access_model: str,
        churn: float,
        recent_ratio: float,
        rng: random.Random,
        *,
        hot_buffer_init: str = "latest",
        lru_mode: bool = False,
        newest_page: bool = False,
    ) -> None:
        self.top_k = top_k
        # Regular LRU slots only.  The newest-token page is modelled
        # separately and does not consume a general-purpose hot slot.
        self.capacity = capacity
        self.access_model = access_model
        self.churn = churn
        self.recent_ratio = recent_ratio
        self.rng = rng
        self.hot_buffer_init = hot_buffer_init
        self.lru_mode = lru_mode
        self.newest_page = newest_page
        self.sticky: Optional[set] = None
        self.hot: "OrderedDict[int, None]" = OrderedDict()
        self.prev_seq_len = 0
        self.prev_recent_start = 0
        self.prev_n = 0
        # The previous step's newest page.  Only the direct-short path keeps
        # its data resident outside the reserved page; on the long-sequence
        # swap path it is not automatically moved into the regular LRU.
        self.prev_newest: Optional[int] = None

    def _recent_count(self, n: int) -> int:
        if n <= 0 or self.recent_ratio <= 0:
            return 0
        if self.recent_ratio >= 1:
            return n
        return min(n, max(1, int(round(n * self.recent_ratio))))

    def _window(self, seq_len: int) -> Tuple[int, int, int]:
        n = min(self.top_k, seq_len)
        # ``recency`` means the most recent top_k positions, independent of the
        # sticky model's recent-window ratio.
        n_recent = n if self.access_model == "recency" else self._recent_count(n)
        recent_start = seq_len - n_recent
        return n, n_recent, recent_start

    def _sample(self, size: int, limit: int, exclude: set) -> set:
        """Sample ``size`` positions from ``[0, limit)`` not in ``exclude``."""
        if size <= 0 or limit <= 0:
            return set()
        chosen: set = set()
        attempts = 0
        max_attempts = max(size * 20, 64)
        while len(chosen) < size and attempts < max_attempts:
            pos = self.rng.randrange(0, limit)
            if pos not in exclude and pos not in chosen:
                chosen.add(pos)
            attempts += 1
        if len(chosen) < size:
            for pos in range(limit):
                if pos not in exclude and pos not in chosen:
                    chosen.add(pos)
                    if len(chosen) >= size:
                        break
        return chosen

    def _seed_hot_buffer(self, seq_len: int) -> None:
        if self.hot_buffer_init == "empty":
            self.hot = OrderedDict()
            return
        if self.hot_buffer_init == "latest":
            start = max(0, seq_len - self.capacity)
        else:  # earliest, matching sglang's alloc_device_buffer(arange())
            start = 0
        end = min(seq_len, start + self.capacity)
        self.hot = OrderedDict((pos, None) for pos in range(start, end))

    def _legacy_first_step(self, seq_len: int) -> Tuple[int, int]:
        n, n_recent, recent_start = self._window(seq_len)
        if self.access_model == "uniform":
            selected = set(self.rng.sample(range(seq_len), n))
            self.sticky = selected
        else:
            target_sticky = max(0, n - n_recent)
            if self.access_model == "recency":
                self.sticky = set()
            else:
                self.sticky = self._sample(target_sticky, recent_start, set())
        self._seed_hot_buffer(seq_len)
        misses = len(self.sticky - set(self.hot))
        for pos in self.sticky:
            self.hot.setdefault(pos, None)
        self.prev_seq_len = seq_len
        self.prev_recent_start = recent_start
        self.prev_n = n
        self._legacy_trim(recent_start)
        return n - misses, misses

    def _lru_first_step(self, seq_len: int) -> Tuple[int, int]:
        n, n_recent, recent_start = self._window(seq_len)
        if self.access_model == "uniform":
            selected = set(self.rng.sample(range(seq_len), n))
        else:
            target_sticky = max(0, n - n_recent)
            if self.access_model == "recency":
                self.sticky = set()
            else:
                self.sticky = self._sample(target_sticky, recent_start, set())
            selected = set(self.sticky or ())
            selected.update(range(recent_start, seq_len))
        self.sticky = selected if self.access_model == "uniform" else self.sticky
        self._seed_hot_buffer(seq_len)
        hits, misses = self._access_selected(selected, seq_len)
        self.prev_seq_len = seq_len
        self.prev_recent_start = recent_start
        self.prev_n = n
        return hits, misses

    def _legacy_trim(self, recent_start: int) -> None:
        # Evict positions that are neither in the recent window nor in the
        # sticky set.  Legacy behaviour is intentionally kept for the
        # non-profile path.
        while len(self.hot) > self.capacity:
            pos, _ = self.hot.popitem(last=False)
            if pos >= recent_start or pos in (self.sticky or ()):
                self.hot[pos] = None
                continue

    def _access_selected(self, selected: set, seq_len: int) -> Tuple[int, int]:
        """Resolve one selected set against the regular LRU + newest page.

        The swap kernel moves the current hits to MRU and inserts misses before
        the hit block.  On the direct-short path the whole compressed sequence
        fits the device buffer, so the previous newest page ages into the
        regular LRU without an H2D miss.  The long-sequence path does not migrate
        it; if selected again, it is a normal swap-in unless already resident.
        """
        hits = 0
        misses = 0
        newest = seq_len - 1
        direct_short = seq_len <= self.capacity
        if (
            direct_short
            and self.newest_page
            and self.prev_newest is not None
            and self.prev_newest >= 0
            and self.prev_newest != newest
            and self.prev_newest not in self.hot
        ):
            self.hot[self.prev_newest] = None

        hit_positions: List[int] = []
        miss_positions: List[int] = []
        for pos in sorted(selected):
            if self.newest_page and pos == newest:
                # The newest token lives in the reserved page, outside the
                # regular LRU.  If it was already resident, retain MRU order.
                hits += 1
                if pos in self.hot:
                    self.hot.move_to_end(pos)
                continue
            if pos in self.hot:
                hits += 1
                hit_positions.append(pos)
            else:
                misses += 1
                miss_positions.append(pos)

        for pos in hit_positions:
            self.hot.move_to_end(pos)
        if miss_positions:
            # Keep misses just below the hit block, matching hit-to-MRU +
            # miss-insert-before-hits rather than token-position ordering.
            insert_at = max(0, len(self.hot) - len(hit_positions))
            items = list(self.hot.items())
            self.hot = OrderedDict(
                items[:insert_at]
                + [(pos, None) for pos in miss_positions]
                + items[insert_at:]
            )
        while len(self.hot) > self.capacity:
            self.hot.popitem(last=False)
        if self.newest_page and newest >= 0:
            self.prev_newest = newest
        return hits, misses

    def _advance_sticky(self, seq_len: int) -> Tuple[int, int, int]:
        n, n_recent, recent_start = self._window(seq_len)

        # 1. Positions that entered the recent window roll out of the sticky
        #    set (at most ``seq_len - prev_seq_len`` of them in steady state).
        if self.sticky:
            if recent_start >= self.prev_recent_start:
                for pos in range(self.prev_recent_start, recent_start):
                    self.sticky.discard(pos)
            else:  # window shrank (early steps); resync completely
                self.sticky = {p for p in self.sticky if p < recent_start}
        elif self.access_model == "recency":
            self.sticky = set()

        # 2. Keep the sticky set at its target size (it grows while the
        #    sequence is still shorter than top_k).
        target_sticky = max(0, n - n_recent) if self.access_model != "recency" else 0
        if len(self.sticky) < target_sticky:
            self.sticky |= self._sample(
                target_sticky - len(self.sticky), recent_start, self.sticky
            )
        elif len(self.sticky) > target_sticky:
            drop = len(self.sticky) - target_sticky
            self.sticky -= set(self.rng.sample(sorted(self.sticky), drop))

        # 3. Churn a fraction of the sticky set.  Dropped positions must not be
        #    eligible for immediate re-selection, otherwise churn=1 silently
        #    becomes a much smaller effective churn.
        if self.access_model == "sticky" and target_sticky > 0:
            k = min(len(self.sticky), int(round(self.churn * target_sticky)))
            if k > 0:
                dropped = set(self.rng.sample(sorted(self.sticky), k))
                self.sticky -= dropped
                replacements = self._sample(
                    k, recent_start, set(self.sticky) | dropped
                )
                self.sticky |= replacements
                # When the working set covers the whole available history,
                # there is no replacement candidate.  Keep the set at its
                # target size rather than silently shrinking Top-K.
                restore = k - len(replacements)
                if restore > 0 and dropped:
                    self.sticky |= set(
                        self.rng.sample(sorted(dropped), restore)
                    )
        return n, n_recent, recent_start

    def _legacy_step(self, seq_len: int) -> Tuple[int, int]:
        if self.sticky is None:
            return self._legacy_first_step(seq_len)
        if seq_len < self.prev_seq_len:
            seq_len = self.prev_seq_len
        if self.access_model == "uniform":
            n, _, recent_start = self._window(seq_len)
            selected = set(self.rng.sample(range(seq_len), n))
            misses = len(selected - set(self.hot))
            for pos in selected:
                self.hot[pos] = None
            self.sticky = selected
        else:
            n, _, recent_start = self._advance_sticky(seq_len)
            new_items = [p for p in (self.sticky or ()) if p not in self.hot]
            misses = len(new_items)
            for pos in new_items:
                self.hot[pos] = None
            # Legacy approximation: every newly decoded token is resident.
            if seq_len > self.prev_seq_len:
                for pos in range(self.prev_seq_len, seq_len):
                    self.hot.setdefault(pos, None)
        self.prev_seq_len = seq_len
        self.prev_recent_start = recent_start
        self.prev_n = n
        self._legacy_trim(recent_start)
        return n - misses, misses

    def _lru_step(self, seq_len: int) -> Tuple[int, int]:
        if self.sticky is None:
            return self._lru_first_step(seq_len)
        if seq_len < self.prev_seq_len:
            seq_len = self.prev_seq_len
        n, n_recent, recent_start = self._window(seq_len)
        if self.access_model == "uniform":
            selected = set(self.rng.sample(range(seq_len), n))
            self.sticky = selected
        else:
            self._advance_sticky(seq_len)
            selected = set(self.sticky or ())
            selected.update(range(recent_start, seq_len))
        hits, misses = self._access_selected(selected, seq_len)
        self.prev_seq_len = seq_len
        self.prev_recent_start = recent_start
        self.prev_n = n
        return hits, misses

    def first_step(self, seq_len: int) -> Tuple[int, int]:
        if self.lru_mode:
            return self._lru_first_step(seq_len)
        return self._legacy_first_step(seq_len)

    def step(self, seq_len: int) -> Tuple[int, int]:
        if self.lru_mode:
            return self._lru_step(seq_len)
        return self._legacy_step(seq_len)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class StepCost:
    hits: int = 0
    misses: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    main_h2d_bytes: int = 0
    index_h2d_bytes: int = 0
    main_d2h_bytes: int = 0
    index_d2h_bytes: int = 0
    index_scan_bytes: int = 0
    critical_h2d_bytes: int = 0
    load_dur: float = 0.0
    backup_dur: float = 0.0


@dataclass
class _RequestGroupState:
    """Per-ratio compressed working set for one request (L1)."""

    ratio: int
    entry_bytes: float
    working: _WorkingSet
    main_entry_bytes: float = 0.0
    index_entry_bytes: float = 0.0
    prev_comp_len: int = 0
    hits: int = 0
    misses: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    main_h2d_bytes: int = 0
    index_h2d_bytes: int = 0
    main_d2h_bytes: int = 0
    index_d2h_bytes: int = 0
    new_entries: int = 0


@dataclass
class _RequestState:
    rid: str
    host_tokens: int
    working: Optional[_WorkingSet]
    steps: int = 0
    hits: int = 0
    misses: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    staged_bytes: int = 0
    finished: bool = False
    groups: List[_RequestGroupState] = field(default_factory=list)


class HiSparseSimulator:
    """Global HiSparse accounting/timing model.

    One instance per simulation process (the hisim scheduler is single
    process).  ``configure`` is called from the model-runner hook once the
    physical KV pool capacity and the model geometry are known.
    """

    _instance: Optional["HiSparseSimulator"] = None

    def __init__(
        self,
        config: HiSparseSimConfig,
        *,
        num_layers: int,
        bytes_per_token_per_layer: float,
        physical_pool_tokens: int,
        page_size: int = 1,
        max_num_reqs: Optional[int] = None,
        read_bandwidth: Optional[float] = None,
        write_bandwidth: Optional[float] = None,
        kv_profile: Optional[KVProfile] = None,
        tp_size: int = 1,
    ) -> None:
        self.config = config
        self.num_layers = max(1, int(num_layers))
        self.bytes_per_token_per_layer = float(bytes_per_token_per_layer)
        physical_tokens = int(physical_pool_tokens)
        if physical_tokens < 1:
            raise ValueError("HiSparse physical pool must contain at least one token")
        self.physical_pool_tokens = physical_tokens
        self.page_size = max(1, int(page_size))
        self.padded_buffer_size = config.padded_buffer_size
        if self.physical_pool_tokens < self.padded_buffer_size:
            raise ValueError(
                f"HiSparse physical pool has {self.physical_pool_tokens} tokens, "
                f"but one request reserves {self.padded_buffer_size}; increase "
                "the hot-pool budget or reduce device_buffer_size"
            )
        self.read_bandwidth = read_bandwidth
        self.write_bandwidth = write_bandwidth
        self.tp_size = max(1, int(tp_size))

        self.kv_profile = (
            KVProfile.from_value(kv_profile)
            if kv_profile is not None
            else KVProfile.from_value(config.kv_profile)
        )
        self.profile_enabled = bool(
            self.kv_profile is not None and self.kv_profile.enabled
        )
        self.bytes_scope = (
            self.kv_profile.bytes_scope if self.profile_enabled else "legacy"
        )
        # ``per_rank`` profiles already carry one rank's bytes (DCP=1, one KV
        # head per rank); ``global`` profiles keep the pre-L2 behaviour and are
        # divided by TP.
        self._tp_scale = (
            1.0
            if self.profile_enabled and self.kv_profile.per_rank
            else 1.0 / self.tp_size
        )
        self.offload_bytes_per_token = (
            float(self.kv_profile.offload_bytes_per_token) * self._tp_scale
            if self.profile_enabled
            else 0.0
        )
        self.indexer_bytes_per_token = (
            float(self.kv_profile.indexer_bytes_per_token) * self._tp_scale
            if self.profile_enabled
            else 0.0
        )
        self.io_groups: List[Tuple[int, float]] = (
            [
                (int(ratio), float(entry_bytes) * self._tp_scale)
                for ratio, entry_bytes in self.kv_profile.io_groups()
            ]
            if self.profile_enabled
            else []
        )
        self.io_channel_groups: List[Tuple[int, float, float]] = (
            [
                (
                    int(ratio),
                    float(main_bytes) * self._tp_scale,
                    float(index_bytes) * self._tp_scale,
                )
                for ratio, main_bytes, index_bytes in self.kv_profile.io_channel_groups()
            ]
            if self.profile_enabled
            else []
        )
        self.capacity_bytes_per_token = (
            float(self.kv_profile.capacity_bytes_per_token) * self._tp_scale
            if self.profile_enabled
            else 0.0
        )
        self.hot_buffer_init = (
            self.config.hot_buffer_init
            or (self.kv_profile.hot_buffer_init if self.profile_enabled else "latest")
        )

        self.logical_pool_tokens = int(
            self.physical_pool_tokens * config.host_to_device_ratio
        )
        # Host pool is sized from the device pool exactly like sglang.
        self.host_pool_tokens = self.logical_pool_tokens
        self.max_reqs_by_hot_pool = (
            self.physical_pool_tokens // self.padded_buffer_size
        )
        self.max_num_reqs = (
            min(max_num_reqs, self.max_reqs_by_hot_pool)
            if max_num_reqs
            else self.max_reqs_by_hot_pool
        )
        self.hbm_swa_bytes = (
            self.max_num_reqs
            * int(self.kv_profile.swa_window_tokens)
            * float(self.kv_profile.swa_bytes_per_token)
            * self._tp_scale
            if self.profile_enabled
            and self.kv_profile.swa_window_tokens
            and self.kv_profile.swa_bytes_per_token
            else 0.0
        )
        self.hbm_compressor_bytes = (
            self.max_num_reqs
            * float(self.kv_profile.compressor_bytes_per_request)
            * self._tp_scale
            if self.profile_enabled
            and self.kv_profile.compressor_bytes_per_request
            else 0.0
        )
        # Worst-case per-request admission overhead (SWA + compressor + null
        # block).  Kept separate from steady-state ``hbm_kv_bytes``: it is an
        # admission upper bound, not decode occupancy.
        self.hbm_admission_overhead_bytes = (
            self.max_num_reqs
            * float(
                self.kv_profile.admission_overhead_bytes(
                    self.kv_profile.swa_admission_tokens
                )
            )
            * self._tp_scale
            if self.profile_enabled
            and self.kv_profile.admission_bytes_per_block
            else 0.0
        )

        self.requests: Dict[str, _RequestState] = {}
        self.finished_requests = 0
        self.decode_steps = 0
        self.decode_batches = 0
        self.hits = 0
        self.misses = 0
        self.hits_bytes = 0
        self.misses_bytes = 0
        self.new_compressed_entries = 0
        self.h2d_bytes_by_ratio: Dict[int, int] = {}
        self.d2h_bytes_by_ratio: Dict[int, int] = {}
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.main_h2d_bytes = 0
        self.index_h2d_bytes = 0
        self.main_d2h_bytes = 0
        self.index_d2h_bytes = 0
        self.index_scan_bytes = 0
        self.critical_h2d_bytes = 0
        self.staging_bytes = 0
        self.staging_dur = 0.0
        self.prefix_reused_tokens = 0
        self.load_dur = 0.0
        self.backup_dur = 0.0
        # Eager-backup latency that is hidden behind the next forward when
        # backup_on_critical is off.
        self.backup_hidden_dur = 0.0
        self.peak_host_tokens = 0
        self.host_overcommit_tokens = 0
        self.host_tokens = 0
        self.device_alloc_fail = 0
        self.cur_host_tokens = 0
        self._host_warned = False
        self._rng = random.Random(config.seed)

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def configure(
        cls,
        config: HiSparseSimConfig,
        *,
        num_layers: int,
        bytes_per_token_per_layer: float,
        physical_pool_tokens: int,
        page_size: int = 1,
        max_num_reqs: Optional[int] = None,
        kv_profile: Optional[KVProfile] = None,
        tp_size: int = 1,
    ) -> "HiSparseSimulator":
        cls._instance = cls(
            config,
            num_layers=num_layers,
            bytes_per_token_per_layer=bytes_per_token_per_layer,
            physical_pool_tokens=physical_pool_tokens,
            page_size=page_size,
            max_num_reqs=max_num_reqs,
            kv_profile=kv_profile,
            tp_size=tp_size,
        )
        logger.info(
            "HiSparse simulation enabled: profile=%s top_k=%d device_buffer=%d "
            "padded_buffer=%d host_to_device_ratio=%d "
            "physical_tokens=%d logical_tokens=%d max_reqs_hot_pool=%d "
            "hbm_kv=%.2fGB host_kv=%.2fGB",
            cls._instance.kv_profile.name if cls._instance.profile_enabled else "legacy",
            config.top_k,
            config.device_buffer_size,
            config.padded_buffer_size,
            config.host_to_device_ratio,
            cls._instance.physical_pool_tokens,
            cls._instance.logical_pool_tokens,
            cls._instance.max_reqs_by_hot_pool,
            cls._instance.hbm_kv_bytes / 1e9,
            cls._instance.host_kv_bytes / 1e9,
        )
        return cls._instance

    @classmethod
    def instance(cls) -> Optional["HiSparseSimulator"]:
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @property
    def hbm_kv_bytes(self) -> float:
        if self.profile_enabled:
            return (
                self.physical_pool_tokens * self.capacity_bytes_per_token
                + self.hbm_swa_bytes
                + self.hbm_compressor_bytes
            )
        return (
            self.physical_pool_tokens
            * self.num_layers
            * self.bytes_per_token_per_layer
        )

    @property
    def host_kv_bytes(self) -> float:
        if self.profile_enabled:
            return self.logical_pool_tokens * self.offload_bytes_per_token
        return (
            self.logical_pool_tokens
            * self.num_layers
            * self.bytes_per_token_per_layer
        )

    @property
    def baseline_hbm_kv_bytes(self) -> float:
        if self.profile_enabled:
            return (
                self.logical_pool_tokens * self.capacity_bytes_per_token
                + self.hbm_swa_bytes
                + self.hbm_compressor_bytes
            )
        return (
            self.logical_pool_tokens
            * self.num_layers
            * self.bytes_per_token_per_layer
        )

    def on_request_start(
        self, rid: str, prompt_len: int, prefix_len: int = 0
    ) -> None:
        prompt_len = max(0, int(prompt_len))
        prefix_len = min(prompt_len, max(0, int(prefix_len)))
        if rid in self.requests:
            return
        new_prompt = prompt_len - prefix_len
        self.prefix_reused_tokens += prefix_len
        if self.profile_enabled:
            groups: List[_RequestGroupState] = []
            staged = 0.0
            for (ratio, entry_bytes), (_, main_bytes, index_bytes) in zip(
                self.io_groups, self.io_channel_groups
            ):
                # Compression emits only complete entries.  Using ceil here
                # would stage a partial final entry and then charge it again as
                # a new D2H entry when the next token completes it.
                full_comp = prompt_len // ratio
                prefix_comp = prefix_len // ratio
                stage_comp = max(0, full_comp - prefix_comp)
                staged += stage_comp * entry_bytes
                groups.append(
                    _RequestGroupState(
                        ratio=ratio,
                        entry_bytes=entry_bytes,
                        main_entry_bytes=main_bytes,
                        index_entry_bytes=index_bytes,
                        # Every complete prompt entry up to here is already in
                        # the host pool (freshly staged or a reused prefix).
                        prev_comp_len=full_comp,
                        working=_WorkingSet(
                            top_k=self.config.top_k,
                            capacity=self.config.device_buffer_size,
                            access_model=self.config.access_model,
                            churn=self.config.churn,
                            recent_ratio=self.config.recent_ratio,
                            rng=random.Random(self._rng.random()),
                            hot_buffer_init=self.hot_buffer_init,
                            lru_mode=True,
                            newest_page=True,
                        ),
                    )
                )
            state = _RequestState(
                rid=rid,
                host_tokens=new_prompt,
                working=None,
                staged_bytes=int(staged),
                groups=groups,
            )
        else:
            working = _WorkingSet(
                top_k=self.config.top_k,
                capacity=self.padded_buffer_size,
                access_model=self.config.access_model,
                churn=self.config.churn,
                recent_ratio=self.config.recent_ratio,
                rng=random.Random(self._rng.random()),
                hot_buffer_init="latest",
            )
            staged = new_prompt * self.num_layers * self.bytes_per_token_per_layer
            state = _RequestState(
                rid=rid,
                host_tokens=new_prompt,
                working=working,
                staged_bytes=int(staged),
            )
        self.requests[rid] = state
        self.staging_bytes += int(staged)
        if staged > 0:
            self.staging_dur += transfer_duration(
                "D2H", staged, self.read_bandwidth, self.write_bandwidth
            )
        self.cur_host_tokens += new_prompt
        if self.cur_host_tokens > self.host_pool_tokens:
            self.host_overcommit_tokens = max(
                self.host_overcommit_tokens, self.cur_host_tokens - self.host_pool_tokens
            )
            if not self._host_warned:
                logger.warning(
                    "HiSparse host pool overcommitted: %d tokens in use > %d "
                    "pool tokens.  Increase HISIM_HISPARSE_HOST_TO_DEVICE_RATIO "
                    "or reduce concurrency.",
                    self.cur_host_tokens,
                    self.host_pool_tokens,
                )
                self._host_warned = True
        self.peak_host_tokens = max(self.peak_host_tokens, self.cur_host_tokens)

    def _decode_one(self, rid: str, past_len: int) -> StepCost:
        """Update one request's access stream and return its byte deltas.

        Transfer timing is *not* computed here: a decode batch is served by
        one swap-in kernel (one fixed launch overhead) whose copies share the
        host<->device bandwidth, so timing is applied once per batch in
        :meth:`on_decode_batch`.
        """
        state = self.requests.get(rid)
        if state is None:
            return StepCost()
        if self.profile_enabled:
            return self._decode_one_profile(state, past_len)
        past_len = max(1, int(past_len))
        hits, misses = state.working.step(past_len)

        n_layers = self.num_layers
        h2d = int(misses * n_layers * self.bytes_per_token_per_layer)
        # The freshly decoded token is written to HBM and eagerly backed up
        # to host; the first decode step is skipped (nothing to back up yet).
        backup_tokens = 1 if state.steps > 0 else 0
        d2h = int(backup_tokens * n_layers * self.bytes_per_token_per_layer)
        hit_bytes = hits * n_layers * self.bytes_per_token_per_layer
        miss_bytes = misses * n_layers * self.bytes_per_token_per_layer

        state.steps += 1
        state.hits += hits * n_layers
        state.misses += misses * n_layers
        state.h2d_bytes += h2d
        state.d2h_bytes += d2h
        # The decoded token extends the host-resident sequence.
        state.host_tokens += 1
        self.cur_host_tokens += 1

        self.decode_steps += 1
        self.hits += hits * n_layers
        self.misses += misses * n_layers
        self.hits_bytes += hit_bytes
        self.misses_bytes += miss_bytes
        self.h2d_bytes += h2d
        self.d2h_bytes += d2h
        self.main_h2d_bytes += h2d
        self.main_d2h_bytes += d2h
        self.peak_host_tokens = max(self.peak_host_tokens, self.cur_host_tokens)
        if self.cur_host_tokens > self.host_pool_tokens:
            self.host_overcommit_tokens = max(
                self.host_overcommit_tokens, self.cur_host_tokens - self.host_pool_tokens
            )

        return StepCost(
            hits=hits * n_layers,
            misses=misses * n_layers,
            h2d_bytes=h2d,
            d2h_bytes=d2h,
            main_h2d_bytes=h2d,
            main_d2h_bytes=d2h,
        )

    def _decode_one_profile(self, state: _RequestState, past_len: int) -> StepCost:
        """Decode step: compressed selection + main/index IO per ratio group.

        Compression cadence, indexer selection cadence and eager-backup cadence
        are intentionally separate.  A new query can select a different Top-K
        even when no new compressed entry was produced.
        """
        past_len = max(0, int(past_len))
        hits = misses = h2d = d2h = 0
        main_h2d = index_h2d = 0
        main_d2h = index_d2h = 0
        hit_bytes = miss_bytes = 0
        new_entries = 0
        for group in state.groups:
            ratio = group.ratio
            comp_len = past_len // ratio
            # Entries up to prev_comp_len are already staged.  The first hook
            # decode can legitimately report past_len=P+1 (prompt plus the first
            # generated token), in which case that new completed entry still
            # needs its one D2H backup on this first step.
            new = max(0, comp_len - group.prev_comp_len)
            ws = group.working
            if ws.sticky is None:
                gh, gm = ws.first_step(comp_len)
            elif self.config.selection_cadence == "every_step" or new > 0:
                gh, gm = ws.step(comp_len)
            else:
                # Explicit sensitivity mode: hold the selected set between
                # compression boundaries.  Not the default semantics.
                gh, gm = min(ws.top_k, comp_len), 0
            group.prev_comp_len = comp_len

            group_main_h2d = int(gm * group.main_entry_bytes)
            index_reads = gh + gm if self.config.index_read_on_hit else gm
            group_index_h2d = int(index_reads * group.index_entry_bytes)
            group_h2d = group_main_h2d + group_index_h2d
            group_main_d2h = int(new * group.main_entry_bytes)
            group_index_d2h = int(new * group.index_entry_bytes)
            group_d2h = group_main_d2h + group_index_d2h
            group_hit_bytes = gh * group.entry_bytes
            group_miss_bytes = gm * group.entry_bytes

            group.hits += gh
            group.misses += gm
            group.h2d_bytes += group_h2d
            group.d2h_bytes += group_d2h
            group.main_h2d_bytes += group_main_h2d
            group.index_h2d_bytes += group_index_h2d
            group.main_d2h_bytes += group_main_d2h
            group.index_d2h_bytes += group_index_d2h
            group.new_entries += new

            hits += gh
            misses += gm
            h2d += group_h2d
            d2h += group_d2h
            main_h2d += group_main_h2d
            index_h2d += group_index_h2d
            main_d2h += group_main_d2h
            index_d2h += group_index_d2h
            hit_bytes += group_hit_bytes
            miss_bytes += group_miss_bytes
            new_entries += new
            self.h2d_bytes_by_ratio[ratio] = (
                self.h2d_bytes_by_ratio.get(ratio, 0) + group_h2d
            )
            self.d2h_bytes_by_ratio[ratio] = (
                self.d2h_bytes_by_ratio.get(ratio, 0) + group_d2h
            )

        state.steps += 1
        state.hits += hits
        state.misses += misses
        state.h2d_bytes += h2d
        state.d2h_bytes += d2h
        state.host_tokens += 1
        self.cur_host_tokens += 1

        self.decode_steps += 1
        self.hits += hits
        self.misses += misses
        self.hits_bytes += hit_bytes
        self.misses_bytes += miss_bytes
        self.new_compressed_entries += new_entries
        self.h2d_bytes += h2d
        self.d2h_bytes += d2h
        self.main_h2d_bytes += main_h2d
        self.index_h2d_bytes += index_h2d
        self.main_d2h_bytes += main_d2h
        self.index_d2h_bytes += index_d2h
        self.index_scan_bytes += index_h2d
        self.peak_host_tokens = max(self.peak_host_tokens, self.cur_host_tokens)
        if self.cur_host_tokens > self.host_pool_tokens:
            self.host_overcommit_tokens = max(
                self.host_overcommit_tokens, self.cur_host_tokens - self.host_pool_tokens
            )

        return StepCost(
            hits=hits,
            misses=misses,
            h2d_bytes=h2d,
            d2h_bytes=d2h,
            main_h2d_bytes=main_h2d,
            index_h2d_bytes=index_h2d,
            main_d2h_bytes=main_d2h,
            index_d2h_bytes=index_d2h,
            index_scan_bytes=index_h2d,
        )

    def _account_transfer(self, cost: StepCost) -> None:
        # Index selection and main-KV swap-in are separate phases.  Charge a
        # launch only for a phase that actually moves/reads bytes.
        fraction = self.config.critical_fraction
        index_dur = transfer_duration(
            "H2D",
            cost.index_h2d_bytes * fraction,
            self.read_bandwidth,
            self.write_bandwidth,
        )
        main_dur = transfer_duration(
            "H2D",
            cost.main_h2d_bytes * fraction,
            self.read_bandwidth,
            self.write_bandwidth,
        )
        cost.load_dur = index_dur + main_dur
        cost.critical_h2d_bytes = int(
            (cost.main_h2d_bytes + cost.index_h2d_bytes) * fraction
        )
        cost.backup_dur = transfer_duration(
            "D2H", cost.d2h_bytes, self.read_bandwidth, self.write_bandwidth
        )
        self.decode_batches += 1
        self.load_dur += cost.load_dur
        self.backup_dur += cost.backup_dur
        self.critical_h2d_bytes += cost.critical_h2d_bytes

    def on_decode_step(self, rid: str, past_len: int) -> StepCost:
        """Single-request decode step (batch of one)."""
        cost = self._decode_one(rid, past_len)
        self._account_transfer(cost)
        return cost

    def on_decode_batch(self, requests: Sequence[Tuple[str, int]]) -> StepCost:
        total = StepCost()
        for rid, past_len in requests:
            cost = self._decode_one(rid, past_len)
            total.hits += cost.hits
            total.misses += cost.misses
            total.h2d_bytes += cost.h2d_bytes
            total.d2h_bytes += cost.d2h_bytes
            total.main_h2d_bytes += cost.main_h2d_bytes
            total.index_h2d_bytes += cost.index_h2d_bytes
            total.main_d2h_bytes += cost.main_d2h_bytes
            total.index_d2h_bytes += cost.index_d2h_bytes
            total.index_scan_bytes += cost.index_scan_bytes
            total.critical_h2d_bytes += cost.critical_h2d_bytes
        self._account_transfer(total)
        return total

    def on_request_finish(self, rid: str) -> None:
        state = self.requests.pop(rid, None)
        if state is None:
            return
        if state.finished:
            return
        state.finished = True
        self.finished_requests += 1
        self.cur_host_tokens -= state.host_tokens
        if self.cur_host_tokens < 0:
            self.cur_host_tokens = 0

    # -- reporting ---------------------------------------------------------

    def metrics(self) -> dict:
        touched = self.hits + self.misses
        touched_bytes = self.hits_bytes + self.misses_bytes
        hbm = self.hbm_kv_bytes
        host = self.host_kv_bytes
        baseline_hbm = self.baseline_hbm_kv_bytes
        p = self.kv_profile if self.profile_enabled else None
        return {
            "hisparse_enabled": 1,
            "hisparse_kv_profile": (
                self.kv_profile.name if self.profile_enabled else "legacy"
            ),
            "hisparse_tp_size": self.tp_size,
            "hisparse_bytes_scope": self.bytes_scope,
            "hisparse_runtime_target": (
                p.runtime_target if p else "legacy"
            ),
            "hisparse_offload_bytes_per_token": self.offload_bytes_per_token,
            "hisparse_capacity_bytes_per_token": self.capacity_bytes_per_token,
            "hisparse_source_layer_ratios": (
                list(p.source_layer_ratios) if p else []
            ),
            "hisparse_source_layer_ids": (
                list(p.source_layer_ids) if p else []
            ),
            "hisparse_source_layer_bytes_per_token": (
                list(p.source_layer_bytes_per_token) if p else []
            ),
            "hisparse_source_layer_bytes_sum": (
                p.source_layer_bytes_sum if p else 0.0
            ),
            "hisparse_source_consumers": (
                [[source, list(consumers)] for source, consumers in p.source_consumers]
                if p
                else []
            ),
            "hisparse_layer_groups": (
                [list(group) for group in p.layer_groups] if p else []
            ),
            "hisparse_profile_num_layers": (
                sum(int(count) for count, _ in p.layer_groups)
                if p and p.layer_groups
                else 0
            ),
            "hisparse_index_topk": int(p.index_topk) if p else 0,
            "hisparse_index_storage_layers": (
                list(p.index_storage_layers) if p else []
            ),
            "hisparse_index_compute_layers": (
                list(p.index_compute_layers) if p else []
            ),
            "hisparse_main_kv_bytes_per_token": (
                p.main_kv_bytes_per_token if p else 0.0
            ),
            "hisparse_index_bytes_per_token": (
                p.index_bytes_per_token if p else 0.0
            ),
            "hisparse_indexer_bytes_per_token": self.indexer_bytes_per_token,
            "hisparse_scheduler_block_tokens": (
                int(p.scheduler_block_tokens) if p else 0
            ),
            "hisparse_page_bytes": int(p.page_bytes) if p else 0,
            "hisparse_slot_count": int(p.slot_count) if p else 0,
            "hisparse_allocator_bytes_per_token": (
                p.allocator_bytes_per_token if p else 0.0
            ),
            "hisparse_payload_bytes_per_block": (
                p.payload_bytes_per_block if p else 0.0
            ),
            "hisparse_candidate_blocks": int(p.candidate_blocks) if p else 0,
            "hisparse_candidate_block_tokens": (
                int(p.candidate_block_tokens) if p else 0
            ),
            "hisparse_candidate_source_layer": (
                int(p.candidate_source_layer) if p else -1
            ),
            "hisparse_candidate_filter_layers": (
                list(p.candidate_filter_layers) if p else []
            ),
            "hisparse_swa_layers": int(p.swa_layers) if p else 0,
            "hisparse_swa_window_tokens": (
                int(p.swa_window_tokens) if p else 0
            ),
            "hisparse_swa_admission_tokens": (
                int(p.swa_admission_tokens) if p else 0
            ),
            "hisparse_hbm_swa_bytes": int(self.hbm_swa_bytes),
            "hisparse_hbm_compressor_bytes": int(self.hbm_compressor_bytes),
            "hisparse_compressor_layers": (
                list(p.compressor_layers) if p else []
            ),
            "hisparse_compressor_page_bytes": (
                int(p.compressor_page_bytes) if p else 0
            ),
            "hisparse_compressor_bytes_per_request": (
                p.compressor_bytes_per_request if p else 0.0
            ),
            "hisparse_admission_bytes_per_block": (
                int(p.admission_bytes_per_block) if p else 0
            ),
            "hisparse_admission_4223_overhead_bytes": (
                p.admission_overhead_bytes(p.swa_admission_tokens) if p else 0
            ),
            "hisparse_admission_128_blocks": (
                p.admission_blocks(128) if p else 0
            ),
            "hisparse_admission_8192_blocks": (
                p.admission_blocks(8192) if p else 0
            ),
            "hisparse_admission_1m_blocks": (
                p.admission_blocks(1_048_576) if p else 0
            ),
            "hisparse_admission_1m_bytes": (
                p.admission_bytes(1_048_576) if p else 0
            ),
            "hisparse_hbm_admission_overhead_bytes": int(
                self.hbm_admission_overhead_bytes
            ),
            "hisparse_io_groups": (
                [[ratio, entry_bytes] for ratio, entry_bytes in self.io_groups]
                if self.profile_enabled
                else []
            ),
            "hisparse_io_channel_groups": (
                [
                    [ratio, main_bytes, index_bytes]
                    for ratio, main_bytes, index_bytes in self.io_channel_groups
                ]
                if self.profile_enabled
                else []
            ),
            "hisparse_top_k": self.config.top_k,
            "hisparse_device_buffer_size": self.config.device_buffer_size,
            "hisparse_padded_buffer_size": self.padded_buffer_size,
            "hisparse_host_to_device_ratio": self.config.host_to_device_ratio,
            "hisparse_access_model": self.config.access_model,
            "hisparse_churn": self.config.churn,
            "hisparse_recent_ratio": self.config.recent_ratio,
            "hisparse_selection_cadence": self.config.selection_cadence,
            "hisparse_hot_buffer_init": self.hot_buffer_init,
            "hisparse_index_read_on_hit": int(self.config.index_read_on_hit),
            "hisparse_shared_index_freq": self.config.shared_index_freq,
            "hisparse_backup_on_critical": int(self.config.backup_on_critical),
            "hisparse_physical_pool_tokens": self.physical_pool_tokens,
            "hisparse_logical_pool_tokens": self.logical_pool_tokens,
            "hisparse_host_pool_tokens": self.host_pool_tokens,
            "hisparse_max_reqs_hot_pool": self.max_reqs_by_hot_pool,
            "hisparse_num_layers": self.num_layers,
            "hisparse_bytes_per_token_per_layer": self.bytes_per_token_per_layer,
            "hisparse_hbm_kv_bytes": int(hbm),
            "hisparse_host_kv_bytes": int(host),
            "hisparse_baseline_hbm_kv_bytes": int(baseline_hbm),
            "hisparse_hbm_saving_bytes": int(max(0, baseline_hbm - hbm)),
            "hisparse_requests": len(self.requests) + self.finished_requests,
            "hisparse_finished_requests": self.finished_requests,
            "hisparse_decode_steps": self.decode_steps,
            "hisparse_decode_batches": self.decode_batches,
            "hisparse_hits": self.hits,
            "hisparse_misses": self.misses,
            "hisparse_hit_rate": (self.hits / touched) if touched else 0.0,
            "hisparse_hits_bytes": int(self.hits_bytes),
            "hisparse_misses_bytes": int(self.misses_bytes),
            "hisparse_hit_rate_bytes": (
                self.hits_bytes / touched_bytes if touched_bytes else 0.0
            ),
            "hisparse_new_compressed_entries": self.new_compressed_entries,
            "hisparse_h2d_bytes_by_ratio": {
                str(k): int(v) for k, v in sorted(self.h2d_bytes_by_ratio.items())
            },
            "hisparse_d2h_bytes_by_ratio": {
                str(k): int(v) for k, v in sorted(self.d2h_bytes_by_ratio.items())
            },
            "hisparse_h2d_bytes": int(self.h2d_bytes),
            "hisparse_d2h_bytes": int(self.d2h_bytes),
            "hisparse_main_kv_h2d_bytes": int(self.main_h2d_bytes),
            "hisparse_index_h2d_bytes": int(self.index_h2d_bytes),
            "hisparse_main_kv_d2h_bytes": int(self.main_d2h_bytes),
            "hisparse_index_d2h_bytes": int(self.index_d2h_bytes),
            "hisparse_index_scan_bytes": int(self.index_scan_bytes),
            "hisparse_critical_h2d_bytes": int(self.critical_h2d_bytes),
            "hisparse_staging_bytes": int(self.staging_bytes),
            "hisparse_prefix_reused_tokens": self.prefix_reused_tokens,
            "hisparse_total_io_bytes": int(self.h2d_bytes + self.d2h_bytes + self.staging_bytes),
            "hisparse_staging_latency_s": self.staging_dur,
            "hisparse_load_latency_s": self.load_dur,
            "hisparse_backup_latency_s": self.backup_dur,
            "hisparse_backup_hidden_latency_s": self.backup_hidden_dur,
            "hisparse_mean_load_us": (
                self.load_dur / self.decode_batches * 1e6
                if self.decode_batches
                else 0.0
            ),
            "hisparse_mean_backup_us": (
                self.backup_dur / self.decode_batches * 1e6
                if self.decode_batches
                else 0.0
            ),
            "hisparse_peak_host_tokens": self.peak_host_tokens,
            "hisparse_host_overcommit_tokens": self.host_overcommit_tokens,
            "hisparse_device_alloc_fail": self.device_alloc_fail,
        }


# ---------------------------------------------------------------------------
# Helpers used by the hooks
# ---------------------------------------------------------------------------


def model_kv_geometry(model_config) -> Tuple[int, float]:
    """Return ``(num_layers, bytes_per_token_per_layer)`` for the HF config.

    Mirrors sglang/hisim: MLA models store ``kv_lora_rank + qk_rope_head_dim``
    elements per token per layer; MHA/GQA models store ``2 * kv_heads * head_dim``.
    """
    num_layers = int(
        getattr(model_config, "num_hidden_layers", 0)
        or getattr(model_config, "num_layers", 0)
        or 1
    )
    dtype = getattr(model_config, "dtype", None)
    try:
        import torch

        itemsize = torch._utils._element_size(dtype) if dtype is not None else 2
    except Exception:
        itemsize = 2

    kv_lora_rank = int(getattr(model_config, "kv_lora_rank", 0) or 0)
    if kv_lora_rank > 0:
        qk_rope = int(getattr(model_config, "qk_rope_head_dim", 0) or 0)
        per_layer = (kv_lora_rank + qk_rope) * itemsize
    else:
        heads = int(
            getattr(model_config, "num_key_value_heads", 0)
            or getattr(model_config, "num_attention_heads", 0)
            or 1
        )
        head_dim = int(
            getattr(model_config, "head_dim", 0)
            or (
                int(getattr(model_config, "hidden_size", 0) or 0)
                // max(1, int(getattr(model_config, "num_attention_heads", 0) or 1))
            )
            or 1
        )
        per_layer = 2 * heads * head_dim * itemsize
    return num_layers, float(per_layer)


def model_info_kv_geometry(
    model_info, tp_size: int = 1, itemsize: Optional[int] = None
) -> Tuple[int, float]:
    """Same as :func:`model_kv_geometry`, but for hisim's ``ModelInfo`` and an
    explicit KV-cache element size (the cache dtype can differ from the model
    dtype, e.g. FP8 KV for DeepSeek)."""
    num_layers = max(1, int(getattr(model_info, "num_hidden_layers", 1) or 1))
    if itemsize is None:
        itemsize = 2
    kv_lora_rank = int(getattr(model_info, "kv_lora_rank", 0) or 0)
    if kv_lora_rank > 0:
        per_layer = (kv_lora_rank + int(getattr(model_info, "qk_rope_head_dim", 0) or 0)) * itemsize
    else:
        heads = max(
            1,
            int(getattr(model_info, "num_key_value_heads", 0) or 1) // max(1, tp_size),
        )
        head_dim = int(getattr(model_info, "head_dim", 0) or 1)
        per_layer = 2 * heads * head_dim * itemsize
    return num_layers, float(per_layer)
