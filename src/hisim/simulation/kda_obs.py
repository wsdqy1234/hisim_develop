"""KDA/MLA Checkpoint observation module for Phase 1 diagnostics.

All functions are no-ops unless HISIM_KDA_OBS=1 is set.
This ensures baseline behavior is unchanged when observation is disabled.

Outputs:
  - checkpoint_events.jsonl: structured event log in HISIM_OUTPUT_DIR
  - per-req KDA metrics via RequestStats fields (populated by sglang_hook.py)

Event types:
  - track: KDA Tracking State established (mamba_last_track_seqlen set)
  - donate: Checkpoint donated to Radix Tree
  - commit: Checkpoint committed to node.component_data[MAMBA].value
  - match: match_prefix result for a request
  - backup_l2: L1→L2 backup (D2H) for a checkpoint
  - evict_l1: Checkpoint evicted from L1 (demoted to L2 or deleted)
  - restore_l2: L2→L1 load-back (H2D) for a checkpoint
  - restore_l3: L3→L2/L1 restore for a checkpoint
  - delete: Checkpoint permanently deleted from tree
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from hisim.utils import get_logger

logger = get_logger("hisim")

_KDA_OBS_ENABLED: Optional[bool] = None
_OUTPUT_DIR: Optional[str] = None
_EVENT_FILE: Optional[str] = None
_MAMBA_SLOT_BYTES = 14653440  # 18 layers × 814080 B/layer
_MLA_ENTRY_BYTES = 5760       # 5 layers × 288 × 2 × 2


def is_obs_enabled() -> bool:
    global _KDA_OBS_ENABLED
    if _KDA_OBS_ENABLED is None:
        _KDA_OBS_ENABLED = os.environ.get("HISIM_KDA_OBS", "").lower() in ("1", "true", "yes")
    return _KDA_OBS_ENABLED


def _get_output_dir() -> str:
    global _OUTPUT_DIR, _EVENT_FILE
    if _OUTPUT_DIR is None:
        _OUTPUT_DIR = os.environ.get("HISIM_OUTPUT_DIR", "/tmp/hisim/output")
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
    if _EVENT_FILE is None:
        _EVENT_FILE = os.path.join(_OUTPUT_DIR, "checkpoint_events.jsonl")
    return _OUTPUT_DIR


def log_event(
    event_type: str,
    *,
    virtual_time: float = -1.0,
    rid: str = "",
    session_id: str = "",
    node_id: int = -1,
    checkpoint_pos: int = -1,
    layer: str = "",          # L1/L2/L3
    state_size: int = 0,
    reason: str = "",         # proactive/alloc_fail/natural/backup/restore
    extra: dict = None,
):
    """Log a checkpoint lifecycle event. No-op if obs disabled."""
    if not is_obs_enabled():
        return
    evt = {
        "ts": time.time(),
        "vtime": virtual_time,
        "event": event_type,
        "rid": rid,
        "session_id": session_id,
        "node_id": node_id,
        "checkpoint_pos": checkpoint_pos,
        "layer": layer,
        "state_size": state_size,
        "reason": reason,
    }
    if extra:
        evt.update(extra)
    try:
        _get_output_dir()
        with open(_EVENT_FILE, "a") as f:
            f.write(json.dumps(evt) + "\n")
    except Exception as e:
        logger.debug(f"kda_obs log_event failed: {e}")


@dataclass
class PerReqKdaObs:
    """Accumulator for per-request KDA metrics. Stored on RequestStats."""
    mla_potential_hit_tokens: int = 0
    mla_l2_hit_tokens: int = 0
    mla_loadback_bytes: int = 0

    kda_l1_hit: bool = False
    kda_l2_hit: bool = False
    kda_l3_hit: bool = False
    kda_recoverable_position: int = 0
    kda_checkpoint_pos: int = 0
    kda_loadback_bytes: int = 0

    effective_prefix_hit_tokens: int = 0
    retreat_gap_tokens: int = 0
    kda_replay_tokens: int = 0
    kda_cow_src_slot: int = -1
    kda_new_tokens: int = 0

    # [A2] New metrics
    scheduler_effective_hit_tokens: int = 0
    verified_logical_reuse: bool = False
    historical_tokens_actually_scheduled: int = 0
    kda_restore_tier: str = ""
    kda_restore_latency_ms: float = 0.0
    mla_restore_latency_ms: float = 0.0

    # Internal tracking (not written to request.jsonl)
    _match_full_kv_hit: int = 0
    _match_host_hit: int = 0
    _match_mamba_host_hit: int = 0
    _match_best_node_id: int = -1
    _extend_length: int = 0
    _total_input: int = 0
    _kda_tier: str = ""
    _shadow_verified: dict = None


def populate_request_stats(req_stats, obs: PerReqKdaObs):
    """Copy PerReqKdaObs fields onto a RequestStats object."""
    if not is_obs_enabled() or obs is None:
        return
    req_stats.mla_potential_hit_tokens = obs.mla_potential_hit_tokens
    req_stats.mla_l2_hit_tokens = obs.mla_l2_hit_tokens
    req_stats.mla_loadback_bytes = obs.mla_loadback_bytes
    req_stats.kda_l1_hit = obs.kda_l1_hit
    req_stats.kda_l2_hit = obs.kda_l2_hit
    req_stats.kda_l3_hit = obs.kda_l3_hit
    req_stats.kda_recoverable_position = obs.kda_recoverable_position
    req_stats.kda_checkpoint_pos = obs.kda_checkpoint_pos
    req_stats.kda_loadback_bytes = obs.kda_loadback_bytes
    req_stats.effective_prefix_hit_tokens = obs.effective_prefix_hit_tokens
    req_stats.retreat_gap_tokens = obs.retreat_gap_tokens
    req_stats.kda_replay_tokens = obs.kda_replay_tokens
    req_stats.kda_cow_src_slot = obs.kda_cow_src_slot
    req_stats.kda_new_tokens = obs.kda_new_tokens
    # [A2] New metrics
    req_stats.scheduler_effective_hit_tokens = obs.scheduler_effective_hit_tokens
    req_stats.verified_logical_reuse = obs.verified_logical_reuse
    req_stats.historical_tokens_actually_scheduled = obs.historical_tokens_actually_scheduled
    req_stats.kda_restore_tier = obs.kda_restore_tier
    req_stats.kda_restore_latency_ms = obs.kda_restore_latency_ms
    req_stats.mla_restore_latency_ms = obs.mla_restore_latency_ms


def calc_replay_tokens(mla_potential_hit: int, kda_recoverable_pos: int, total_input: int) -> tuple:
    """Calculate retreat_gap and replay_tokens.

    Returns (retreat_gap, replay_tokens, new_tokens, effective_prefix).

    Based on the F5 correction:
    - effective_prefix = min(mla_potential, kda_recoverable)
    - retreat_gap = max(0, mla_potential - kda_recoverable)
    - replay_tokens = retreat_gap (KDA needs to recompute the gap)
    - new_tokens = total_input - mla_potential (actual new input beyond MLA)
    - extend_range.length = total_input - effective_prefix = new_tokens + replay_tokens
    """
    effective_prefix = min(mla_potential_hit, kda_recoverable_pos)
    retreat_gap = max(0, mla_potential_hit - kda_recoverable_pos)
    new_tokens = max(0, total_input - mla_potential_hit)
    replay_tokens = retreat_gap  # KDA must recompute the gap
    return retreat_gap, replay_tokens, new_tokens, effective_prefix


# [v2-fix A1] Per-rid match result storage (populated in wrapped_match_prefix)
_MATCH_RESULTS: dict = {}  # rid -> {full_kv_hit, device_hit, host_hit, mamba_host_hit, best_match_node}


def store_match_result(rid: str, full_kv_hit: int, device_hit: int,
                        host_hit: int, mamba_host_hit: int, best_match_node: int):
    """Store match_prefix result per-rid for later use in get_new_batch_prefill."""
    _MATCH_RESULTS[rid] = {
        "full_kv_hit": full_kv_hit,
        "device_hit": device_hit,
        "host_hit": host_hit,
        "mamba_host_hit": mamba_host_hit,
        "best_match_node": best_match_node,
    }


def store_match_result_minimal(rid: str, best_match_node: int, host_hit: int = 0,
                               mamba_host_hit: int = 0):
    """Store minimal match result (used when full match_prefix data is unavailable,
    e.g. from init_load_back hook which has rid but not full_kv_hit)."""
    if rid not in _MATCH_RESULTS:
        _MATCH_RESULTS[rid] = {}
    _MATCH_RESULTS[rid]["best_match_node"] = best_match_node
    _MATCH_RESULTS[rid]["host_hit"] = host_hit
    _MATCH_RESULTS[rid]["mamba_host_hit"] = mamba_host_hit


def get_match_result(rid: str) -> dict:
    """Get stored match result for a rid. Returns empty dict if not found."""
    return _MATCH_RESULTS.get(rid, {})


def update_match_result(rid: str, key: str, value):
    """Update a field in the stored match result (e.g. init_load_back tier)."""
    if rid in _MATCH_RESULTS:
        _MATCH_RESULTS[rid][key] = value


def clear_match_results():
    """Clear all stored match results (called at metrics dump time)."""
    _MATCH_RESULTS.clear()


_MAMBA_EVICTABLE_OVERRIDE: int = 0  # HiSim-managed counter


def inc_mamba_evictable(n: int = 1):
    """Increment HiSim-managed mamba evictable counter."""
    global _MAMBA_EVICTABLE_OVERRIDE
    _MAMBA_EVICTABLE_OVERRIDE += n


def dec_mamba_evictable(n: int = 1):
    """Decrement HiSim-managed mamba evictable counter (on eviction)."""
    global _MAMBA_EVICTABLE_OVERRIDE
    _MAMBA_EVICTABLE_OVERRIDE = max(0, _MAMBA_EVICTABLE_OVERRIDE - n)


def get_mamba_evictable() -> int:
    """Get HiSim-managed mamba evictable counter."""
    return _MAMBA_EVICTABLE_OVERRIDE


def reset_mamba_evictable():
    """Reset HiSim-managed counter (at metrics dump)."""
    global _MAMBA_EVICTABLE_OVERRIDE
    _MAMBA_EVICTABLE_OVERRIDE = 0


def bytes_per_mamba_slot() -> int:
    return _MAMBA_SLOT_BYTES


def bytes_per_mla_token() -> int:
    return _MLA_ENTRY_BYTES


# ============================================================================
# [A2] Shadow State Registry — lightweight checkpoint identity tracking
# ============================================================================
# HiSim does not execute real KDA kernels, so a slot ID alone cannot prove
# the state content is correct.  The shadow registry tags every checkpoint
# with a semantic identity (donate_rid, track_position, prefix_hash) and
# follows it through the L1→L2→L3→restore chain.
#
# All methods are no-ops when HISIM_KDA_OBS is not set.

_SHADOW: dict = {}  # slot_hash -> shadow record
_SHADOW_FWD: dict = {}  # rid -> {slot_hash, ready, consumed}


def shadow_register_checkpoint(slot_hash: str, *, rid: str = "",
                                node_id: int = -1, track_position: int = -1,
                                prefix_hash: str = "", device_slot: int = -1):
    """Tag a freshly-donated KDA checkpoint with its semantic identity."""
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    if rec is None:
        rec = {"donate_rid": rid, "node_id": node_id,
               "track_position": track_position, "prefix_hash": prefix_hash,
               "device_slot": device_slot, "generation": 0,
               "l2_slot": -1, "l3_key": "",
               "restored_slot": -1, "ready": False, "consumed": False,
               "state": "donated"}
        _SHADOW[slot_hash] = rec
    else:
        rec["generation"] = rec.get("generation", 0) + 1
        rec["donate_rid"] = rid
        rec["node_id"] = node_id
        rec["track_position"] = track_position
        rec["device_slot"] = device_slot
        rec["restored_slot"] = -1
        rec["ready"] = False
        rec["consumed"] = False
        rec["state"] = "donated"
    log_event("shadow_register", rid=rid, node_id=node_id,
              checkpoint_pos=track_position, layer="L1",
              extra={"slot_hash": slot_hash, "device_slot": device_slot,
                     "prefix_hash": prefix_hash[:16],
                     "generation": rec["generation"]})


def shadow_backup_l2(slot_hash: str, l2_slot: int):
    """Record L1→L2 backup (D2H) for a checkpoint."""
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    if rec is not None:
        rec["l2_slot"] = l2_slot
        rec["state"] = "l2_backed"
    log_event("backup_l2", node_id=_shadow_nid(slot_hash), layer="L2",
              extra={"slot_hash": slot_hash, "l2_slot": l2_slot})


def shadow_backup_l3(slot_hash: str):
    """Record L2→L3 overflow eviction (write-through or overflow)."""
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    if rec is not None:
        rec["l3_key"] = slot_hash
        rec["state"] = "l3_backed"
    log_event("backup_l3", node_id=_shadow_nid(slot_hash), layer="L3",
              extra={"slot_hash": slot_hash})


def shadow_restore_l2(slot_hash: str, device_slot: int):
    """Record L2→L1 load-back (H2D) for a checkpoint.

    Sets ready=True — the mock load_back is synchronous, so the data is
    available immediately after this call.  In a real async system the
    ready flag would be set by the ACK callback.
    """
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    if rec is not None:
        rec["restored_slot"] = device_slot
        rec["ready"] = True
        rec["state"] = "restored_l2"
    log_event("restore_l2", node_id=_shadow_nid(slot_hash), layer="L1",
              reason="restore",
              extra={"slot_hash": slot_hash, "device_slot": device_slot,
                     "ready": True})


def shadow_restore_l3(slot_hash: str, device_slot: int):
    """Record L3→L2→L1 restore for a checkpoint."""
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    if rec is not None:
        rec["restored_slot"] = device_slot
        rec["ready"] = True
        rec["state"] = "restored_l3"
    log_event("restore_l3", node_id=_shadow_nid(slot_hash), layer="L1",
              reason="restore",
              extra={"slot_hash": slot_hash, "device_slot": device_slot,
                     "ready": True})


def shadow_mark_forward_consume(rid: str, slot_hash: str):
    """Mark that a forward pass consumed the restored KDA state."""
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    ready = rec["ready"] if rec else False
    if rec is not None:
        rec["consumed"] = True
        rec["state"] = "consumed"
    _SHADOW_FWD[rid] = {"slot_hash": slot_hash, "ready": ready, "consumed": True}
    log_event("forward_consume", rid=rid, node_id=_shadow_nid(slot_hash),
              layer="L1",
              extra={"slot_hash": slot_hash, "was_ready": ready})


def shadow_mark_ready(slot_hash: str, tier: str = "L1"):
    """Mark a checkpoint as ready for forward consumption (no transfer needed).

    Used for L1 hits where the state is already device-resident.
    """
    if not is_obs_enabled():
        return
    rec = _SHADOW.get(slot_hash)
    if rec is not None:
        rec["ready"] = True
        if rec["state"] == "donated":
            rec["state"] = f"l1_ready_{tier}"
    log_event("mark_ready", node_id=_shadow_nid(slot_hash), layer=tier,
              extra={"slot_hash": slot_hash, "ready": True})


def shadow_verify(rid: str, slot_hash: str) -> dict:
    """Verify the restored slot matches the original checkpoint identity.

    Returns a dict with:
      - identity_ok: slot_hash exists in registry
      - ready: transfer was marked complete before forward consume
      - consumed: forward consumed this state
      - state: current shadow state
      - generation: slot generation (reuse count)
    """
    if not is_obs_enabled():
        return {}
    rec = _SHADOW.get(slot_hash, {})
    return {
        "identity_ok": slot_hash in _SHADOW,
        "ready": rec.get("ready", False),
        "consumed": rec.get("consumed", False),
        "state": rec.get("state", "unknown"),
        "generation": rec.get("generation", 0),
        "donate_rid": rec.get("donate_rid", ""),
        "track_position": rec.get("track_position", -1),
        "restored_slot": rec.get("restored_slot", -1),
    }


def shadow_get_all() -> dict:
    """Return a snapshot of the shadow registry (for metrics dump)."""
    return {k: dict(v) for k, v in _SHADOW.items()}


def shadow_clear():
    """Clear the shadow registry (called at metrics dump time)."""
    _SHADOW.clear()
    _SHADOW_FWD.clear()


def _shadow_nid(slot_hash: str) -> int:
    """Extract node_id from slot_hash (format: kda_{nid}_{i})."""
    try:
        parts = slot_hash.split("_")
        return int(parts[1]) if len(parts) >= 2 else -1
    except (ValueError, IndexError):
        return -1
