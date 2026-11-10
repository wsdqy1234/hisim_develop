import math


class StateManager:
    _iteration: int = 0
    _global_clock: float = 0
    _last_inference_dur: float = 0
    _current_inference_dur: float = 0
    _hicache_l2_load_dur: float = 0
    _hicache_l2_backup_dur: float = 0
    _hicache_l3_backup_dur: float = 0
    _last_l2_load_dur: float = 0
    _last_l2_backup_dur: float = 0
    _last_l3_write_overflow: float = 0
    _l3_direct_load_count: int = 0
    _hicache_l3_direct_load_dur: float = 0
    # HBF L1.5 tier timing
    _hicache_hbf_backup_dur: float = 0
    _hicache_hbf_load_dur: float = 0
    _last_hbf_backup_dur: float = 0
    _last_hbf_load_dur: float = 0
    # L3 read timing (Phase 2d): SSD read duration from calc_prefetch_pages.
    # Overlaps with inference (best_effort); only the overflow (read not
    # completing within the round budget) delays data availability.
    _hicache_l3_load_dur: float = 0

    @classmethod
    def reset(cls):
        cls._iteration = 0
        cls._global_clock = 0
        cls._last_inference_dur = 0
        cls._current_inference_dur = 0
        cls._hicache_l2_backup_dur = 0
        cls._hicache_l2_load_dur = 0
        cls._hicache_l3_backup_dur = 0
        cls._last_l2_load_dur = 0
        cls._last_l2_backup_dur = 0
        cls._last_l3_write_overflow = 0
        cls._l3_direct_load_count = 0
        cls._hicache_l3_direct_load_dur = 0
        cls._hicache_hbf_backup_dur = 0
        cls._hicache_hbf_load_dur = 0
        cls._last_hbf_backup_dur = 0
        cls._last_hbf_load_dur = 0
        cls._hicache_l3_load_dur = 0

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        # Phase 3E: Use max (parallel events within a round), not sum
        # This matches HBF's MAX semantics for fair concurrent comparison
        if dur > cls._hicache_l2_load_dur:
            cls._hicache_l2_load_dur = dur

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        # Phase 3E: Use max (parallel events within a round), not sum
        if dur > cls._hicache_l2_backup_dur:
            cls._hicache_l2_backup_dur = dur

    @classmethod
    def inc_hicache_l3_backup_dur(cls, dur: float) -> None:
        cls._hicache_l3_backup_dur += dur

    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0
        return dur

    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0
        return dur

    @classmethod
    def pop_hicache_l3_backup_dur(cls) -> float:
        dur = cls._hicache_l3_backup_dur
        cls._hicache_l3_backup_dur = 0
        return dur

    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock

    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        if not math.isfinite(dur) or dur < 0:
            raise ValueError(f"Simulation clock requires a finite nonnegative step, got {dur}")
        cls._global_clock += dur

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        if not math.isfinite(clock) or clock < cls._global_clock:
            raise ValueError(
                f"Simulation clock cannot move from {cls._global_clock} to {clock}; "
                "use reset() only at an experiment boundary"
            )
        cls._global_clock = clock

    @classmethod
    def advance_global_clock_to(cls, clock: float) -> None:
        """Advance to a due event; past arrivals never rewind execution."""
        if not math.isfinite(clock):
            raise ValueError(f"Simulation event time must be finite, got {clock}")
        if clock > cls._global_clock:
            cls.set_global_clock(clock)

    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur

    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur

    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur

    @classmethod
    def set_last_l2_load_dur(cls, dur: float) -> None:
        cls._last_l2_load_dur = dur

    @classmethod
    def get_last_l2_load_dur(cls) -> float:
        return cls._last_l2_load_dur

    @classmethod
    def set_last_l2_backup_dur(cls, dur: float) -> None:
        cls._last_l2_backup_dur = dur

    @classmethod
    def get_last_l2_backup_dur(cls) -> float:
        return cls._last_l2_backup_dur

    @classmethod
    def set_last_l3_write_overflow(cls, dur: float) -> None:
        cls._last_l3_write_overflow = dur

    @classmethod
    def get_last_l3_write_overflow(cls) -> float:
        return cls._last_l3_write_overflow

    @classmethod
    def inc_l3_direct_load_count(cls, count: int) -> None:
        cls._l3_direct_load_count += count

    @classmethod
    def get_l3_direct_load_count(cls) -> int:
        return cls._l3_direct_load_count

    @classmethod
    def pop_l3_direct_load_count(cls) -> int:
        count = cls._l3_direct_load_count
        cls._l3_direct_load_count = 0
        return count

    @classmethod
    def inc_hicache_l3_direct_load_dur(cls, dur: float) -> None:
        cls._hicache_l3_direct_load_dur += dur

    @classmethod
    def pop_hicache_l3_direct_load_dur(cls) -> float:
        dur = cls._hicache_l3_direct_load_dur
        cls._hicache_l3_direct_load_dur = 0
        return dur

    # ---- HBF L1.5 timing ----
    @classmethod
    def inc_hicache_hbf_backup_dur(cls, dur: float) -> None:
        # Use max (parallel events within a round), not sum
        if dur > cls._hicache_hbf_backup_dur:
            cls._hicache_hbf_backup_dur = dur

    @classmethod
    def inc_hicache_hbf_load_dur(cls, dur: float) -> None:
        # Use max (parallel events within a round), not sum
        if dur > cls._hicache_hbf_load_dur:
            cls._hicache_hbf_load_dur = dur

    @classmethod
    def pop_hicache_hbf_backup_dur(cls) -> float:
        dur = cls._hicache_hbf_backup_dur
        cls._hicache_hbf_backup_dur = 0
        return dur

    @classmethod
    def pop_hicache_hbf_load_dur(cls) -> float:
        dur = cls._hicache_hbf_load_dur
        cls._hicache_hbf_load_dur = 0
        return dur

    @classmethod
    def set_last_hbf_backup_dur(cls, dur: float) -> None:
        cls._last_hbf_backup_dur = dur

    @classmethod
    def get_last_hbf_backup_dur(cls) -> float:
        return cls._last_hbf_backup_dur

    @classmethod
    def set_last_hbf_load_dur(cls, dur: float) -> None:
        cls._last_hbf_load_dur = dur

    @classmethod
    def get_last_hbf_load_dur(cls) -> float:
        return cls._last_hbf_load_dur

    # ---- L3 read timing (Phase 2d) ----
    @classmethod
    def inc_hicache_l3_load_dur(cls, dur: float) -> None:
        # Use max (parallel reads within a round), consistent with HBF semantics
        if dur > cls._hicache_l3_load_dur:
            cls._hicache_l3_load_dur = dur

    @classmethod
    def pop_hicache_l3_load_dur(cls) -> float:
        dur = cls._hicache_l3_load_dur
        cls._hicache_l3_load_dur = 0
        return dur
