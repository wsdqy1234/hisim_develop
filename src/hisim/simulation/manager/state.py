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
    # KDA/Mamba state timing (separate from MLA KV, take max in clock model)
    _hicache_kda_l2_load_dur: float = 0
    _hicache_kda_l2_backup_dur: float = 0
    _hicache_kda_l3_backup_dur: float = 0
    _last_kda_l2_load_dur: float = 0
    _last_kda_l2_backup_dur: float = 0

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
        cls._hicache_kda_l2_load_dur = 0
        cls._hicache_kda_l2_backup_dur = 0
        cls._hicache_kda_l3_backup_dur = 0
        cls._last_kda_l2_load_dur = 0
        cls._last_kda_l2_backup_dur = 0

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur

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
        cls._global_clock += dur

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        cls._global_clock = clock

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

    # ---- KDA/Mamba state timing (parallel to MLA, take max in clock) ----

    @classmethod
    def inc_hicache_kda_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_kda_l2_load_dur += dur

    @classmethod
    def inc_hicache_kda_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_kda_l2_backup_dur += dur

    @classmethod
    def inc_hicache_kda_l3_backup_dur(cls, dur: float) -> None:
        cls._hicache_kda_l3_backup_dur += dur

    @classmethod
    def pop_hicache_kda_l2_load_dur(cls) -> float:
        dur = cls._hicache_kda_l2_load_dur
        cls._hicache_kda_l2_load_dur = 0
        return dur

    @classmethod
    def pop_hicache_kda_l2_backup_dur(cls) -> float:
        dur = cls._hicache_kda_l2_backup_dur
        cls._hicache_kda_l2_backup_dur = 0
        return dur

    @classmethod
    def pop_hicache_kda_l3_backup_dur(cls) -> float:
        dur = cls._hicache_kda_l3_backup_dur
        cls._hicache_kda_l3_backup_dur = 0
        return dur

    @classmethod
    def set_last_kda_l2_load_dur(cls, dur: float) -> None:
        cls._last_kda_l2_load_dur = dur

    @classmethod
    def get_last_kda_l2_load_dur(cls) -> float:
        return cls._last_kda_l2_load_dur

    @classmethod
    def set_last_kda_l2_backup_dur(cls, dur: float) -> None:
        cls._last_kda_l2_backup_dur = dur

    @classmethod
    def get_last_kda_l2_backup_dur(cls) -> float:
        return cls._last_kda_l2_backup_dur
