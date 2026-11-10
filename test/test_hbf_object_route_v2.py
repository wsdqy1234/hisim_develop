"""Independent resource/slot assertions for the research trace correction.

These tests inspect real WIN counters and time cursors, not returned WIN
labels, so a cosmetic relabel cannot satisfy the routing contract.
"""
import pytest

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel, WinSpec

KV = 114688


def make_pool(**kw):
    return HbfWinPoolModel(Hbf8WinConfig(**kw))


def test_stripe_target_is_real_resource_and_partial_restore_stays_there():
    pool = make_pool(placement="striped", stripe_block_tokens=256)
    written = pool.submit_object(0, range(768, 1024), KV, is_read=False)
    assert [w.io_count for w in pool.wins] == [0, 0, 0, 1, 0, 0, 0, 0]
    assert pool.wins[3].bytes_written == 256 * KV
    assert pool.wins[0].media_cursor() == 0
    read = pool.submit_object(1, range(800, 830), KV, is_read=True)
    assert pool.wins[3].bytes_read_seq == 30 * KV
    assert sum(w.io_count for w in pool.wins) == 2
    assert [w.used_bytes for w in pool.wins] == pool.registry.win_bytes
    assert sum(w.used_bytes for w in pool.wins) == 256 * KV
    assert not pool._objects  # no read-fragment capacity allocation
    assert written.trace_details["sub_ios"][0]["win"] == 3
    assert read.trace_details["sub_ios"][0]["win"] == 3


def test_hotspot_nonzero_resource_competes_and_other_win_is_independent():
    pool = make_pool(placement="hotspot", hotspot_win=6)
    first = pool.submit_object(0, range(256), KV, is_read=False)
    second = pool.submit_object(0, range(256, 512), KV, is_read=False)
    assert pool.wins[6].io_count == 2
    assert second.media_start == pytest.approx(first.media_completion)
    other = pool.submit(0, 256 * KV, is_read=False, win_id=1)
    assert other.media_start == 0
    assert other.media_completion == pytest.approx(first.media_completion)
    assert pool.wins[0].io_count == 0


def test_free_fragment_reuse_and_generations_do_not_leak_capacity():
    pool = make_pool(placement="striped", stripe_block_tokens=16)
    initial = pool.submit_object(0, range(32, 64), KV, is_read=False,
                                 object_key=42)
    pool.notify_slots_freed(range(32, 48))
    assert sum(w.used_bytes for w in pool.wins) == 16 * KV
    pool.submit_object(1, range(48, 56), KV, is_read=True)
    assert sum(w.used_bytes for w in pool.wins) == 16 * KV
    replacement = pool.submit_object(2, range(32, 64), KV, is_read=False,
                                     object_key=42)
    assert sum(w.used_bytes for w in pool.wins) == 32 * KV
    assert replacement.trace_details["object_refs"] == [{"key": 42, "generation": 2}]
    pool.notify_slots_freed(range(32, 64))
    assert all(w.used_bytes == 0 for w in pool.wins)
    assert not pool.registry.slot_owner
    reused = pool.submit_object(3, range(32, 64), KV, is_read=False,
                                object_key=42)
    assert reused.trace_details["object_refs"][0]["generation"] == 3
    assert pool.registry.verify_conservation()["slots_consistent"]
    assert pool.get_io_events()[0] == initial.trace_details


def test_overlapping_new_object_shrinks_old_owner_without_double_charge():
    pool = make_pool(placement="striped", stripe_block_tokens=8)
    pool.submit_object(0, range(32), KV, is_read=False)
    pool.submit_object(1, range(16, 48), KV, is_read=False)
    state = pool.registry.verify_conservation()
    assert state["slots_consistent"] and state["bytes_consistent"]
    assert state["total_slots"] == 48
    assert sum(w.used_bytes for w in pool.wins) == 48 * KV
    pool.notify_slots_freed(range(48))
    assert sum(w.used_bytes for w in pool.wins) == 0


def test_restore_waits_for_all_backups_and_keeps_resource_wait_separate():
    pool = make_pool(placement="balanced")
    fast = pool.submit_object(0, range(8), KV, is_read=False)
    slow = pool.submit_object(0, range(8, 520), KV, is_read=False)
    read = pool.submit_object(0, range(520), KV, is_read=True)
    details = read.trace_details
    assert slow.media_completion > fast.media_completion
    assert details["effective_arrival_s"] == pytest.approx(slow.media_completion)
    assert details["dependency_wait_s"] == pytest.approx(slow.media_completion)
    assert len(details["object_refs"]) == 2
    assert all(s["arrival_s"] >= slow.media_completion for s in details["sub_ios"])
    assert read.total == pytest.approx(details["completion_s"])
    assert read.queue_wait >= details["dependency_wait_s"]


def test_missing_striped_restore_has_no_phantom_resource_or_useful_bytes():
    pool = make_pool(placement="striped")
    pool.submit_object(0, range(8), KV, is_read=False)
    pool.notify_slots_freed(range(8))
    counts = [w.io_count for w in pool.wins]
    miss = pool.submit_object(1, range(8), KV, is_read=True)
    assert [w.io_count for w in pool.wins] == counts
    assert miss.trace_details["status"] == "miss"
    assert miss.trace_details["bytes"] == 0
    assert miss.trace_details["requested_bytes"] == 8 * KV
    assert not miss.trace_details["sub_ios"]


def test_capacity_preflight_is_atomic_and_replacement_reuses_own_space():
    pool = make_pool(n_wins=1, placement="balanced",
                     win=WinSpec(capacity_bytes=8 * KV))
    pool.submit_object(0, range(8), KV, is_read=False, object_key=9)
    pool.submit_object(1, range(8), KV, is_read=False, object_key=9)
    assert pool.wins[0].used_bytes == 8 * KV
    with pytest.raises(ValueError, match="capacity exceeded"):
        pool.submit_object(2, range(8, 9), KV, is_read=False)
    assert pool.wins[0].used_bytes == 8 * KV
    assert pool.registry.verify_conservation()["slots_consistent"]
    assert len(pool.get_io_events()) == 2


def test_trace_preserves_full_precision_history_and_unambiguous_service():
    pool = make_pool(placement="hotspot", hotspot_win=4)
    arrival = 7.1234567890123
    result = pool.submit_object(arrival, range(16), KV, is_read=False)
    event = pool.get_io_events()[0]
    child = event["sub_ios"][0]
    assert event["arrival_s"] == arrival
    assert child["win"] == 4
    assert child["completion_s"] == pytest.approx(arrival + result.total)
    assert child["bytes"] == sum(b - a for a, b in child["extents"]) * KV
    assert set(child["slots"]) == set(range(16))
    pool.notify_slots_freed(range(16))
    assert pool.get_io_events()[0] == event
    event["sub_ios"][0]["win"] = 0
    assert pool.get_io_events()[0]["sub_ios"][0]["win"] == 4
    summary = pool.summary()
    win = summary["per_win"][4]
    assert win["media_cursor_span_s"] > 7
    assert win["media_service_time_s"] < .001
    assert "not utilization" in summary["metric_definitions"]["media_busy_s"]


def test_legacy_submit_round_robin_and_explicit_free_capacity_are_preserved():
    pool = make_pool()
    assert [pool.submit(0, 8 * KV, is_read=False, object_id=i).win
            for i in range(8)] == list(range(8))
    assert sum(w.used_bytes for w in pool.wins) == 64 * KV
    assert pool.free_object(0)
    assert pool.wins[0].used_bytes == 0
    assert sum(w.used_bytes for w in pool.wins) == 56 * KV
