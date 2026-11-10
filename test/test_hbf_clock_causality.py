"""Regression for local batch ownership and monotonic virtual event time."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from hisim.simulation.manager import StateManager
from hisim.simulation.sglang.sglang_hook import C_SchedulerHook
from hisim.simulation.types import MockSimulationMode, RequestStats


def batch(finished=False):
    return SimpleNamespace(reqs=[SimpleNamespace(finished=lambda: finished)])


def empty_batch():
    return SimpleNamespace(reqs=[])


@pytest.fixture
def scheduler(monkeypatch):
    class Scheduler:
        def __init__(self):
            pass

        def get_new_batch_prefill(self, running_batch=None):
            return SimpleNamespace(batch_to_run=None, running_batch=running_batch)

        def run_batch(self, *args, **kwargs):
            return None

        def process_batch_result(self, *args, **kwargs):
            return None

        def event_loop_normal(self, *args, **kwargs):
            return None

        def process_input_requests(self, requests, *args, **kwargs):
            self.dispatched = requests

    monkeypatch.setattr(C_SchedulerHook, "SIM_MODE", MockSimulationMode.OFFLINE)
    monkeypatch.setattr(C_SchedulerHook, "FUTURE_QUEUE", [(2.0, 0, object())])
    monkeypatch.setattr(C_SchedulerHook, "PARKED", {})
    monkeypatch.setattr(C_SchedulerHook, "REQUEST_STATS", defaultdict(RequestStats))
    monkeypatch.setattr(C_SchedulerHook, "LAST_CPU_TS", 0)
    C_SchedulerHook.hook(Scheduler)
    instance = Scheduler.__new__(Scheduler)
    instance.running_batch = empty_batch()
    instance.waiting_queue = []
    instance.last_batch = None
    instance.cur_batch_for_debug = None
    instance.chunked_req = None
    instance.result_queue = []
    StateManager.reset()
    StateManager.step_global_clock(0.005)
    yield instance
    StateManager.reset()


def test_locally_owned_decode_batch_prevents_jump_to_future_arrival(scheduler):
    scheduler.get_new_batch_prefill(batch())
    assert StateManager.get_global_clock() == 0.005


@pytest.mark.parametrize("view", ["running_batch", "cur_batch", "cur_batch_for_debug", "last_batch"])
def test_unfinished_batch_views_prevent_idle_jump(scheduler, view):
    setattr(scheduler, view, batch())
    scheduler.get_new_batch_prefill(empty_batch())
    assert StateManager.get_global_clock() == 0.005


def test_finished_stale_views_allow_idle_jump(scheduler):
    scheduler.last_batch = batch(finished=True)
    scheduler.cur_batch_for_debug = batch(finished=True)
    scheduler.get_new_batch_prefill(empty_batch())
    assert StateManager.get_global_clock() == 2.0


def test_due_future_event_does_not_rewind_clock(scheduler):
    StateManager.advance_global_clock_to(3.0)
    scheduler.get_new_batch_prefill(empty_batch())
    assert StateManager.get_global_clock() == 3.0


def test_prefetch_poll_does_not_advance_while_local_decode_is_active(scheduler):
    scheduler.waiting_queue = [object()]
    scheduler.get_new_batch_prefill(batch())
    assert StateManager.get_global_clock() == 0.005
    scheduler.get_new_batch_prefill(empty_batch())
    assert StateManager.get_global_clock() == 0.010


@pytest.mark.parametrize("pending", ["chunked_req", "result_queue"])
def test_pending_chunk_or_result_prevents_idle_jump(scheduler, pending):
    setattr(scheduler, pending, SimpleNamespace(finished=lambda: False)
            if pending == "chunked_req" else [object()])
    scheduler.get_new_batch_prefill(empty_batch())
    assert StateManager.get_global_clock() == 0.005


def test_debug_queue_start_and_first_received_request_preserve_execution_time(scheduler, monkeypatch):
    StateManager.advance_global_clock_to(3.0)
    monkeypatch.setattr(C_SchedulerHook, "_offline_collect_and_release",
                        classmethod(lambda cls, scheduler, requests, dispatch: requests))
    request_type = type("TokenizedGenerateReqInput", (), {})
    request = request_type()
    request.rid = "later-dispatch"
    request.input_ids = [1, 2, 3]
    request.sampling_params = SimpleNamespace(max_new_tokens=4, custom_params={
        "simulation": {"created_time": 2.0, "queue_start": 2.0}})
    scheduler.process_input_requests([request])
    stats = C_SchedulerHook.REQUEST_STATS[request.rid]
    assert StateManager.get_global_clock() == 3.0
    assert stats.created_time == 2.0 and stats.queue_start == 3.0
    assert scheduler.dispatched == [request]


def test_clock_reset_is_explicit_and_backward_steps_are_rejected():
    StateManager.reset()
    try:
        StateManager.step_global_clock(1.0)
        with pytest.raises(ValueError, match="cannot move"):
            StateManager.set_global_clock(0.5)
        with pytest.raises(ValueError, match="nonnegative"):
            StateManager.step_global_clock(-0.01)
        with pytest.raises(ValueError):
            StateManager.advance_global_clock_to(float("nan"))
        assert StateManager.get_global_clock() == 1.0
        StateManager.reset()
        assert StateManager.get_global_clock() == 0.0
    finally:
        StateManager.reset()
