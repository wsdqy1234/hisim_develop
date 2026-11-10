#!/usr/bin/env python3
"""Phase 4-Closure P1: PageLevelMediaSim unit tests.

Validates that the discrete-event simulator produces timing from model
BEHAVIOR (page splitting, channel striping, plane parallelism, read
pipelining, write trailing program latency), not from a closed-form
formula. These tests are the audit trail for the efficiency curves.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_page_sim import PageLevelMediaSim

CH = 32
PLANES = 32
PAGE = 16384
TR = 10e-6
TW = 100e-6
RATE = 1.6e9  # per-channel → aggregate 51.2 GB/s
REF = 51.2e9


def mk(**kw):
    sim = PageLevelMediaSim(channels=kw.get("channels", CH),
                            planes_per_channel=kw.get("planes", PLANES),
                            page_bytes=PAGE, read_latency=TR,
                            write_latency=TW, channel_rate=RATE)
    return sim


def test_single_page_read_latency_plus_transfer():
    """4KB (1 page, 1 channel): completion = tR + page transfer (serial)."""
    sim = mk()
    s = sim.submit(0.0, 4096, is_read=True)
    expect = TR + PAGE / RATE  # 10 + 10.24 = 20.24μs
    assert abs(s.media_time - expect) < 1e-9, f"{s.media_time*1e6} != {expect*1e6}"
    assert s.channels_used == 1, "1 page → 1 channel"
    eff = 4096 / s.media_time
    assert eff / REF < 0.01, f"4KB single-page efficiency should be <1%: {eff/REF:.2%}"
    print(f"✓ 4KB read: {s.media_time*1e6:.2f}μs = tR+transfer, "
          f"eff={eff/REF:.2%}, 1/32 channels")


def test_512kb_read_channel_parallel():
    """512KB = 32 pages → 32 channels, 1 page each: all planes read in
    parallel (tR), then 1 page transfer per channel. Completion =
    tR + transfer — exposes the initial tR latency the formula hid."""
    sim = mk()
    s = sim.submit(0.0, 512 * 1024, is_read=True)
    expect = TR + PAGE / RATE
    assert abs(s.media_time - expect) < 1e-9
    assert s.channels_used == 32
    eff = (512 * 1024) / s.media_time
    # 512KB / 20.24μs = 25.3 GB/s ≈ 49.4% (formula claimed 100%!)
    assert 0.45 < eff / REF < 0.52, f"512KB eff={eff/REF:.2%}, expect ~49%"
    print(f"✓ 512KB read: {s.media_time*1e6:.2f}μs, 32 channels, "
          f"eff={eff/REF:.1%} (formula said 100% — tR exposure was hidden)")


def test_16mb_read_pipelining():
    """16MB = 1024 pages, 32/channel, 32 planes → 1 page/plane.
    Channel is the bottleneck: tR + 32 × transfer. Plane reads overlap
    with channel transfer (pipelining)."""
    sim = mk()
    s = sim.submit(0.0, 16 * 1024 * 1024, is_read=True)
    expect = TR + 32 * (PAGE / RATE)  # 10 + 327.68 = 337.68μs
    assert abs(s.media_time - expect) < 1e-9
    eff = (16 * 1024 * 1024) / s.media_time
    assert 0.90 < eff / REF < 0.99, f"16MB eff={eff/REF:.2%}, expect ~97%"
    print(f"✓ 16MB read: {s.media_time*1e6:.2f}μs = tR + 32×transfer "
          f"(pipelined), eff={eff/REF:.1%}")


def test_read_pipeline_across_pages():
    """Two pages on one channel (different planes): 2nd page's array read
    overlaps 1st page's channel transfer → completion = tR + 2×transfer."""
    sim = PageLevelMediaSim(channels=1, planes_per_channel=2, page_bytes=PAGE,
                            read_latency=TR, write_latency=TW, channel_rate=RATE)
    s = sim.submit(0.0, 2 * PAGE, is_read=True)
    expect = TR + 2 * (PAGE / RATE)
    assert abs(s.media_time - expect) < 1e-9, \
        f"{s.media_time*1e6} != {expect*1e6} — pipelining broken"
    print(f"✓ read pipeline: 2 pages/1 channel = tR + 2×transfer "
          f"({s.media_time*1e6:.2f}μs) — array reads overlap transfers")


def test_single_page_write_trailing_program():
    """4KB write: channel transfer THEN program — serial for one page.
    completion = transfer + tW = 10.24 + 100 = 110.24μs (formula's max()
    hid the trailing program)."""
    sim = mk()
    s = sim.submit(0.0, 4096, is_read=False)
    expect = PAGE / RATE + TW
    assert abs(s.media_time - expect) < 1e-9, f"{s.media_time*1e6} != {expect*1e6}"
    print(f"✓ 4KB write: {s.media_time*1e6:.2f}μs = transfer + tW "
          f"(trailing program exposed)")


def test_write_plane_parallelism():
    """512KB write = 32 pages on 32 channels, each page: transfer + program.
    Programs on different planes/channels run in parallel → completion =
    transfer + tW per page (parallel)."""
    sim = mk()
    s = sim.submit(0.0, 512 * 1024, is_read=False)
    expect = PAGE / RATE + TW
    assert abs(s.media_time - expect) < 1e-9
    eff = (512 * 1024) / s.media_time
    # 512KB / 110.24μs = 4.65 GB/s ≈ 9.1%
    assert 0.07 < eff / REF < 0.11, f"512KB write eff={eff/REF:.2%}, expect ~9%"
    print(f"✓ 512KB write: {s.media_time*1e6:.2f}μs (32 planes parallel), "
          f"eff={eff/REF:.1%}")


def test_write_16mb():
    """16MB write: channel 327.68μs, last page program ends +100μs after
    its transfer → completion ≈ 32×transfer + tW = 427.68μs."""
    sim = mk()
    s = sim.submit(0.0, 16 * 1024 * 1024, is_read=False)
    expect = 32 * (PAGE / RATE) + TW
    assert abs(s.media_time - expect) < 1e-9
    eff = (16 * 1024 * 1024) / s.media_time
    assert 0.70 < eff / REF < 0.80, f"16MB write eff={eff/REF:.2%}, expect ~76%"
    print(f"✓ 16MB write: {s.media_time*1e6:.2f}μs, eff={eff/REF:.1%} "
          f"(trailing program is the write-path loss)")


def test_qd32_small_io_channel_fill():
    """32 concurrent 4KB reads: pages spread over all 32 channels →
    throughput 32× the single-I/O case (QD fills idle channels)."""
    sim = mk()
    t0 = 0.0
    last = 0.0
    for i in range(32):
        s = sim.submit(t0, 4096, is_read=True)
        last = max(last, s.media_completion)
    total = 32 * 4096
    agg = total / (last - t0)
    single = 4096 / (TR + PAGE / RATE)
    assert agg > single * 20, f"QD=32 should boost small-I/O BW ~32x: {agg/single:.1f}x"
    print(f"✓ QD=32 × 4KB: {agg/1e9:.2f}GB/s vs single {single/1e9:.2f}GB/s "
          f"({agg/single:.0f}x — channels filled by concurrent I/Os)")


def test_concurrent_large_io_bw_conservation():
    """4 concurrent 16MB reads: aggregate throughput must not exceed the
    channel aggregate BW (51.2 GB/s)."""
    sim = mk()
    last = 0.0
    for i in range(4):
        s = sim.submit(0.0, 16 * 1024 * 1024, is_read=True)
        last = max(last, s.media_completion)
    agg = 4 * 16 * 1024 * 1024 / last
    assert agg <= REF * 1.02, f"aggregate {agg/1e9:.1f}GB/s exceeds media BW"
    assert agg > REF * 0.90, f"aggregate {agg/1e9:.1f}GB/s too low"
    print(f"✓ 4×16MB concurrent: {agg/1e9:.1f}GB/s ≈ channel aggregate "
          f"(BW conserved, no oversubscription)")


def test_mechanism_switches():
    """Loss-attribution switches produce the expected deltas."""
    # page latency off → large read = pure channel time (100% eff)
    sim = mk()
    sim.enable_page_latency = False
    s = sim.submit(0.0, 16 * 1024 * 1024, is_read=True)
    assert abs(s.media_time - 32 * PAGE / RATE) < 1e-9
    # channel transfer off → single 4KB read = tR only
    sim = mk()
    sim.enable_channel_transfer = False
    s = sim.submit(0.0, 4096, is_read=True)
    assert abs(s.media_time - TR) < 1e-9
    # plane contention off → write with 2 pages on 1 plane: no plane wait
    sim = PageLevelMediaSim(channels=1, planes_per_channel=1, page_bytes=PAGE,
                            read_latency=TR, write_latency=TW, channel_rate=RATE)
    s_contended = sim.submit(0.0, 2 * PAGE, is_read=False)
    # 1 plane, 2 pages: page0 prog 10.24→110.24; page1 transfer waits for
    # channel (10.24→20.48), prog waits for plane (110.24→210.24)
    assert abs(s_contended.media_time - (PAGE / RATE + 2 * TW)) < 1e-9, \
        f"contended={s_contended.media_time*1e6}μs"
    sim2 = PageLevelMediaSim(channels=1, planes_per_channel=1, page_bytes=PAGE,
                             read_latency=TR, write_latency=TW, channel_rate=RATE)
    sim2.enable_plane_contention = False
    s_free = sim2.submit(0.0, 2 * PAGE, is_read=False)
    # no plane wait: page1 prog starts at its own transfer end
    assert abs(s_free.media_time - (2 * PAGE / RATE + TW)) < 1e-9
    delta = s_contended.media_time - s_free.media_time
    assert abs(delta - (TW - PAGE / RATE)) < 1e-9, \
        "plane contention cost = tW - one page transfer"
    print(f"✓ mechanism switches: page-lat/channel/plane toggles all "
          f"shift timing (plane contention cost = {delta*1e6:.0f}μs)")


def test_cross_batch_state_preserved():
    """Busy cursors persist across begin_batch (in-flight I/O)."""
    sim = mk()
    sim.begin_batch(0.0)
    s1 = sim.submit(0.0, 16 * 1024 * 1024, is_read=True)  # ~337.7μs
    # new batch starts at t=100μs — I/O still in flight
    sim.begin_batch(100e-6)
    s2 = sim.submit(100e-6, 4096, is_read=True)
    # the 4KB I/O's page lands on a channel still busy from batch 1
    assert s2.media_time > TR + PAGE / RATE, \
        "batch-2 I/O should see batch-1's busy channels"
    print(f"✓ cross-batch: batch-2 4KB read waited for batch-1 16MB "
          f"(media_time={s2.media_time*1e6:.1f}μs > 20.24μs)")


def test_kv_size_389mb():
    """38.9MB (KV 339 tokens @ 114,688 B/token): the actual workload size."""
    sim = mk()
    b = 339 * 114688
    sr = sim.submit(0.0, b, is_read=True)
    eff_r = b / sr.media_time
    sim2 = mk()
    sw = sim2.submit(0.0, 340 * 114688, is_read=False)
    eff_w = 340 * 114688 / sw.media_time
    print(f"✓ 38.9MB KV I/O: read {sr.media_time*1e6:.1f}μs "
          f"(eff={eff_r/REF:.1%}), write {sw.media_time*1e6:.1f}μs "
          f"(eff={eff_w/REF:.1%})")
    assert 0.93 < eff_r / REF < 0.99
    # write: trailing tW amortizes with pages/channel (75 pages → 868μs)
    assert 0.82 < eff_w / REF < 0.92, f"write eff={eff_w/REF:.2%}"


def run_all():
    tests = [
        test_single_page_read_latency_plus_transfer,
        test_512kb_read_channel_parallel,
        test_16mb_read_pipelining,
        test_read_pipeline_across_pages,
        test_single_page_write_trailing_program,
        test_write_plane_parallelism,
        test_write_16mb,
        test_qd32_small_io_channel_fill,
        test_concurrent_large_io_bw_conservation,
        test_mechanism_switches,
        test_cross_batch_state_preserved,
        test_kv_size_389mb,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({passed+failed} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
