#!/usr/bin/env python3
"""Phase 5B first-round validation (model correctness only).

Six required scenarios (+ shared-interface variants):
  V1  single WIN, 38.9MB sequential read
  V2  single WIN, 38.9MB random read
  V3  single WIN, 38.9MB write
  V4  8 WINs, each concurrently serving a 38.9MB sequential read
      (a: no interface  b: package-shared 112GB/s)
  V5  8 WINs, each concurrently serving a 38.9MB write
  V6  8 WINs, mixed read/write with hotspot placement + shared interface

Records per WIN: I/O bytes, media time, queue wait, completion,
throughput, interface-constrained flag.

NOTE: this validates MODEL CORRECTNESS and hardware-config feasibility
only. It does NOT claim the real HBF reaches 80% efficiency.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel

GB = 1e9
KV = 114688
READ_IO = 339 * KV      # 38.88 MB
WRITE_IO = 340 * KV     # 38.99 MB

OUT_DIR = "/tmp/p5b_validation"
os.makedirs(OUT_DIR, exist_ok=True)


def print_win_table(pool, label):
    print(f"\n--- {label} ---")
    hdr = (f"{'WIN':>3} {'rd_seq_MB':>10} {'rd_rand_MB':>10} "
           f"{'wr_MB':>7} {'mediaBusy_ms':>13} {'qw_max_μs':>10} "
           f"{'ioConstr':>8}")
    print(hdr)
    for pw in pool.per_win_summary():
        print(f"{pw['win']:>3} {pw['bytes_read_seq_gb']*1000:>10.1f} "
              f"{pw['bytes_read_rand_gb']*1000:>10.1f} "
              f"{pw['bytes_written_gb']*1000:>7.1f} "
              f"{pw['media_busy_s']*1000:>13.3f} "
              f"{pw['queue_wait_max_us']:>10.1f} "
              f"{pw['n_interface_constrained']:>8}")


def run(label, cfg, ios):
    """ios: list of (bytes, is_read, mode, object_id)."""
    pool = HbfWinPoolModel(cfg)
    pool.begin_batch(0.0)
    for b, rd, mode, oid in ios:
        pool.submit(0.0, b, is_read=rd, access_mode=mode, object_id=oid)
    s = pool.summary()
    print_win_table(pool, label)
    print(f"  span={s['batch_span_s']*1000:.2f}ms  "
          f"read={s['read_throughput_gbps']:.1f}GB/s  "
          f"write={s['write_throughput_gbps']:.1f}GB/s  "
          f"(caps: seq {s['aggregate_cap_read_gbps']:.0f} / "
          f"write {s['aggregate_cap_write_gbps']:.0f} GB/s"
          + (f"  sharedIC {s['shared_interface_bw_gbps']}GB/s"
             if s["shared_interface_bw_gbps"] else "") + ")")
    return s


def main():
    print("=" * 88)
    print("Phase 5B first-round validation — MODEL CORRECTNESS ONLY")
    print("(does NOT claim real HBF 80% efficiency)")
    print("=" * 88)

    results = {}

    # ---- V1-V3: single WIN ----
    single = Hbf8WinConfig(n_wins=1)
    results["V1"] = run("V1: 1 WIN × 38.88MB SEQ READ",
                        single, [(READ_IO, True, "sequential", "v1")])
    results["V2"] = run("V2: 1 WIN × 38.88MB RANDOM READ",
                        single, [(READ_IO, True, "random", "v2")])
    results["V3"] = run("V3: 1 WIN × 38.99MB WRITE",
                        single, [(WRITE_IO, False, None, "v3")])

    # ---- V4: 8 WINs, each a 38.88MB seq read (concurrent) ----
    ios8r = [(READ_IO, True, "sequential", f"r{i}") for i in range(8)]
    results["V4a"] = run("V4a: 8 WIN × 38.88MB SEQ READ (no interface)",
                         Hbf8WinConfig(), ios8r)
    results["V4b"] = run(
        "V4b: 8 WIN × 38.88MB SEQ READ (shared 112GB/s interface)",
        Hbf8WinConfig(package_shared_interface_bw=112 * GB), ios8r)

    # ---- V5: 8 WINs, each a 38.99MB write (concurrent) ----
    ios8w = [(WRITE_IO, False, None, f"w{i}") for i in range(8)]
    results["V5"] = run("V5: 8 WIN × 38.99MB WRITE",
                        Hbf8WinConfig(), ios8w)

    # ---- V6: mixed R/W + hotspot placement + shared interface ----
    cfg6 = Hbf8WinConfig(placement="hotspot", hotspot_win=0,
                         package_shared_interface_bw=112 * GB)
    ios6 = (
        [(READ_IO, True, "sequential", f"h{i}") for i in range(4)] +
        [(WRITE_IO, False, None, f"hw{i}") for i in range(2)]
    )
    results["V6"] = run("V6: 8 WIN mixed 4R+2W hotspot@win0 + shared IC",
                        cfg6, ios6)

    # ---- sanity assertions (model correctness gates) ----
    print("\n" + "=" * 88)
    checks = [
        ("V1 seq read ≈ 194.4μs", abs(
            results["V1"]["batch_span_s"] - READ_IO / 200e9) < 1e-6),
        ("V2 rand read ≈ 388.8μs", abs(
            results["V2"]["batch_span_s"] - READ_IO / 100e9) < 1e-6),
        ("V3 write ≈ 3.899ms", abs(
            results["V3"]["batch_span_s"] - WRITE_IO / 10e9) < 1e-6),
        ("V4a aggregate = 1600GB/s", abs(
            results["V4a"]["read_throughput_gbps"] - 1600) < 1.0),
        ("V4b aggregate ≤ 112GB/s", results["V4b"]["read_throughput_gbps"]
         <= 112 * 1.02),
        ("V5 aggregate = 80GB/s", abs(
            results["V5"]["write_throughput_gbps"] - 80) < 0.1),
        ("V6 hotspot concentrated on win0",
         results["V6"]["per_win"][0]["io_count"] >= 4),
    ]
    ok = True
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
        ok = ok and passed

    # ---- effective config snapshot ----
    eff_path = os.path.join(OUT_DIR, "hbf_win_effective_config.json")
    Hbf8WinConfig().dump_effective_json(eff_path)
    with open(eff_path) as f:
        cfg_preview = json.load(f)
    print(f"\nEffective config snapshot → {eff_path}")
    print(json.dumps(cfg_preview, indent=1)[:900])

    with open(os.path.join(OUT_DIR, "validation_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nResults → {OUT_DIR}/validation_results.json")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
