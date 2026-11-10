#!/usr/bin/env python3
"""Phase 5D P4-B: capacity-benefit trace simulator at the 800GB scale.

Session-level KV trace replay with VIRTUAL OBJECTS (no CPU forward, no
physical memory): each session has a 340-token prefix, builds once and
reuses with agent-style spacing. Object-level LRU on L1 (small) and Host
(DRAM part + HBF part). HBF I/O timing from the 8-WIN envelope (striped
or balanced, real per-WIN resource cursors); DRAM from the 13.6GB/s path;
recompute from the E2E-calibrated per-token rate (14μs/tk).

Comparison protocol (per Phase 5D P4 requirements):
  - Equal-capacity: A (DRAM-only, Host=C) vs B (HBF 70% + DRAM 30%, SAME
    total C) — capacity story isolated from medium performance.
  - Expansion: C scenario (DRAM C + HBF 800GB extra) reported separately,
    never compared against a smaller DRAM pool as a "medium" claim.

Usage: python3 test_phase5d_capacity_trace.py
"""
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hisim.simulation.hbf_win_pool import Hbf8WinConfig, HbfWinPoolModel

GB = 1e9
KV = 114688            # bytes/token
PLEN = 340             # prefix tokens per session
RECOMPUTE_US_PER_TK = 14.0   # E2E-calibrated (Qwen3 CPU mock build TTFT)
DRAM_BW = 13.6e9
DRAM_T0 = 6.67e-6
L1_TOKENS = 1024       # device cache (drives backup traffic)


class SessionTrace:
    """Object-level KV cache trace simulator (virtual objects)."""

    def __init__(self, n_sessions, host_tokens, mode, hbf_frac=0.7,
                 reuse_rounds=2, spacing=10.0, seed=7):
        self.n_sessions = n_sessions
        self.host_tokens = host_tokens
        self.mode = mode                  # "A" DRAM-only | "B" mixed | "C" expansion
        self.hbf_frac = hbf_frac
        # per-object host placement: HBF via WIN pool, DRAM via arithmetic
        self.hbf_capacity = int(host_tokens * hbf_frac) if mode != "A" else 0
        if mode == "C":
            self.hbf_capacity = int(800 * GB / KV)      # extra 800GB
        self.dram_capacity = host_tokens - (int(host_tokens * hbf_frac)
                                            if mode == "B" else 0)
        self.win_pool = HbfWinPoolModel(Hbf8WinConfig(
            placement="striped", stripe_block_tokens=43))
        self.l1 = OrderedDict()           # prefix_id -> tokens (device)
        self.host_hbf = OrderedDict()     # prefix_id -> tokens
        self.host_dram = OrderedDict()    # prefix_id -> tokens
        self.clock = 0.0
        # metrics
        self.n_l1_hit = self.n_hbf_hit = self.n_dram_hit = self.n_recompute = 0
        self.ttfts = []
        self.backup_p95s = []
        self.hbf_read_bytes = self.hbf_write_bytes = 0.0
        self.dram_read_bytes = self.dram_write_bytes = 0.0
        # build the trace: builds first (t = s*0.1), then reuses AFTER the
        # whole build phase (host capacity genuinely gates hit rate)
        import random
        rng = random.Random(seed)
        self.events = []
        build_span = n_sessions * 0.1
        for s in range(n_sessions):
            self.events.append((s * 0.1, "build", s))
        reuse_base = build_span + 10.0
        for s in range(n_sessions):
            for r in range(reuse_rounds):
                t = reuse_base + s * 0.05 + r * 0.01 + rng.random() * 0.001
                self.events.append((t, "reuse", s))
        self.events.sort()

    def _l1_evict_to_fit(self, need):
        evicted = []
        while len(self.l1) and sum(self.l1.values()) + need > L1_TOKENS:
            pid, tok = self.l1.popitem(last=False)
            evicted.append((pid, tok))
        return evicted

    def _host_insert(self, pid, tokens):
        """Insert object into host. V0-like alternating placement (≈70% HBF
        / 30% DRAM), each medium evicts its OWN LRU when full — fair
        comparison vs mode A (identical LRU semantics, same total bytes).
        Returns (medium, io_duration)."""
        if self.mode == "A" or self.hbf_capacity < tokens:
            while len(self.host_dram) and \
                    sum(self.host_dram.values()) + tokens > self.dram_capacity:
                self.host_dram.popitem(last=False)
            self.host_dram[pid] = tokens
            dur = DRAM_T0 + tokens * KV / DRAM_BW
            self.dram_write_bytes += tokens * KV
            return "dram", dur
        # B/C: alternate placement (V0-like 70/30), per-medium LRU eviction
        self._ins_n = getattr(self, "_ins_n", 0) + 1
        to_hbf = (self._ins_n % 10) < 7
        hbf_used = sum(self.host_hbf.values())
        if to_hbf or self.dram_capacity < tokens:
            while len(self.host_hbf) and \
                    hbf_used + tokens > self.hbf_capacity:
                self.host_hbf.popitem(last=False)
                hbf_used = sum(self.host_hbf.values())
            if hbf_used + tokens <= self.hbf_capacity:
                self.host_hbf[pid] = tokens
                chunk = tokens * KV / 8
                last = 0.0
                for w in range(8):
                    st = self.win_pool.submit(self.clock, chunk, is_read=False,
                                              object_id=f"b{pid}@{w}")
                    last = max(last, max(st.media_completion,
                                         st.ic_completion))
                dur = last - self.clock
                self.hbf_write_bytes += tokens * KV
                return "hbf", dur
        # DRAM (not selected or HBF cannot hold even after eviction)
        while len(self.host_dram) and \
                sum(self.host_dram.values()) + tokens > self.dram_capacity:
            self.host_dram.popitem(last=False)
        self.host_dram[pid] = tokens
        dur = DRAM_T0 + tokens * KV / DRAM_BW
        self.dram_write_bytes += tokens * KV
        return "dram", dur

    def _restore(self, pid):
        """Restore an object from host. Returns (medium, duration, tokens)
        on hit, or None on miss (object not in host)."""
        if pid in self.host_hbf:
            tokens = self.host_hbf.pop(pid)
            chunk = tokens * KV / 8
            last = 0.0
            for w in range(8):
                st = self.win_pool.submit(self.clock, chunk, is_read=True,
                                          object_id=f"r{pid}@{w}")
                last = max(last, max(st.media_completion, st.ic_completion))
            self.hbf_read_bytes += tokens * KV
            return "hbf", last - self.clock, tokens
        if pid in self.host_dram:
            tokens = self.host_dram.pop(pid)
            dur = DRAM_T0 + tokens * KV / DRAM_BW
            self.dram_read_bytes += tokens * KV
            return "dram", dur, tokens
        return None

    def run(self):
        for t, kind, pid in self.events:
            self.clock = max(self.clock, t)
            if kind == "build":
                evicted = self._l1_evict_to_fit(PLEN)
                for ep, etok in evicted:
                    med, dur = self._host_insert(ep, etok)
                    self.backup_p95s.append(dur)
                    self.clock = max(self.clock, t + dur)
                self.l1[pid] = PLEN
                self.n_recompute += 1
                self.ttfts.append(PLEN * RECOMPUTE_US_PER_TK * 1e-6)
                self.clock += PLEN * RECOMPUTE_US_PER_TK * 1e-6
            else:  # reuse
                if pid in self.l1:
                    self.n_l1_hit += 1
                    self.ttfts.append(0.0)
                    self.l1.move_to_end(pid)
                    continue
                res = self._restore(pid)
                if res is None:
                    # recompute
                    evicted = self._l1_evict_to_fit(PLEN)
                    for ep, etok in evicted:
                        med, dur = self._host_insert(ep, etok)
                        self.backup_p95s.append(dur)
                    self.n_recompute += 1
                    rt = PLEN * RECOMPUTE_US_PER_TK * 1e-6
                    self.ttfts.append(rt)
                    self.clock += rt
                    self.l1[pid] = PLEN
                else:
                    med, dur, tokens = res
                    if med == "hbf":
                        self.n_hbf_hit += 1
                    else:
                        self.n_dram_hit += 1
                    evicted = self._l1_evict_to_fit(tokens)
                    for ep, etok in evicted:
                        med2, dur2 = self._host_insert(ep, etok)
                        self.backup_p95s.append(dur2)
                    self.ttfts.append(dur)
                    self.clock = max(self.clock, t + dur)
                    self.l1[pid] = tokens
        return self._summary()

    def _summary(self):
        total = self.n_l1_hit + self.n_hbf_hit + self.n_dram_hit + self.n_recompute
        tt = sorted(self.ttfts)
        bw = sorted(self.backup_p95s) if self.backup_p95s else [0]

        def p95(vals):
            return vals[min(len(vals) - 1, int(0.95 * len(vals)))]

        ws_tokens = self.n_sessions * PLEN
        return {
            "mode": self.mode,
            "n_sessions": self.n_sessions,
            "working_set_gb": round(ws_tokens * KV / GB, 1),
            "host_total_gb": round(self.host_tokens * KV / GB, 1),
            "hbf_gb": round(self.hbf_capacity * KV / GB, 1),
            "dram_gb": round(self.dram_capacity * KV / GB, 1),
            "n_requests": total,
            "hit_l1": self.n_l1_hit, "hit_hbf": self.n_hbf_hit,
            "hit_dram": self.n_dram_hit, "recompute": self.n_recompute,
            "hit_rate_pct": round(
                (self.n_hbf_hit + self.n_dram_hit + self.n_l1_hit)
                / max(1, total) * 100, 1),
            "ttft_p50_ms": round(p95(tt) if tt else 0 * 1000, 2) * 0 + round(
                tt[len(tt) // 2] * 1000, 2) if tt else 0,
            "ttft_p95_ms": round(p95(tt) * 1000, 2),
            "backup_p95_ms": round(p95(bw) * 1000, 2),
            "hbf_read_gb": round(self.hbf_read_bytes / GB, 1),
            "hbf_write_gb": round(self.hbf_write_bytes / GB, 1),
            "dram_read_gb": round(self.dram_read_bytes / GB, 1),
            "dram_write_gb": round(self.dram_write_bytes / GB, 1),
            "per_win": self.win_pool.per_win_summary(),
        }


def main():
    print("=" * 100)
    print("Phase 5D P4-B: 800GB-scale capacity trace (virtual objects, "
          "340tk sessions, agent-style reuse ×2)")
    print("Equal-capacity A/B keep total Host bytes IDENTICAL; expansion C "
          "reported separately")
    print("=" * 100)
    rows = []

    def sweep(sessions, alpha, tag):
        ws_gb = sessions * PLEN * KV / GB
        host_tokens = int(sessions * PLEN * alpha)
        for mode in (("A", "B") if alpha else ("A", "B")):
            sim = SessionTrace(sessions, host_tokens, mode)
            r = sim.run()
            r["alpha"] = alpha
            r["tag"] = tag
            rows.append(r)
            print(f"[{tag}] ws={ws_gb:6.0f}GB α={alpha:.1f} mode={mode} "
                  f"host={r['host_total_gb']:6.0f}GB "
                  f"(HBF {r['hbf_gb']:5.0f} + DRAM {r['dram_gb']:5.0f}) "
                  f"hit={r['hit_rate_pct']:5.1f}% "
                  f"(L1 {r['hit_l1']}/HBF {r['hit_hbf']}/DRAM {r['hit_dram']}"
                  f"/RC {r['recompute']}) "
                  f"TTFT p50/p95={r['ttft_p50_ms']:.2f}/"
                  f"{r['ttft_p95_ms']:.2f}ms "
                  f"backupP95={r['backup_p95_ms']:.2f}ms")
        return rows

    # equal-capacity sweep: working set 0.5× / 1× / 2× / 800GB-scale hosts
    for sessions, alpha in ((2000, 0.5), (2000, 1.0), (2000, 2.0),
                            (8000, 0.5), (8000, 1.0), (8000, 2.0)):
        sweep(sessions, alpha, f"S{sessions}")
    # 800GB scale: working set = 800GB (20,517 sessions), host = 1× and 2×
    sweep(20517, 1.0, "S20517-800GB")
    # expansion (separately reported): working set 1.17TB (30K sessions)
    # EXCEEDS the DRAM-only 800GB pool → capacity benefit materializes
    sim = SessionTrace(30000, int(800 * GB / KV), "A")
    r = sim.run()
    r["alpha"] = "A-800GB-pool"
    r["tag"] = "expansion-baseline"
    rows.append(r)
    print(f"[expansion-baseline] ws=1170GB mode=A DRAM=800GB "
          f"hit={r['hit_rate_pct']:.1f}% "
          f"TTFT p50/p95={r['ttft_p50_ms']:.2f}/{r['ttft_p95_ms']:.2f}ms")
    sim = SessionTrace(30000, int(800 * GB / KV), "C")
    r = sim.run()
    r["alpha"] = "C"
    r["tag"] = "expansion"
    rows.append(r)
    print(f"[expansion] ws=1170GB mode=C DRAM=800GB+HBF-extra=800GB "
          f"(1.6TB host) hit={r['hit_rate_pct']:.1f}% "
          f"TTFT p50/p95={r['ttft_p50_ms']:.2f}/{r['ttft_p95_ms']:.2f}ms "
          f"backupP95={r['backup_p95_ms']:.2f}ms")

    # CSV dump
    import csv
    keys = ["tag", "mode", "alpha", "working_set_gb", "host_total_gb",
            "hbf_gb", "dram_gb", "n_requests", "hit_l1", "hit_hbf",
            "hit_dram", "recompute", "hit_rate_pct", "ttft_p50_ms",
            "ttft_p95_ms", "backup_p95_ms", "hbf_read_gb", "hbf_write_gb",
            "dram_read_gb", "dram_write_gb"]
    os.makedirs("/tmp/p5d_results", exist_ok=True)
    with open("/tmp/p5d_results/p4b_capacity_trace.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n→ /tmp/p5d_results/p4b_capacity_trace.csv")


if __name__ == "__main__":
    main()
