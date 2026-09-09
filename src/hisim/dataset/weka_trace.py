"""Weka agentic trace loader (SemiAnalysis cc-traces-weka format).

This module is intentionally dependency-free (stdlib only) so that it can be
loaded standalone (e.g. via importlib from the offline cache analyzers) even
when the full `hisim` package (which imports transformers/sglang) is not
installed.

Input format (one JSON object per line, each line is a full session trace):

    {"id": "...", "block_size": 64, "hash_id_scope": "local",
     "requests": [
       {"t": 10.6, "type": "s", "in": 65408, "out": 265,
        "hash_ids": [10, 11, ...], "api_time": 3.4, "think_time": 1.07},
       {"type": "subagent", "agent_id": "...", "t": ...,
        "requests": [ ...inner requests... ]}
     ]}

Key semantics:
- No text/token content; each request carries token counts plus a sequence of
  KV prefix block hashes (`hash_ids`, `block_size` tokens per block, 64 for
  Anthropic). Prefix reuse across rounds is encoded in shared `hash_ids`
  prefixes.
- `in == len(hash_ids) * block_size` (block-aligned).
- Subagent groups run concurrently with the main stream; this version skips
  them (main-stream rounds only).

Scheduling model (static timeline, consumed by hisim's virtual clock):
- Traces are loaded from one or more jsonl files in order, up to `num_traces`.
- At most `max_concurrent_traces` traces are in flight: the j-th initial trace
  starts its first round at t = j * trace_start_interval.
- When a trace ends (last kept round's recorded `t + api_time`), the next
  trace takes over its slot after `trace_start_interval` seconds.
- Within a trace, subsequent rounds keep the recorded relative time gaps
  (the recorded gaps already encode "wait for the reply, then think").
- `max_rounds_per_trace` truncates a trace to its first N rounds.
- `max_round_tokens` ends a trace right before the first round whose
  (in + out) exceeds the limit; the next trace takes over the slot.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence


@dataclass
class WekaRound:
    """One planned (scheduled) main-stream round of a weka trace."""

    global_seq: int  # sequence number in the planned, time-sorted stream
    trace_idx: int  # index of the owning trace (in load order)
    round_idx: int  # round index within the trace (0-based)
    timestamp: float  # planned arrival time (seconds, 0-based)
    api_time: float  # recorded request duration (seconds)
    hash_ids: tuple  # per-trace local block hashes of this round's input
    block_base: int  # global block-id offset assigned to the owning trace
    input_length: int  # tokens = len(hash_ids) * block_size
    output_length: int  # recorded output token count
    # Link to the next round of the same trace (set by the planner). Used to
    # derive this round's output tokens so they match the next round's input.
    next_hash_ids: tuple = ()
    next_block_base: int = 0


@dataclass
class WekaPlan:
    rounds: List[WekaRound]  # all planned rounds, sorted by timestamp
    num_traces: int
    block_size: int
    # per-trace info, indexed by trace_idx
    trace_infos: List[dict] = field(default_factory=list)


def split_filepaths(filepath: str) -> List[str]:
    """Split a ';'-separated (os.pathsep) path string into a list."""
    if not filepath:
        return []
    sep = ";" if os.pathsep == ";" else os.pathsep
    return [p.strip() for p in filepath.split(sep) if p.strip()]


def _load_trace_lines(filepaths: Sequence[str]) -> Iterator[dict]:
    for fp in filepaths:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _extract_main_rounds(trace_obj: dict) -> List[dict]:
    """Main-stream requests only (skip subagent groups), sorted by `t`."""
    reqs = [
        r
        for r in trace_obj.get("requests", [])
        if r.get("type") != "subagent" and r.get("hash_ids")
    ]
    reqs.sort(key=lambda r: r["t"])
    return reqs


def _apply_round_limits(
    reqs: List[dict], max_rounds_per_trace: int, max_round_tokens: int
) -> tuple:
    """Truncate a trace's rounds. Returns (kept_reqs, n_cut_by_tokens)."""
    if max_rounds_per_trace and max_rounds_per_trace > 0:
        reqs = reqs[:max_rounds_per_trace]
    n_cut = 0
    if max_round_tokens and max_round_tokens > 0:
        kept = []
        for r in reqs:
            if r["in"] + r["out"] > max_round_tokens:
                break
            kept.append(r)
        n_cut = len(reqs) - len(kept)
        reqs = kept
    return reqs, n_cut


def plan_weka_traces(
    filepaths: Sequence[str],
    num_traces: int = -1,
    max_concurrent_traces: int = 1,
    trace_start_interval: float = 0.5,
    max_rounds_per_trace: int = -1,
    max_round_tokens: int = -1,
    block_size: int = 64,
) -> WekaPlan:
    """Parse weka traces.jsonl file(s) and build a static schedule.

    See the module docstring for the scheduling model.
    """
    traces = []
    for obj in _load_trace_lines(filepaths):
        reqs = _extract_main_rounds(obj)
        reqs, n_cut = _apply_round_limits(
            reqs, max_rounds_per_trace, max_round_tokens
        )
        if not reqs:
            continue
        bs = obj.get("block_size") or block_size
        max_hash = 0
        for r in reqs:
            if r["hash_ids"]:
                max_hash = max(max_hash, max(r["hash_ids"]))
        traces.append(
            {
                "id": obj.get("id"),
                "reqs": reqs,
                "block_size": bs,
                "max_hash": max_hash,
                "n_cut_by_tokens": n_cut,
            }
        )
        if 0 < num_traces <= len(traces):
            break

    if not traces:
        raise ValueError(f"No usable weka traces found in: {list(filepaths)}")

    # Global block-id space: each trace owns [base, base + max_hash].
    base = 0
    for t in traces:
        t["block_base"] = base
        base += t["max_hash"] + 1

    # Slot scheduling.
    n_slots = max(1, max_concurrent_traces)
    slots = [j * trace_start_interval for j in range(n_slots)]
    rounds: List[WekaRound] = []
    trace_infos = []
    for ti, t in enumerate(traces):
        slot = min(range(n_slots), key=lambda k: slots[k])
        start = slots[slot]
        reqs = t["reqs"]
        t0 = reqs[0]["t"]
        last = reqs[-1]
        duration = (last["t"] - t0) + last.get("api_time", 0.0)
        slots[slot] = start + duration + trace_start_interval
        t["start"] = start
        t["duration"] = duration
        trace_infos.append(
            {
                "trace_idx": ti,
                "trace_id": t["id"],
                "num_rounds": len(reqs),
                "n_cut_by_tokens": t["n_cut_by_tokens"],
                "start": start,
                "duration": duration,
                "slot": slot,
            }
        )
        for ri, r in enumerate(reqs):
            rounds.append(
                WekaRound(
                    global_seq=0,  # assigned below after sorting
                    trace_idx=ti,
                    round_idx=ri,
                    timestamp=start + (r["t"] - t0),
                    api_time=r.get("api_time", 0.0),
                    hash_ids=tuple(r["hash_ids"]),
                    block_base=t["block_base"],
                    input_length=len(r["hash_ids"]) * t["block_size"],
                    output_length=r["out"],
                )
            )

    rounds.sort(key=lambda r: r.timestamp)
    for seq, rd in enumerate(rounds):
        rd.global_seq = seq

    # Link consecutive rounds of the same trace (for output derivation).
    prev: Optional[WekaRound] = None
    for rd in rounds:
        if prev is not None and prev.trace_idx == rd.trace_idx:
            prev.next_hash_ids = rd.hash_ids
            prev.next_block_base = rd.block_base
        prev = rd

    return WekaPlan(
        rounds=rounds,
        num_traces=len(traces),
        block_size=traces[0]["block_size"],
        trace_infos=trace_infos,
    )


def expand_hash_ids(
    hash_ids: Sequence[int], block_base: int, block_size: int = 64
) -> List[int]:
    """Expand block hashes into a synthetic, globally unique token sequence.

    Block `h` of trace with offset `block_base` maps to tokens
    [(block_base + h) * block_size, ... + block_size). Two requests share a
    token prefix iff they share the corresponding hash-id prefix, which is
    exactly the KV prefix-reuse structure of the recorded session.
    """
    tokens: List[int] = []
    for h in hash_ids:
        bid = (block_base + h) * block_size
        tokens.extend(range(bid, bid + block_size))
    return tokens


def derive_output_tokens(rd: WekaRound, block_size: int = 64) -> List[int]:
    """Tokens generated by this round, aligned with the next round's input.

    In a real multi-turn agent session the assistant reply (and the tool
    results appended after it) becomes part of the next round's input, which
    is already encoded in the next round's `hash_ids`. We therefore derive
    this round's output tokens from the next round's input so that prefix
    matching against the radix tree works exactly like the real session.

    Falls back to unique virtual tokens when there is no next round (last
    round of a trace) or when the next round's input is not longer (context
    compaction).
    """
    if rd.next_hash_ids:
        cur_len = rd.input_length
        want = cur_len + rd.output_length
        nxt = expand_hash_ids(rd.next_hash_ids, rd.next_block_base, block_size)
        if len(nxt) > cur_len:
            out = nxt[cur_len : min(want, len(nxt))]
            if out:
                return out
    return [-(rd.global_seq + 1)] * rd.output_length


def iter_round_entries(plan: WekaPlan) -> Iterator[dict]:
    """Stream planned rounds as analyzer-compatible entries (time-sorted).

    Yields dicts with: timestamp, input_ids, output_ids, input_length,
    output_length, trace_idx, round_idx. Tokens are expanded lazily, one
    round at a time, so peak memory stays at a single request's size.
    """
    for rd in plan.rounds:
        yield {
            "timestamp": rd.timestamp,
            "input_ids": expand_hash_ids(rd.hash_ids, rd.block_base, plan.block_size),
            "output_ids": derive_output_tokens(rd, plan.block_size),
            "input_length": rd.input_length,
            "output_length": rd.output_length,
            "trace_idx": rd.trace_idx,
            "round_idx": rd.round_idx,
        }


def add_weka_arguments(parser) -> None:
    """Register the weka-trace argument group on an argparse parser."""
    g = parser.add_argument_group("weka trace arguments")
    g.add_argument(
        "--weka-traces",
        default=None,
        help=(
            "One or more weka traces.jsonl paths separated by ';'. "
            "When given, --input is optional and the weka loader is used."
        ),
    )
    g.add_argument(
        "--num-traces",
        type=int,
        default=-1,
        help="Total number of traces to load (-1 = all).",
    )
    g.add_argument(
        "--max-concurrent-traces",
        type=int,
        default=1,
        help="Max traces in flight; excess traces wait for a free slot.",
    )
    g.add_argument(
        "--trace-start-interval",
        type=float,
        default=0.5,
        help=(
            "Seconds between the first rounds of concurrently started traces, "
            "and between a finished trace and its slot successor (default 0.5)."
        ),
    )
    g.add_argument(
        "--max-rounds-per-trace",
        type=int,
        default=-1,
        help="Max rounds per trace (-1 = unlimited).",
    )
    g.add_argument(
        "--max-round-tokens",
        type=int,
        default=-1,
        help=(
            "Max tokens (in + out) per round; a trace ends right before its "
            "first round exceeding this (-1 = unlimited)."
        ),
    )


def weka_plan_from_args(args) -> WekaPlan:
    """Build a WekaPlan from parsed argparse args (see add_weka_arguments)."""
    return plan_weka_traces(
        split_filepaths(args.weka_traces),
        num_traces=args.num_traces,
        max_concurrent_traces=args.max_concurrent_traces,
        trace_start_interval=args.trace_start_interval,
        max_rounds_per_trace=args.max_rounds_per_trace,
        max_round_tokens=args.max_round_tokens,
    )
