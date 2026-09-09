#!/usr/bin/env python3
"""HiSim 3-tier cache hit rate theoretical analyzer.

Simulates L1 (block-level prefix cache, vllm-style) / L2 (Host DRAM) / L3 (SSD/kvcm) cache
hierarchy with LRU eviction, using the same cumulative-hash prefix
matching as HiSim's RadixCache.

L1: consecutive prefix match with LRU eviction (capacity-limited)
L2: consecutive prefix match beyond L1 hit (capacity-limited, receives L1 evictions)
L3: hash-based lookup beyond L2 hit (persistent, receives all backup writes)

Usage:
    python hisim_cache_analyzer.py -i trace.jsonl \
        --l1-tokens 200000 --l2-gb 80 --kv-bytes-per-token 262144 --block-size 16
"""

import json
import argparse
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional


class HiSimCacheAnalyzer:
    def __init__(
        self,
        block_size: int = 16,
        kv_bytes_per_token: int = 262144,
        l1_capacity_tokens: int = 0,
        l2_capacity_tokens: int = 0,
        l3_capacity_tokens: int = 0,
    ):
        self.block_size = block_size
        self.kv_bytes_per_token = kv_bytes_per_token
        self.l1_capacity = l1_capacity_tokens  # 0 = unlimited
        self.l2_capacity = l2_capacity_tokens
        self.l3_capacity = l3_capacity_tokens

        # L1: OrderedDict[hash -> (origin_req, last_req)] — LRU ordered
        self.l1_cache: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self.l1_used = 0

        # L2: OrderedDict[hash -> (origin_req, last_req)] — LRU ordered
        self.l2_cache: OrderedDict[str, Tuple[int, int]] = OrderedDict()
        self.l2_used = 0

        # L3: set of hashes — persistent (no eviction unless capacity set)
        self.l3_store: set = set()

        # Stats
        self.total_l1_hits = 0
        self.total_l2_hits = 0
        self.total_l3_hits = 0
        self.total_new = 0
        self.total_l1_evictions = 0
        self.total_l2_evictions = 0

    def compute_block_hashes(self, tokens: List[int]) -> List[str]:
        """Compute cumulative prefix hashes (same semantics as RadixCache).

        Uses a rolling hash: each block's hash incorporates the previous
        block's hash, ensuring that block N's hash depends on tokens[0:N*bs].
        This is O(block_size) per block, not O(total_tokens).
        """
        hashes = []
        num_blocks = len(tokens) // self.block_size
        prev_hash = 0
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = tuple(tokens[start:end])
            prev_hash = hash((prev_hash, block_tokens))
            hashes.append(str(prev_hash))
        return hashes

    def l1_evict(self, blocks_needed: int) -> int:
        """Evict LRU blocks from L1 to make room. Returns number evicted."""
        evicted = 0
        max_blocks = self.l1_capacity // self.block_size if self.l1_capacity > 0 else float('inf')
        while self.l1_used + blocks_needed > max_blocks and self.l1_cache:
            _, (origin_req, last_req) = self.l1_cache.popitem(last=False)
            self.l1_used -= 1
            evicted += 1
            self.total_l1_evictions += 1
            # Evicted from L1 -> goes to L2 and L3
            self.l2_cache[None] = None  # placeholder, will set below
            self.l2_cache.pop(None, None)
            self.l2_cache[str(origin_req) + "_" + str(last_req)] = (origin_req, last_req)
            # Actually we need the block hash, not origin_req. Let me fix this.
        return evicted

    def l1_evict_v2(self, block_hash: str) -> int:
        """Evict one LRU block from L1. The evicted block goes to L2 and L3."""
        if not self.l1_cache:
            return 0
        evicted_hash, (origin_req, last_req) = self.l1_cache.popitem(last=False)
        self.l1_used -= 1
        self.total_l1_evictions += 1
        # Add to L2
        if evicted_hash not in self.l2_cache:
            self._l2_add(evicted_hash, origin_req, last_req)
        # Add to L3 (always)
        self.l3_store.add(evicted_hash)
        return 1

    def _l2_add(self, block_hash: str, origin_req: int, last_req: int) -> bool:
        """Add a block to L2, evicting non-L1-resident LRU if necessary.

        Models HiSim's evict_host which can only evict nodes that are
        evicted from L1 (in evictable_host_leaves). Blocks still in L1
        are 'pinned' in L2 and cannot be evicted.
        Returns False if L2 is full and all remaining blocks are pinned.
        """
        if self.l2_capacity > 0:
            max_blocks = self.l2_capacity // self.block_size
            while len(self.l2_cache) >= max_blocks:
                # Find oldest block NOT in L1 (evictable)
                evicted_hash = None
                for h in self.l2_cache:  # OrderedDict, oldest first
                    if h not in self.l1_cache:
                        evicted_hash = h
                        break
                if evicted_hash is None:
                    # All L2 blocks are pinned (in L1), can't evict
                    return False
                self.l2_cache.pop(evicted_hash)
                self.total_l2_evictions += 1
        self.l2_cache[block_hash] = (origin_req, last_req)
        return True

    def _l1_add(self, block_hash: str, request_id: int):
        """Add a block to L1, evicting LRU if necessary."""
        if self.l1_capacity > 0:
            max_blocks = self.l1_capacity // self.block_size
            while self.l1_used >= max_blocks and self.l1_cache:
                self.l1_evict_v2(block_hash)
        if block_hash in self.l1_cache:
            self.l1_cache[block_hash] = (self.l1_cache[block_hash][0], request_id)
            self.l1_cache.move_to_end(block_hash)
        else:
            self.l1_cache[block_hash] = (request_id, request_id)
            self.l1_used += 1

    def analyze_request(
        self, block_hashes: List[str], request_id: int
    ) -> Dict:
        """Analyze a single request's cache hits across 3 tiers."""
        total_blocks = len(block_hashes)
        if total_blocks == 0:
            return {"l1_hit": 0, "l2_hit": 0, "l3_hit": 0, "new": 0}

        # L1: consecutive prefix match from start
        l1_hit_blocks = 0
        for bh in block_hashes:
            if bh in self.l1_cache:
                l1_hit_blocks += 1
            else:
                break

        # L2: consecutive prefix match beyond L1 hit
        l2_hit_blocks = 0
        for i in range(l1_hit_blocks, total_blocks):
            if block_hashes[i] in self.l2_cache:
                l2_hit_blocks += 1
            else:
                break

        # L3: hash-based lookup beyond L2 hit (not necessarily consecutive)
        l3_hit_blocks = 0
        for i in range(l1_hit_blocks + l2_hit_blocks, total_blocks):
            if block_hashes[i] in self.l3_store:
                l3_hit_blocks += 1

        new_blocks = total_blocks - l1_hit_blocks - l2_hit_blocks - l3_hit_blocks

        # Store all blocks in L1 (with eviction)
        # First, "touch" L1-hit blocks (update LRU)
        for i in range(l1_hit_blocks):
            bh = block_hashes[i]
            origin = self.l1_cache[bh][0]
            self.l1_cache[bh] = (origin, request_id)
            self.l1_cache.move_to_end(bh)

        # Store new + L3-hit blocks in L1 (they're now computed/loaded)
        for i in range(l1_hit_blocks, total_blocks):
            bh = block_hashes[i]
            if bh not in self.l1_cache:
                self._l1_add(bh, request_id)
            else:
                self.l1_cache[bh] = (self.l1_cache[bh][0], request_id)
                self.l1_cache.move_to_end(bh)

        # When request finishes: backup all blocks to L2 and L3
        # (models cache_finished_req -> write_backup -> batch_set)
        # Only blocks in L1 are backed up (write_backup called on matched
        # non-evicted nodes). L2 can fail if full of pinned (L1-resident) blocks.
        for i in range(total_blocks):
            bh = block_hashes[i]
            if bh not in self.l2_cache:
                # Only backup if block is in L1 (matched, non-evicted)
                if bh in self.l1_cache:
                    self._l2_add(bh, request_id, request_id)
            else:
                self.l2_cache[bh] = (self.l2_cache[bh][0], request_id)
                self.l2_cache.move_to_end(bh)
            self.l3_store.add(bh)

        # Update stats
        self.total_l1_hits += l1_hit_blocks
        self.total_l2_hits += l2_hit_blocks
        self.total_l3_hits += l3_hit_blocks
        self.total_new += new_blocks

        return {
            "l1_hit": l1_hit_blocks,
            "l2_hit": l2_hit_blocks,
            "l3_hit": l3_hit_blocks,
            "new": new_blocks,
            "total": total_blocks,
        }

    def get_summary(self, total_input_tokens: int) -> Dict:
        total_blocks = self.total_l1_hits + self.total_l2_hits + self.total_l3_hits + self.total_new
        return {
            "total_blocks": total_blocks,
            "l1_hit_blocks": self.total_l1_hits,
            "l2_hit_blocks": self.total_l2_hits,
            "l3_hit_blocks": self.total_l3_hits,
            "new_blocks": self.total_new,
            "l1_evictions": self.total_l1_evictions,
            "l2_evictions": self.total_l2_evictions,
            "l1_hit_rate": f"{self.total_l1_hits / total_blocks * 100:.2f}%" if total_blocks else "0%",
            "l2_hit_rate": f"{self.total_l2_hits / total_blocks * 100:.2f}%" if total_blocks else "0%",
            "l3_hit_rate": f"{self.total_l3_hits / total_blocks * 100:.2f}%" if total_blocks else "0%",
            "l1_hit_tokens": self.total_l1_hits * self.block_size,
            "l2_hit_tokens": self.total_l2_hits * self.block_size,
            "l3_hit_tokens": self.total_l3_hits * self.block_size,
            "new_tokens": self.total_new * self.block_size,
            "l1_hit_token_rate": f"{self.total_l1_hits * self.block_size / total_input_tokens * 100:.2f}%" if total_input_tokens else "0%",
            "l2_hit_token_rate": f"{self.total_l2_hits * self.block_size / total_input_tokens * 100:.2f}%" if total_input_tokens else "0%",
            "l3_hit_token_rate": f"{self.total_l3_hits * self.block_size / total_input_tokens * 100:.2f}%" if total_input_tokens else "0%",
            "final_l1_blocks": self.l1_used,
            "final_l2_blocks": len(self.l2_cache),
            "final_l3_blocks": len(self.l3_store),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="HiSim 3-tier cache hit rate theoretical analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (Qwen3-32B, L1=200K, L2=80GB, L3=unlimited)
  %(prog)s -i trace.jsonl

  # Custom capacities
  %(prog)s -i trace.jsonl --l1-tokens 200000 --l2-gb 80

  # Unlimited L1 (no eviction)
  %(prog)s -i trace.jsonl --l1-tokens 0
        """
    )
    parser.add_argument("-i", "--input", default=None,
                        help="Input JSONL trace file (hisim_collection format); optional if --weka-traces is given")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL results file")
    parser.add_argument("--block-size", type=int, default=16, help="Block size in tokens (default: 16, kvcm block size)")
    parser.add_argument("--kv-bytes-per-token", type=int, default=262144,
                        help="KV cache bytes per token (default: 262144 = 8*128*64*2*2 for Qwen3-32B FP16)")
    parser.add_argument("--l1-tokens", type=int, default=200000, help="L1 (HBM) capacity in tokens (default: 200000)")
    parser.add_argument("--l2-gb", type=float, default=80, help="L2 (DRAM) capacity in GB (default: 80)")
    parser.add_argument("--l3-gb", type=float, default=0, help="L3 (SSD) capacity in GB (0 = unlimited)")
    parser.add_argument("--use-output-ids", action="store_true",
                        help="Include output_ids in token sequence (default: input_ids only)")
    parser.add_argument("--max-requests", type=int, default=0, help="Max requests to analyze (0 = all)")
    _load_weka_module_for_args(parser)
    args = parser.parse_args()
    return args


def _load_weka_module_for_args(parser):
    """Register weka-trace arguments; returns nothing (args parsed later)."""
    try:
        from hisim.dataset import weka_trace
        weka_trace.add_weka_arguments(parser)
    except Exception:
        import importlib.util
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent
             / "src" / "hisim" / "dataset" / "weka_trace.py")
        spec = importlib.util.spec_from_file_location("hisim_weka_trace", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.add_weka_arguments(parser)


def _load_weka_module():
    """Load the weka_trace loader without requiring the full hisim package."""
    try:
        from hisim.dataset import weka_trace
        return weka_trace
    except Exception:
        import importlib.util
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent
             / "src" / "hisim" / "dataset" / "weka_trace.py")
        spec = importlib.util.spec_from_file_location("hisim_weka_trace", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


def main():
    args = parse_args()

    l2_tokens = int(args.l2_gb * 1e9 / args.kv_bytes_per_token) if args.l2_gb > 0 else 0
    l3_tokens = int(args.l3_gb * 1e9 / args.kv_bytes_per_token) if args.l3_gb > 0 else 0

    print("HiSim 3-Tier Cache Analyzer")
    print(f"  Input: {args.weka_traces if args.weka_traces else args.input}")
    print(f"  Block size: {args.block_size} tokens")
    print(f"  KV bytes/token: {args.kv_bytes_per_token:,} ({args.kv_bytes_per_token / 1024:.0f} KB)")
    print(f"  L1 capacity: {args.l1_tokens:,} tokens ({args.l1_tokens * args.kv_bytes_per_token / 1e9:.1f} GB virtual)" if args.l1_tokens else "  L1 capacity: UNLIMITED")
    print(f"  L2 capacity: {l2_tokens:,} tokens ({args.l2_gb} GB virtual)" if l2_tokens else "  L2 capacity: UNLIMITED")
    print(f"  L3 capacity: {l3_tokens:,} tokens ({args.l3_gb} GB virtual)" if l3_tokens else "  L3 capacity: UNLIMITED")
    print()

    analyzer = HiSimCacheAnalyzer(
        block_size=args.block_size,
        kv_bytes_per_token=args.kv_bytes_per_token,
        l1_capacity_tokens=args.l1_tokens,
        l2_capacity_tokens=l2_tokens,
        l3_capacity_tokens=l3_tokens,
    )

    results = []
    total_input_tokens = 0

    if args.weka_traces:
        weka_mod = _load_weka_module()
        plan = weka_mod.weka_plan_from_args(args)
        print(f"  Weka plan: {plan.num_traces} traces, {len(plan.rounds)} rounds, "
              f"block_size={plan.block_size}")
        entries = weka_mod.iter_round_entries(plan)  # time-sorted generator
        total = len(plan.rounds)
        use_output = True  # derived output_ids must enter the tree for reuse
    else:
        if not args.input:
            raise SystemExit("error: --input is required unless --weka-traces is given")
        with open(args.input, "r") as f:
            entries = [json.loads(l.strip()) for l in f]
        entries.sort(key=lambda d: d.get("timestamp", 0))
        total = len(entries)
        use_output = args.use_output_ids

    max_req = args.max_requests if args.max_requests > 0 else total

    for idx, data in enumerate(entries):
        if idx >= max_req:
            break
        try:
            input_ids = data.get("input_ids", [])
            output_ids = data.get("output_ids", [])
            input_len = data.get("input_length", len(input_ids))

            if use_output:
                tokens = input_ids + output_ids
            else:
                tokens = input_ids

            total_input_tokens += input_len

            block_hashes = analyzer.compute_block_hashes(tokens)
            result = analyzer.analyze_request(block_hashes, idx + 1)
            result["request_id"] = idx + 1
            result["input_length"] = input_len
            results.append(result)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} requests...")

        except json.JSONDecodeError:
            print(f"  Warning: Failed to parse line {idx}")

    # Print summary
    summary = analyzer.get_summary(total_input_tokens)
    print("\n" + "=" * 80)
    print("3-Tier Cache Hit Rate Summary")
    print("=" * 80)
    print(f"  Total requests: {len(results)}")
    print(f"  Total input tokens: {total_input_tokens:,}")
    print(f"  Total blocks: {summary['total_blocks']:,}")
    print()
    print(f"  L1 (Radix Cache):")
    print(f"    Hit blocks: {summary['l1_hit_blocks']:,}  Hit tokens: {summary['l1_hit_tokens']:,}")
    print(f"    Hit rate:   {summary['l1_hit_rate']}  ({summary['l1_hit_token_rate']} of input)")
    print(f"    Evictions: {summary['l1_evictions']:,}")
    print(f"    Final cached: {summary['final_l1_blocks']:,} blocks")
    print()
    print(f"  L2 (Host DRAM):")
    print(f"    Hit blocks: {summary['l2_hit_blocks']:,}  Hit tokens: {summary['l2_hit_tokens']:,}")
    print(f"    Hit rate:   {summary['l2_hit_rate']}  ({summary['l2_hit_token_rate']} of input)")
    print(f"    Evictions: {summary['l2_evictions']:,}")
    print(f"    Final cached: {summary['final_l2_blocks']:,} blocks")
    print()
    print(f"  L3 (SSD/kvcm):")
    print(f"    Hit blocks: {summary['l3_hit_blocks']:,}  Hit tokens: {summary['l3_hit_tokens']:,}")
    print(f"    Hit rate:   {summary['l3_hit_rate']}  ({summary['l3_hit_token_rate']} of input)")
    print(f"    Final stored: {summary['final_l3_blocks']:,} blocks")
    print()
    print(f"  New (need computation):")
    print(f"    New blocks: {summary['new_blocks']:,}  New tokens: {summary['new_tokens']:,}")
    print()
    print(f"  Hierarchy check: L1={summary['final_l1_blocks']} < L2={summary['final_l2_blocks']} < L3={summary['final_l3_blocks']}")
    print("=" * 80)

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
