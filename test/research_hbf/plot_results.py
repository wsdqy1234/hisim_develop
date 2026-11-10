"""Export the first-round research figures (simulation, not hardware data)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def figures(supply, cases, output, summary=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_csv(Path(supply) / "supply_scan.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    sizes = [4096, 65536, 262144, 1048576, 4194304, 16777216, 339 * 114688]
    qds = [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    for ax, mode in zip(axes, ("sequential", "random", "write")):
        selected = {(int(r["size_bytes"]), int(r["logical_qd"])): float(r["fixed_pool_efficiency"])
                    for r in data if r["mode"] == mode and r["placement"] == "balanced"
                    and float(r["shared_interface_gbps"]) == 0}
        values = np.array([[selected[(size, qd)] for qd in qds] for size in sizes])
        im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(qds)), qds)
        ax.set_yticks(range(len(sizes)), ["4 KiB", "64 KiB", "256 KiB", "1 MiB", "4 MiB", "16 MiB", "38.9 MB"])
        ax.set_xlabel("Outstanding single-WIN I/Os (QD)")
        ax.set_title(mode.title())
        for y in range(len(sizes)):
            for x in range(len(qds)):
                value = values[y, x]
                ax.text(x, y, f"{value * 100:.1f}" + ("*" if value > .8 + 1e-12 else ""),
                        ha="center", va="center", fontsize=8,
                        color="white" if value > .5 else "black")
    fig.colorbar(im, ax=axes, label="Fixed 8-WIN rated efficiency")
    fig.suptitle("Layer-A service envelope: percentages; * strictly >80%\nUnbounded interface hypothesis; no calibrated WIN-internal QD model", fontsize=12)
    fig.savefig(output / "supply_efficiency.png", dpi=180)
    plt.close(fig)

    interfaces = [0, 112, 896, 1280, 1600]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    x = np.arange(len(interfaces))
    for index, (mode, color) in enumerate(zip(("sequential", "random", "write"), ("#166a8f", "#2d9477", "#d18c2f"))):
        vals = []
        for interface in interfaces:
            row = next(r for r in data if r["mode"] == mode and r["placement"] == "balanced"
                       and int(r["size_bytes"]) == 4194304 and int(r["logical_qd"]) == 16
                       and float(r["shared_interface_gbps"]) == interface)
            vals.append(float(row["fixed_pool_efficiency"]) * 100)
        ax.bar(x + (index - 1) * .23, vals, .23, label=mode, color=color)
    ax.axhline(80, color="#a32630", linestyle="--", label="80% boundary (strict target is above)")
    ax.set_xticks(x, ["Unbounded", "112", "896", "1280", "1600"])
    ax.set_xlabel("Hypothesized package-shared interface (GB/s)")
    ax.set_ylabel("Fixed 8-WIN rated efficiency (%)")
    ax.set_ylim(0, 111)
    ax.set_title("Interface sensitivity: 4 MiB, QD16, balanced\nLayer-A conditional simulation")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.savefig(output / "interface_efficiency.png", dpi=180)
    plt.close(fig)

    summaries = []
    for case in sorted(Path(cases).iterdir()):
        path = case / "demand_analysis/demand_summary.json"
        if path.exists():
            summaries.append((case.name, json.loads(path.read_text())))
    if summaries:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
        seen_workloads = set()
        for name, result in summaries:
            logical = result["logical_demand"]
            if not logical or not logical["unique_reusable_prefix_bytes"]:
                continue
            manifest_path = Path(cases) / name / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            fingerprint = manifest.get("workload_content_sha256")
            if fingerprint in seen_workloads:
                continue
            seen_workloads.add(fingerprint)
            union = logical["unique_reusable_prefix_bytes"]
            curve = [r for r in logical["capacity_curve"] if 0 < r["capacity_bytes"] <= union]
            spec = manifest["workload_spec"]
            label = f"{spec['prefix_tokens']} tokens; churn {spec['churn']}/wave"
            ax.plot([r["capacity_bytes"] / 1e9 for r in curve],
                    [100 * (r["repeat_reference_byte_hit_ratio"] or 0) for r in curve],
                    marker="o", label=label)
        ax.set_xlabel("Offline byte-LRU capacity (decimal GB)")
        ax.set_ylabel("Repeated-reference byte hit ratio (%)")
        ax.set_title("Counterfactual prefix-reference capacity curves\nNot the observed SGLang Host admission policy")
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(output / "logical_capacity_curves.png", dpi=180)
        plt.close(fig)
    if summary:
        matrix = load_csv(Path(summary) / "case_matrix.csv")
        churn = sorted([r for r in matrix if int(r["churn_per_wave"]) > 0],
                       key=lambda r: float(r["configured_host_gb"]))
        if churn:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
            capacities = [float(r["configured_host_gb"]) for r in churn]
            axes[0].plot(capacities, [float(r["reuse_prefix_hit_ratio_all_tiers"]) * 100 for r in churn],
                         marker="o", color="#166a8f")
            axes[0].set_ylabel("Reuse-phase prefix hit ratio, all tiers (%)")
            axes[0].set_ylim(0, 105)
            axes[1].plot(capacities, [float(r["hbf_restore_bytes"]) / 1e6 for r in churn],
                         marker="o", label="HBF -> HBM")
            axes[1].plot(capacities, [float(r["dram_restore_bytes"]) / 1e6 for r in churn],
                         marker="o", label="DRAM -> HBM")
            axes[1].set_ylabel("Observed restore payload (MB)")
            axes[1].legend()
            for ax in axes:
                ax.set_xlabel("Configured total Host capacity (GB)")
                ax.grid(alpha=.2)
            fig.suptitle("Actual fixed-workload Host-capacity sensitivity\nFixed L1, placement policy, L3 and arrival sequence; CPU Mock simulation")
            fig.savefig(output / "actual_host_capacity.png", dpi=180)
            plt.close(fig)

    replay_paths = sorted(Path(cases).glob("*/interface_replay/interface_replay.csv"))
    if replay_paths:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
        for path in replay_paths:
            rows = load_csv(path)
            ax.plot(range(len(rows)), [float(r["restore_duration_max_ms"]) for r in rows],
                    marker="o", label=path.parents[1].name)
        ax.set_xticks(range(5), ["Unbounded", "112", "896", "1280", "1600"])
        ax.set_xlabel("Hypothesized shared interface (GB/s)")
        ax.set_ylabel("Maximum logical restore duration (ms)")
        ax.set_yscale("log")
        ax.set_title("Fixed-arrival Host I/O replay\nIncludes backup dependency; excludes scheduler/free-event feedback")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(output / "restore_interface_replay.png", dpi=180)
        plt.close(fig)
    placement_paths = sorted(Path(cases).glob("*/placement_replay/placement_replay.csv"))
    if placement_paths:
        path = placement_paths[0]
        rows = load_csv(path)
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        values = [float(row["per_io_fixed_seq_efficiency"]) * 100 for row in rows]
        ax.bar(range(len(rows)), values, color=["#2d9477" if v > 80 else "#166a8f" for v in values])
        ax.axhline(80, color="#a32630", linestyle="--")
        ax.set_xticks(range(len(rows)), [row["stripe_block_tokens"] for row in rows])
        ax.set_ylim(0, 112)
        for i, (row, value) in enumerate(zip(rows, values)):
            ax.text(i, value + 2, f"{value:.1f}%\n{row['wins_touched']} WINs", ha="center", fontsize=9)
        ax.set_xlabel("Stripe block size (tokens)")
        ax.set_ylabel("Per-I/O effective BW / 1600 GB/s (%)")
        ax.set_title("340-token fixed-arrival mapping sensitivity\nLayer-A single-restore result; not sustained or hardware efficiency")
        fig.savefig(output / "short_prefix_stripe_sensitivity.png", dpi=180)
        plt.close(fig)
    print(json.dumps({"output": str(output), "figures": [p.name for p in output.glob("*.png")]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supply", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    figures(args.supply, args.cases, args.output, args.summary)
