import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("replay_host_trace", Path(__file__).with_name("replay_host_trace.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_partial_miss_never_counts_unserved_payload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "win_config.json").write_text(json.dumps({"placement": "striped", "stripe_block_tokens": 1,
                                                        "per_win_interface_bw": 1.0}))
    events = []
    for op, arrival, slots in (("backup", 0.0, [0]), ("load", .01, [0, 1])):
        events.append({"pool": "hbf", "path": "real", "op": op, "arrival_s": arrival,
                       "completion_s": arrival + .001, "dur_s": .001,
                       "requested_bytes": len(slots) * 10,
                       "details": {"slots": slots, "requested_bytes": len(slots) * 10,
                                   "object_refs": [{"key": 1, "generation": 0}]}})
    (source / "io_trace.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    m.replay(source, tmp_path / "replay")
    rows = list(csv.DictReader((tmp_path / "replay/interface_replay.csv").open()))
    assert all(int(row["requested_read_bytes"]) == 20 for row in rows)
    assert all(int(row["completed_read_bytes"]) == 10 for row in rows)
    assert all(int(row["restore_miss_ios"]) == 1 for row in rows)
    # interface=0 clears a source per-WIN constraint, so 1-byte/s is not inherited.
    assert float(rows[0]["restore_duration_max_ms"]) < .1
