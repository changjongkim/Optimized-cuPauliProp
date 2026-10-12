"""Campaign r: every SectorQ run of campaign g (1-16 GPUs) again with bin/pp_engine_i, the engine whose exchange
between nodes runs in rounds.   env/bin/python3 jobs/r_items.py   writes queue_r/todo_r.jsonl

The runs keep their ids, limits and ladders, so a result of campaign r replaces the result of campaign g with the
same id. Only the binary changes (all runs, also those on one GPU and one node, so that every SectorQ number comes
from one binary) and the checkpoint files go to results/r. The baselines are not run again: their code did not change."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = "bin/pp_engine_i"

items, seen = [], set()
for path in sorted(glob.glob(os.path.join(ROOT, "queue_g", "todo*.jsonl"))):
    for line in open(path):
        if not line.strip():
            continue
        i = json.loads(line)
        if i["kind"] != "eng" or i["id"] in seen:
            continue
        seen.add(i["id"])
        i["bin"] = BIN
        i["args"] = i["args"].replace("/results/g/", "/results/r/")
        items.append(i)
out = os.path.join(ROOT, "queue_r", "todo_r.jsonl")
with open(out + ".tmp", "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
os.replace(out + ".tmp", out)
by = {}
for i in items:
    by[i["nodes"] * i["per"]] = by.get(i["nodes"] * i["per"], 0) + 1
print(f"wrote {len(items)} runs to {out}; by GPU count: {dict(sorted(by.items()))}")
