"""Campaign o: cuPauliProp on the original circuits (1-16 GPUs).   env/bin/python3 jobs/o_items.py   writes queue_o/todo_o.jsonl

Campaign g gave cuPauliProp the list of operations that SectorQ's Clifford frame leaves (the Clifford gates are folded
into the rotations that follow them), which is not something cuPauliProp does. Here every run of cuPauliProp's
single-GPU mode and of its default distributed mode gets the circuit as it is (--frame 0): the library applies the
Clifford gates to the terms itself. Only the four circuits with Clifford gates are run (kicked Ising, noisy kicked
Ising, QAOA, the hardware-efficient ansatz); on TFIM heavy-hex, TFIM 11x11 and XXZ the two inputs are the same.
The runs keep their ids, limits and ladders, so a result of campaign o replaces the result of campaign g with the
same id. The runs with deferred dedup are not repeated: that mode is put together by this work, with its input."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIFFORD = ("--circuit ki_lab", "--circuit ki_noisy", "--circuit qaoa", "--circuit hea")

items, seen = [], set()
for path in sorted(glob.glob(os.path.join(ROOT, "queue_g", "todo*.jsonl"))):
    for line in open(path):
        if not line.strip():
            continue
        i = json.loads(line)
        a = i.get("args", "")
        if i["kind"] not in ("ref", "cupp") or i["id"] in seen or "--cupp-tuned" in a or "--frame" in a:
            continue
        if not any(c in a for c in CLIFFORD):
            continue
        seen.add(i["id"])
        i["args"] = a + " --frame 0"
        items.append(i)
out = os.path.join(ROOT, "queue_o", "todo_o.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out + ".tmp", "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
os.replace(out + ".tmp", out)
by = {}
for i in items:
    by[i["nodes"] * i["per"]] = by.get(i["nodes"] * i["per"], 0) + 1
print(f"wrote {len(items)} runs to {out}; by GPU count: {dict(sorted(by.items()))}")
