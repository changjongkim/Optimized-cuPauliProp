"""Campaign r, trimmed to the runs that the new exchange can change:   env/bin/python3 jobs/r_trim.py [--write]

The exchange between nodes is the only code that differs between bin/pp_engine_i and the engine of campaign g, and the
first 54 runs of campaign r gave the same expectation values and 0.97-1.03 times the run times. A run is therefore
run again only if it uses two or more nodes and, in campaign g, took 30 s or more, ran out of memory or time (or was
skipped after such a step of its ladder), or is a time-breakdown run. Every other open run takes over the record of
campaign g with "reused": "campaign g", so that the ladders of the runs that remain go on."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q, G = os.path.join(ROOT, "queue_r"), os.path.join(ROOT, "queue_g", "done")
LIMITS = ("oom", "tl", "predicted_tl")

items = [json.loads(line) for line in open(os.path.join(Q, "todo_r.jsonl")) if line.strip()]
old = {i["id"]: json.load(open(os.path.join(G, i["id"]))) for i in items if os.path.exists(os.path.join(G, i["id"]))}
hit = {i["ladder"] for i in items if i.get("ladder") and old.get(i["id"], {}).get("status") in LIMITS}     # ladders that ended at a limit
keep, reuse = [], []
for i in items:
    if os.path.exists(os.path.join(Q, "done", i["id"])) or os.path.isdir(os.path.join(Q, "lock", i["id"])):
        continue
    o = old.get(i["id"])
    if o is None:
        keep.append(i)
        continue
    st = o.get("status")
    again = st in LIMITS or (st == "skipped_after_failure" and i.get("ladder") in hit) or (st == "ok" and (o.get("wall") or 0) >= 30)
    if i["nodes"] >= 2 and (again or i["id"].startswith("bd.")):
        keep.append(i)
    else:
        reuse.append((i, o))
print(f"{len(keep)} runs stay open, {len(reuse)} take over the record of campaign g")
if "--write" in sys.argv:
    for i, o in reuse:
        with open(os.path.join(Q, "done", i["id"]), "w") as f:
            json.dump(dict(o, reused="campaign g"), f)
    print("written")
else:
    seconds = sum(old[i["id"]]["wall"] * i["nodes"] for i in keep if old.get(i["id"], {}).get("status") == "ok")
    print(f"node-hours of the runs that stay (finished ones, campaign g): {seconds / 3600:.1f}")
    print(" ".join(i["id"] for i in keep))
