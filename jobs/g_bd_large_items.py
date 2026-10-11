#!/usr/bin/env python3
"""Campaign g, added 2026-10-10 (3): where the time goes on 16 GPUs for the large setting.

  python3 jobs/g_bd_large_items.py        writes queue_g/todo_h.jsonl

The time-composition runs of the campaign (bd: SectorQ, bdc: cuPauliProp's distributed mode with the profiling plugin,
bdd: the same with the deduplication deferred) were made at cutoff 1e-4 and 3e-5, where SectorQ runs for one to three
seconds. These are the same runs at cutoff 1e-5, the largest setting that cuPauliProp's distributed mode finishes on
16 GPUs: all three circuits, GPU utilization sampled every 0.2 s. The deferred mode runs out of memory on the two
noiseless circuits unless it also deduplicates every three operations (as in the sweep: defer3).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
ARGS = {"ki": ("--circuit ki_lab --weight 0", 6000000, 3), "tfim": ("--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20", 4000000, 3),
        "noise": ("--circuit ki_noisy --noise 0.005", 6000000, 0)}
x = "1e-5"
items = []
for c, (a, target, every) in ARGS.items():
    base = dict(nodes=4, per=4, limit=1200, smi=200, coef=float(x), est=300)
    items.append(dict(id=f"bd.{c}.{x}.g16", kind="eng", args=f"{a} --coef {x} --target {target} --trace 1 --vram-frac 0.9", **base))
    items.append(dict(id=f"bdc.{c}.{x}.g16", kind="cupp", plugin="prof", args=f"{a} --coef {x} --vram-frac 0.9", **base))
    items.append(dict(id=f"bdd.{c}.{x}.g16", kind="cupp", plugin="prof",
                      args=f"{a} --coef {x} --vram-frac 0.9 --cupp-tuned 1" + (f" --cupp-dedup-every {every}" if every else ""), **base))
tmp = os.path.join(Q, ".todo_h.tmp")
with open(tmp, "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
os.replace(tmp, os.path.join(Q, "todo_h.jsonl"))
print(f"{len(items)} runs in queue_g/todo_h.jsonl; open: {sum(1 for i in items if not os.path.exists(os.path.join(Q, 'done', i['id'])))}")
