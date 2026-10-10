#!/usr/bin/env python3
"""Campaign g, last part: where the answer converges.   python3 jobs/g_conv_items.py  -> queue_g/later/todo_c.jsonl

The kicked Ising circuit of the sweep (127 qubits, 20 steps, <Z_62>) at eight more rotation angles, from 0.05 pi
to 0.45 pi (the sweep has 0.25 pi). For every angle a ladder over the cutoff with SectorQ on 1 and 16 GPUs and
cuPauliProp on one GPU and on 16 GPUs (distributed, default): the expectation value against the number of terms
shows where it settles, and how far each implementation gets on the same hardware. The limit of a run is 20
minutes here (the same for both implementations); a ladder stops at its first failure as everywhere else.
jobs/g_chain.sh puts the list into the queue when the rest of the campaign has ended.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
THETAS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.35", "0.4", "0.45"]
RUNGS = ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"]
LIMIT = 1200
items = []
for t in THETAS:
    circ = f"--circuit ki_lab --theta {t} --weight 0"
    for coef in RUNGS:
        for g, nodes in ((1, 1), (16, 4)):
            items.append({"id": f"conv.eng.t{t}.{coef}.g{g}", "nodes": nodes, "per": 1 if g == 1 else 4, "kind": "eng", "limit": LIMIT,
                          "args": f"{circ} --coef {coef} --target 6000000 --trace 1 --vram-frac 0.9", "smi": 1000,
                          "ladder": f"conv.eng.t{t}.g{g}", "coef": float(coef)})
        items.append({"id": f"conv.cupp.t{t}.{coef}.g1", "nodes": 1, "per": 1, "kind": "ref", "limit": LIMIT,
                      "args": f"{circ} --coef {coef} --vram-frac 0.9", "smi": 1000, "plugin": "orig",
                      "ladder": f"conv.cupp.t{t}.g1", "coef": float(coef)})
        items.append({"id": f"conv.cupp.t{t}.{coef}.g16", "nodes": 4, "per": 4, "kind": "cupp", "limit": LIMIT,
                      "args": f"{circ} --coef {coef} --vram-frac 0.9", "smi": 1000, "plugin": "orig",
                      "ladder": f"conv.cupp.t{t}.g16", "coef": float(coef)})
items.sort(key=lambda i: -i["coef"])
assert len({i["id"] for i in items}) == len(items)
os.makedirs(os.path.join(Q, "later"), exist_ok=True)
with open(os.path.join(Q, "later", "todo_c.jsonl"), "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
print(len(items), "runs:", len(THETAS), "angles x", len(RUNGS), "cutoffs x (SectorQ on 1 and 16 GPUs, cuPauliProp on 1 and 16 GPUs)")
