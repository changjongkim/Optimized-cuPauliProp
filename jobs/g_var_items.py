#!/usr/bin/env python3
"""Campaign g, planned 2026-10-11: run-to-run variability of SectorQ and cuPauliProp's distributed mode (1-16 GPUs).

  python3 jobs/g_var_items.py             writes the plan to queue_g/hold/todo_v.jsonl (no runner reads that directory)
  python3 jobs/g_var_items.py --release   moves the plan to queue_g/todo_v.jsonl, where the runners take it

Every other measurement of the campaign is one run per setting. These runs repeat a few settings ten times, in the
form of the variability section of the ScaleQsim paper (ten runs per problem size on 16 GPUs, ten runs per GPU count
at one problem size, against the main baseline):

  A  kicked Ising on 16 GPUs, five problem sizes (cutoff 1e-4 ... 1e-5): SectorQ and cuPauliProp (default mode)
  B  kicked Ising at cutoff 3e-5: SectorQ on 1, 2, 4, 8 GPUs and cuPauliProp on 4, 8 GPUs (16 GPUs is part of A;
     cuPauliProp runs out of memory on 1 and 2 GPUs at this cutoff)

The runs are listed round by round (every setting once per round), so that the two implementations alternate on the
same nodes and a drift of the nodes meets both alike. Settings and binaries are those of the sweep (jobs/g_items.py).
160 runs, about one hour on one four-node allocation.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
HOLD = os.path.join(Q, "hold", "todo_v.jsonl")
LIVE = os.path.join(Q, "todo_v.jsonl")
REPS = 10
SIZES = ("1e-4", "5e-5", "3e-5", "2e-5", "1e-5")          # all of them finish on 16 GPUs with both implementations
AT = "3e-5"                                                # the problem of the GPU-count part
GPUS = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (2, 4), 16: (4, 4)}   # GPU count -> (nodes, GPUs per node)
ENG = "--circuit ki_lab --weight 0 --coef {x} --target 6000000 --trace 1 --vram-frac 0.9"
CUPP = "--circuit ki_lab --weight 0 --coef {x} --vram-frac 0.9"


def item(kind, x, g, r):
    nodes, per = GPUS[g]
    d = dict(id=f"var.{kind}.ki.{x}.g{g}.r{r:02d}", nodes=nodes, per=per, kind=kind, limit=600, coef=float(x), smi=1000,
             est=60 if kind == "eng" else 120, args=(ENG if kind == "eng" else CUPP).format(x=x), rep=r)
    if kind == "cupp":
        d["plugin"] = "orig"
    return d


def plan():
    out = []
    for r in range(1, REPS + 1):
        for x in SIZES:                                    # A: 16 GPUs
            out += [item("eng", x, 16, r), item("cupp", x, 16, r)]
        for g in (1, 2, 4, 8):                             # B: the same problem on fewer GPUs
            out.append(item("eng", AT, g, r))
            if g >= 4:
                out.append(item("cupp", AT, g, r))
    return out


if __name__ == "__main__":
    if "--release" in sys.argv:
        if not os.path.exists(HOLD):
            sys.exit("no plan in queue_g/hold: run without --release first")
        os.replace(HOLD, LIVE)
        print(f"released: {LIVE}")
    else:
        items = plan()
        os.makedirs(os.path.dirname(HOLD), exist_ok=True)
        tmp = HOLD + ".tmp"
        with open(tmp, "w") as f:
            for i in items:
                f.write(json.dumps(i) + "\n")
        os.replace(tmp, HOLD)
        by = {}
        for i in items:
            by[i["nodes"]] = by.get(i["nodes"], 0) + 1
        print(f"{len(items)} runs planned in queue_g/hold/todo_v.jsonl (by node count: {dict(sorted(by.items()))})")
