#!/usr/bin/env python3
"""Item list of campaign g, external baselines (kind "ext", run by jobs/g_ext_impl.py).

  python3 jobs/g_ext_items.py test     -> queue_g/todo_0test.jsonl  one small run per launch path, ahead of everything
  python3 jobs/g_ext_items.py          -> queue_g/todo_b.jsonl      the full list

The implementations and their settings are those of campaign f (queue_f/ext_todo.txt): SPD on one GPU; ORQA
(reproduction) with 64 ranks per node and MonoProp with 8 ranks of 8 threads per node on 1, 2 and 4 nodes;
PauliPropagation.jl, ppvm and OBP with 64 threads and Qiskit pauli-prop (one thread) on one node. A CPU
implementation takes the place of 4, 8 and 16 GPUs with 1, 2 and 4 nodes, and of 1 and 2 GPUs with 16 and 32 cores.
Runs of campaign f that ended (success, out of memory) are not repeated: their status goes to queue_g/done/<id>.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
LIMIT = 2400
FINE = ["1e-3", "5e-4", "3e-4", "2e-4", "1e-4", "5e-5", "3e-5", "2e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6"]
LADDER = {
    "ki": FINE, "tfim": FINE, "noise": FINE,
    "tfimsq": ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"],
    "xxz":    ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"],
    "qaoa":   ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"],
    "hea":    ["1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7"],
}
MAIN = ["ki", "tfim", "noise"]
WEAK = {"L": {1: "2e-5", 2: "1.4e-5", 4: "1e-5", 8: "7e-6", 16: "4.8e-6"},
        "S": {1: "6.2e-5", 2: "4.3e-5", 4: "3e-5", 8: "2.1e-5", 16: "1.5e-5"}}
SET = {"spd": ("matched", 1, 8), "orqa": ("matched", 64, 1), "monoprop": ("native", 8, 8), "julia": ("matched", 1, 64),
       "ppvm": ("matched", 1, 64), "qiskit": ("matched", 1, 1), "obp": ("native", 1, 64)}
SINGLE = ("spd", "julia", "ppvm", "qiskit")                                # use one GPU or one node whatever they are given
SMALL = {"julia": "1e-4", "ppvm": "3e-4", "qiskit": "1e-3"}                # a size each of them finishes in about a minute
combos = sorted({(f[2], f[3], int(f[1])) for f in (l.strip().split("|") for l in open(os.path.join(ROOT, "queue_f", "ext_todo.txt")) if l.strip())})
cases_of = {}
for b, c, n in combos:
    cases_of.setdefault(b, set()).add(c)
items = []


def add(item_id, backend, case, coef, nodes, ladder=None, rpn=None, threads=None, **extra):
    policy, r0, t0 = SET[backend]
    item = {"id": item_id, "nodes": nodes, "kind": "ext", "limit": LIMIT, "backend": backend, "case": case, "coef": float(coef),
            "policy": policy, "rpn": rpn or r0, "threads": threads or t0, "predict": True}
    if backend == "obp":
        item["limit"] = 900                    # OBP works inside a time budget (campaign f: 900 s); it reports how far it got
    if ladder:
        item["ladder"] = ladder
    else:
        item["est"] = 900
    item.update(extra)
    items.append(item)


def cores(backend, n):
    """Ranks per node and threads per rank of a CPU implementation on n cores of one node."""
    return {"orqa": (n, 1), "monoprop": (n // 8, 8)}.get(backend, (1, n))


if len(sys.argv) > 1 and sys.argv[1] == "test":
    add("test.spd", "spd", "ki", "1e-3", 1, sample=1)
    add("test.spd.g4", "spd", "ki", "1e-3", 1, gpus=4, sample=1, gates=True)
    add("test.orqa", "orqa", "ki", "1e-3", 1, prof=True, sample=1)
    add("test.orqa.c16", "orqa", "ki", "1e-3", 1, rpn=16)
    add("test.monoprop", "monoprop", "ki", "1e-3", 1, prof=True, sample=1)
    add("test.monoprop.c16", "monoprop", "ki", "1e-3", 1, rpn=2)
    add("test.julia", "julia", "ki", "1e-3", 1, gates=True, sample=1)
    add("test.julia.t16", "julia", "ki", "1e-3", 1, threads=16)
    add("test.ppvm", "ppvm", "ki", "1e-3", 1, gates=True)
    add("test.qiskit", "qiskit", "ki", "1e-3", 1, gates=True)
    add("test.obp", "obp", "ki", "1e-3", 1)
    add("test.orqa.n4", "orqa", "ki", "1e-3", 4, prof=True, sample=1)
    add("test.monoprop.n4", "monoprop", "ki", "1e-3", 4, prof=True, sample=1)
    add("test.julia.n4", "julia", "ki", "1e-3", 4, use=1, sample=1)
    with open(os.path.join(Q, "todo_0test.jsonl"), "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    print(len(items), "test runs")
    sys.exit(0)

# 1. sweep over the number of terms, with the weak-scaling cutoffs added for the implementations that use one GPU or node
extra = sorted({c for load in WEAK.values() for c in load.values()} - set(FINE), key=float, reverse=True)
for b, c, n in combos:
    ladder = list(LADDER[c])
    if b in SINGLE and c in MAIN:
        ladder = sorted(set(ladder) | set(extra), key=float, reverse=True)
    for coef in ladder:
        add(f"{b}.{c}.{coef}.n{n}", b, c, coef, n, ladder=f"x.{b}.{c}.n{n}")
# 2. weak scaling of the distributed CPU implementations: 16 and 32 cores, 1, 2 and 4 nodes
for load, cut in WEAK.items():
    for b in ("monoprop", "orqa"):
        for c in MAIN:
            if c not in cases_of[b]:
                continue
            for g in (1, 2):
                r, t = cores(b, 16 * g)
                add(f"weak{load}.{b}.{c}.{cut[g]}.g{g}", b, c, cut[g], 1, rpn=r, threads=t)
            for g, n in ((4, 1), (8, 2), (16, 4)):
                if cut[g] not in LADDER[c]:
                    add(f"weak{load}.{b}.{c}.{cut[g]}.g{g}", b, c, cut[g], n)
# 3. strong scaling inside a node: 16 and 32 cores (64 cores are the one-node runs of the sweep)
for b, sizes in (("monoprop", ("1e-4", "3e-5")), ("orqa", ("1e-4", "3e-5")), ("julia", ("1e-4",)), ("ppvm", ("3e-4",))):
    for c in MAIN:
        if c not in cases_of[b]:
            continue
        for coef in sizes:
            for n in (16, 32):
                r, t = cores(b, n)
                add(f"thr.{b}.{c}.{coef}.c{n}", b, c, coef, 1, rpn=r, threads=t)
for c in ("ki", "tfim"):                       # Qiskit pauli-prop has no thread parallelism: given 16 and 64 cores
    for n in (16, 64):
        add(f"thr.qiskit.{c}.1e-3.c{n}", "qiskit", c, "1e-3", 1, threads=n, sample=1)
# 4. implementations of one GPU or one node, given more: SPD with 2, 4, 8, 16 GPUs; the CPU ones with 2 and 4 nodes
for c in ("ki", "tfim"):
    for coef in ("1e-4", "3e-5"):
        for g, n in ((2, 1), (4, 1), (8, 2), (16, 4)):
            add(f"giv.spd.{c}.{coef}.g{g}", "spd", c, coef, n, use=1, gpus=min(g, 4), sample=1)
    for b, coef in SMALL.items():
        for n in (2, 4):
            add(f"giv.{b}.{c}.{coef}.n{n}", b, c, coef, n, use=1, sample=1)
# 5. where the time goes: MPI calls of the distributed CPU implementations, per-gate times of the others
for b in ("monoprop", "orqa"):
    for c in MAIN:
        if c not in cases_of[b]:
            continue
        for coef, n in (("1e-4", 1), ("3e-5", 1), ("1e-4", 4), ("3e-5", 4), ("1e-5", 4)):
            add(f"bd.{b}.{c}.{coef}.n{n}", b, c, coef, n, prof=True, sample=1)
for c in ("ki", "tfim"):
    for coef in ("1e-4", "3e-5"):
        add(f"bd.spd.{c}.{coef}.n1", "spd", c, coef, 1, gates=True, sample=0.2)
    for b, coef in SMALL.items():
        add(f"bd.{b}.{c}.{coef}.n1", b, c, coef, 1, gates=True, sample=1)

assert len({i["id"] for i in items}) == len(items)
items.sort(key=lambda i: -i["coef"])
with open(os.path.join(Q, "todo_b.jsonl"), "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
# runs of campaign f that ended
reused = 0
for i in items:
    done = os.path.join(Q, "done", i["id"])
    fdone = os.path.join(ROOT, "queue_f", "ext_done", i["id"])
    if not i.get("ladder") or os.path.exists(done) or not os.path.exists(fdone):
        continue
    status = open(fdone).read().split()[0]
    if status in ("timeout", "skipped_after_failure", "error", "failed", "running"):
        continue
    rec, n = None, 0
    while True:                                 # the last launch record of the run
        p = os.path.join(ROOT, "results", "f_ext", i["id"] + (f".retry{n}" if n else "") + ".launch.json")
        if not os.path.exists(p):
            break
        rec, n = (json.load(open(p)), p), n + 1
    if rec is None:
        continue
    json.dump({"status": "ok" if status == "success" else status, "wall": rec[0].get("solver_seconds"), "expectation": rec[0].get("expectation"),
               "seconds": rec[0].get("cold_end_to_end_seconds"), "log": os.path.relpath(rec[1], ROOT), "reused": "campaign f"}, open(done, "w"))
    reused += 1
by = {}
for i in items:
    by[i["id"].split(".")[0] if i["id"].split(".")[0] in ("weakL", "weakS", "thr", "giv", "bd") else "sweep"] = by.get(
        i["id"].split(".")[0] if i["id"].split(".")[0] in ("weakL", "weakS", "thr", "giv", "bd") else "sweep", 0) + 1
print(len(items), "external runs:", by, f"; {reused} taken from campaign f")
