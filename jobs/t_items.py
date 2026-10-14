#!/usr/bin/env python3
"""Baselines that the scale campaign left out (32-512 GPUs = 8-128 nodes): queue_s/todo_t_fill.jsonl.

  env/bin/python3 jobs/t_items.py            # print the runs and the minutes they are expected to take, by node count
  env/bin/python3 jobs/t_items.py --write    # write the list

jobs/s_items.py planned every baseline on 32-256 GPUs, but the ladders of the CPU implementations (the ORQA
reproduction, MonoProp) stop at cutoff 3e-6 of kicked Ising and have one cutoff (1e-5) of the other two circuits, and
512 GPUs had SectorQ with only kicked Ising for cuPauliProp. This list fills that in, as campaign g did on 1-16 GPUs:
every implementation that runs on several nodes, on every node count, down each ladder until it fails (out of memory,
over the 40 minutes of a run, or predicted from its earlier steps to take more than that).

  1. the CPU implementations on 8-128 nodes: kicked Ising on from 3e-6 (from 1e-4 on 128 nodes), TFIM heavy-hex and
     (the ORQA reproduction only: MonoProp does not take noise) the noisy kicked Ising from 1e-4; weak scaling on 512
     GPUs. Runs that exist keep their ids, so their records count as the first steps of a ladder.
  2. cuPauliProp on 512 GPUs: the default distributed mode on the other two circuits and on the cutoffs above 1e-5 of
     kicked Ising, deduplication deferred on all three, the default mode with the original circuit (--frame 0, ids
     "o.<id>") where the circuit has Clifford gates, and the weak-scaling points.

The name sorts after todo_s.jsonl, so that a ladder continues the steps that are already there in their order.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "queue_s", "todo_t_fill.jsonl")
LIMIT = 2400
ARGS = {"ki": "--circuit ki_lab --weight 0", "tfim": "--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20",
        "noise": "--circuit ki_noisy --noise 0.005"}
CLIFFORD = ("ki", "noise")                         # circuits whose original form differs from the list the frame leaves
WEAK_L = {512: "8e-7"}
WEAK_S = {512: "2.4e-6"}
UPPER = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6")
LOWER = ("2e-6", "1e-6", "7e-7", "5e-7", "3e-7", "2e-7")
items = []


def add(item_id, g, kind, ladder, coef, est, **more):
    item = dict(id=item_id, nodes=g // 4, kind=kind, limit=LIMIT, coef=float(coef), est=est, **more)
    if ladder:
        item["ladder"] = ladder
    items.append(item)


def ext(item_id, b, c, x, g, ladder=None, est=900):
    rpn, threads, policy = {"orqa": (64, 1, "matched"), "monoprop": (8, 8, "native")}[b]
    add(item_id, g, "ext", ladder, x, est, backend=b, case=c, policy=policy, rpn=rpn, threads=threads, predict=True)


def cupp(item_id, c, x, g, ladder=None, tuned=False, original=False, est=300):
    args = f"{ARGS[c]} --coef {x} --vram-frac 0.9" + (" --cupp-tuned 1" if tuned else "") + (" --frame 0" if original else "")
    add(("o." if original else "") + item_id, g, "cupp", ("o." + ladder if original and ladder else ladder), x, est, per=4, plugin="orig",
        args=args)


# 2. cuPauliProp on 512 GPUs (first: short runs that complete the comparison tables)
g = 512
for c in ("tfim", "noise"):
    for x in UPPER + ("2e-6", "1e-6"):
        cupp(f"cupp.{c}.{x}.g{g}", c, x, g, f"cupp.{c}.g{g}")
for x in ("1e-4", "3e-5"):                         # the ladder of kicked Ising on 512 GPUs started at 1e-5
    cupp(f"cupp.ki.{x}.g{g}", "ki", x, g, f"cupp.ki.g{g}u")
for c in ("ki", "tfim", "noise"):
    for x in UPPER + ("2e-6", "1e-6"):
        cupp(f"defer.{c}.{x}.g{g}", c, x, g, f"defer.{c}.g{g}", tuned=True)
for x in ("1e-4", "3e-5"):
    cupp(f"cupp.ki.{x}.g{g}", "ki", x, g, f"cupp.ki.g{g}u", original=True)
for x in UPPER + ("2e-6", "1e-6"):
    cupp(f"cupp.noise.{x}.g{g}", "noise", x, g, f"cupp.noise.g{g}", original=True)
for c in ("ki", "tfim", "noise"):
    cupp(f"weakSc.{c}.{WEAK_S[g]}.g{g}", c, WEAK_S[g], g)
    cupp(f"weakSd.{c}.{WEAK_S[g]}.g{g}", c, WEAK_S[g], g, tuned=True)
    cupp(f"weakLc.{c}.{WEAK_L[g]}.g{g}", c, WEAK_L[g], g)
    if c in CLIFFORD:
        cupp(f"weakSc.{c}.{WEAK_S[g]}.g{g}", c, WEAK_S[g], g, original=True)
        cupp(f"weakLc.{c}.{WEAK_L[g]}.g{g}", c, WEAK_L[g], g, original=True)

# 1. the CPU implementations on every node count
for g in (32, 64, 128, 256, 512):
    n = g // 4
    for b in ("monoprop", "orqa"):
        for x in (UPPER if g == 512 else ()) + LOWER:
            ext(f"{b}.ki.{x}.n{n}", b, "ki", x, g, f"x.{b}.ki.n{n}")
    for b, cs in (("monoprop", ("tfim",)), ("orqa", ("tfim", "noise"))):
        for c in cs:
            for x in UPPER + ("2e-6", "1e-6"):
                ext(f"{b}.{c}.{x}.n{n}", b, c, x, g, f"x.{b}.{c}.n{n}")
    if g == 512:
        for b, cs in (("monoprop", ("ki", "tfim")), ("orqa", ("ki", "tfim", "noise"))):
            for c in cs:
                ext(f"weakS.{b}.{c}.{WEAK_S[g]}.g{g}", b, c, WEAK_S[g], g)
        for b, cs in (("monoprop", ("ki", "tfim")), ("orqa", ("noise",))):
            for c in cs:
                ext(f"weakL.{b}.{c}.{WEAK_L[g]}.g{g}", b, c, WEAK_L[g], g)

if __name__ == "__main__":
    assert len({i["id"] for i in items}) == len(items)
    done = lambda i: os.path.exists(os.path.join(ROOT, "queue_s", "done", i["id"]))
    by = {}
    for i in items:
        k = i["nodes"] * 4
        by.setdefault(k, [0, 0, 0])
        by[k][0] += 1
        if not done(i):
            by[k][1] += 1
            by[k][2] += i["est"]
    for k, (n_all, n_open, est) in sorted(by.items()):
        print(f"{k:4d} GPUs: {n_all} runs, {n_open} open, at most {round(est / 60)} min if every step ran (ladders stop at the first failure)")
    if "--write" in sys.argv:
        with open(OUT + ".tmp", "w") as f:
            for i in items:
                f.write(json.dumps(i) + "\n")
        os.replace(OUT + ".tmp", OUT)
        print("wrote", OUT)
