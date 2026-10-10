#!/usr/bin/env python3
"""Item list of campaign g (SectorQ and cuPauliProp): everything on 1, 2, 4, 8 and 16 GPUs, one run each.

Writes queue_g/todo_a.jsonl, one JSON object per run:
  id      <tag>.<circuit>.<cutoff>.g<GPUs>[.<variant>]
  nodes   nodes of the run (1, 2 or 4); per: GPUs (eng) or ranks (cupp) on every node
  kind    eng   SectorQ engine
          ref   cuPauliProp on one GPU (the engine's --mode ref)
          cupp  cuPauliProp distributed, one rank per GPU
  ladder  runs of one ladder go from the easiest cutoff down and stop at the first failure (jobs/g_run.py)
  coef    cutoff, for the prediction of the next step of a ladder
  plugin  MPI plugin of cuPauliProp: orig (as shipped) or prof (times every MPI call)
  smi     sampling period of nvidia-smi in ms during the run
Runs of campaign f whose code path did not change (cuPauliProp on the circuits without noise) are not repeated:
their status goes to queue_g/done/<id> with the log of campaign f.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
LAYOUT = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (2, 4), 16: (4, 4)}          # GPUs -> (nodes, per node)
GPUS = (1, 2, 4, 8, 16)
LIMIT = 2400                                                               # seconds, the same for every implementation
CIRC = {
    "ki":     ("--circuit ki_lab --weight 0", 6000000),
    "tfim":   ("--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20", 4000000),
    "noise":  ("--circuit ki_noisy --noise 0.005", 6000000),
    "tfimsq": ("--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10", 4000000),
    "xxz":    ("--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6", 4000000),
    "qaoa":   ("--circuit qaoa --n 100 --layers 4", 4000000),
    "hea":    ("--circuit hea --n 100 --layers 16 --amp 0.4", 4000000),
}
MAIN = ["ki", "tfim", "noise"]
FINE = ["1e-3", "5e-4", "3e-4", "2e-4", "1e-4", "5e-5", "3e-5", "2e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6"]
LADDER = {
    "ki": FINE, "tfim": FINE, "noise": FINE,
    "tfimsq": ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"],
    "xxz":    ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"],
    "qaoa":   ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"],
    "hea":    ["1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7"],
}
# weak scaling, terms per GPU held constant: cutoff = c4 * (GPUs / 4)^-0.52
#   L: 1e-5 on 4 GPUs (about 40 million terms per GPU), S: 3e-5 on 4 GPUs (about 5 million terms per GPU)
WEAK = {"L": {1: "2e-5", 2: "1.4e-5", 4: "1e-5", 8: "7e-6", 16: "4.8e-6"},
        "S": {1: "6.2e-5", 2: "4.3e-5", 4: "3e-5", 8: "2.1e-5", 16: "1.5e-5"}}
items = []


def add(item_id, g, kind, args, ladder=None, coef=None, **extra):
    nodes, per = LAYOUT[g]
    if kind == "ref":
        nodes, per = 1, 1
    item = {"id": item_id, "nodes": nodes, "per": per, "kind": kind, "limit": LIMIT, "args": args, "smi": 1000}
    if ladder:
        item["ladder"] = ladder
    if coef:
        item["coef"] = float(coef)
    if not ladder:
        item["est"] = 600                  # seconds assumed when the run is fitted into what is left of an allocation
    item.update(extra)
    items.append(item)


def eng(tag, c, coef, g, more="", ladder=None, suffix="", target=None, **extra):
    args, t = CIRC[c]
    add(f"{tag}.{c}.{coef}.g{g}{suffix}", g, "eng", f"{args} --coef {coef} --target {target or t} --trace 1 --vram-frac 0.9{more}",
        ladder, coef, **extra)


def cupp(tag, c, coef, g, more="", ladder=None, **extra):
    args, _ = CIRC[c]
    add(f"{tag}.{c}.{coef}.g{g}", g, "ref" if g == 1 else "cupp", f"{args} --coef {coef} --vram-frac 0.9{more}",
        ladder, coef, plugin=extra.pop("plugin", "orig"), **extra)


# 1. sweep over the number of terms: SectorQ, cuPauliProp (one GPU, distributed, deduplication deferred)
for c, ladder in LADDER.items():
    for g in GPUS:
        for coef in ladder:
            eng("eng", c, coef, g, ladder=f"eng.{c}.g{g}")
        for coef in ladder:
            cupp("cupp", c, coef, g, ladder=f"cupp.{c}.g{g}")
        if g > 1:
            for coef in ladder:
                cupp("defer", c, coef, g, " --cupp-tuned 1", ladder=f"defer.{c}.g{g}")
            if c in MAIN:           # deferred, with an extra deduplication every 3 and every 2 operations
                for k in (3, 2):
                    for coef in ladder[ladder.index("5e-5"):]:
                        cupp(f"defer{k}", c, coef, g, f" --cupp-tuned 1 --cupp-dedup-every {k}", ladder=f"defer{k}.{c}.g{g}")
# cuPauliProp as shipped: the original circuit, no Clifford frame
for c in ("ki", "noise"):
    for coef in ("1e-4", "1e-5"):
        for g in GPUS:
            cupp("orig", c, coef, g, " --frame 0")
# 2. weak scaling at two loads per GPU
for load, cut in WEAK.items():
    for c in MAIN:
        for g in GPUS:
            if cut[g] not in LADDER[c]:
                eng(f"weak{load}", c, cut[g], g)
                cupp(f"weak{load}c", c, cut[g], g)
                if g > 1:
                    cupp(f"weak{load}d", c, cut[g], g, " --cupp-tuned 1")
# 3. where the time goes on 2, 4, 8 and 16 GPUs: MPI calls timed by the profiling plugin, GPU utilization every 200 ms
for c in MAIN:
    for coef in ("1e-4", "3e-5"):
        for g in (2, 4, 8, 16):
            eng("bd", c, coef, g, smi=200)
            cupp("bdc", c, coef, g, plugin="prof", smi=200)
            cupp("bdd", c, coef, g, " --cupp-tuned 1", plugin="prof", smi=200)
# 4. SectorQ with less GPU memory (the default is 0.9 of the 40 GB)
for c in MAIN:
    for g in (1, 4):
        for frac in ("0.45", "0.2", "0.1"):
            args, t = CIRC[c]
            add(f"vram.{c}.1e-5.g{g}.f{frac}", g, "eng", f"{args} --coef 1e-5 --target {t} --trace 1 --vram-frac {frac}", coef="1e-5")
# 5. design parameters: desired sector size, sectors per GPU
for c in MAIN:
    t0 = CIRC[c][1]
    for g in (4, 16):
        for t in (t0 // 4, t0 // 2, t0 * 2):
            eng("tgt", c, "1e-5", g, suffix=f".t{t}", target=t)
        for k in (1, 4, 8):
            eng("pg", c, "1e-5", g, f" --per-gpu {k}", suffix=f".p{k}")
# 6. checkpoint and resume (one node and four nodes), exact values of the 30-qubit lattice
for g in (4, 16):
    ck = f"{ROOT}/results/g/ki_g{g}.ckpt"
    eng("ckptw", "ki", "1e-5", g, f" --ckpt-window 12 --ckpt-file {ck} --stop-at-ckpt 1", ladder=f"ckpt.g{g}")
    eng("ckptr", "ki", "1e-5", g, f" --resume {ck}", ladder=f"ckpt.g{g}", target=3000000)
for steps, coefs in ((6, ("1e-5", "3e-6", "1e-6")), (4, ("1e-6", "1e-7"))):
    for coef in coefs:
        add(f"exact.tfim30.s{steps}.{coef}.g4", 4, "eng",
            f"--circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps {steps} --coef {coef} --target 4000000 --trace 1 --vram-frac 0.9", coef=coef)

assert len({i["id"] for i in items}) == len(items)
# small problems first, the largest last (a ladder keeps its own order)
items.sort(key=lambda i: -i.get("coef", 1e-3))
os.makedirs(os.path.join(Q, "done"), exist_ok=True)
os.makedirs(os.path.join(Q, "lock"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs", "g"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results", "g"), exist_ok=True)
with open(os.path.join(Q, "todo_a.jsonl"), "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")

# runs of campaign f that stand: cuPauliProp on the circuits without noise (same library calls, same operations)
reused = 0
for i in items:
    tag, c = i["id"].split(".")[:2]
    log = os.path.join(ROOT, "logs", "f", i["id"] + ".out")
    done = os.path.join(Q, "done", i["id"])
    if tag not in ("cupp", "defer", "defer2", "defer3", "orig") or c == "noise" or not os.path.isfile(log) or os.path.exists(done):
        continue
    text = open(log).read()
    code = re.search(r"^exit=(\d+)", text, re.M)
    wall = re.search(r"^wall=([0-9.]+)", text, re.M)
    exp = re.search(r"^expectation=(\S+)", text, re.M)
    if not code:
        continue
    ok = code.group(1) == "0" and exp is not None
    status = "ok" if ok else ("tl" if code.group(1) == "124" else "oom" if re.search(r"out of memory|capacity", text) or i["kind"] == "ref" else "fail")
    json.dump({"status": status, "exit": int(code.group(1)), "wall": float(wall.group(1)) if wall else None,
               "expectation": float(exp.group(1)) if exp else None, "log": f"logs/f/{i['id']}.out", "reused": "campaign f"},
              open(done, "w"))
    reused += 1
by = {}
for i in items:
    by[(i["kind"], i["nodes"])] = by.get((i["kind"], i["nodes"]), 0) + 1
print(len(items), "items;", ", ".join(f"{k[0]} on {k[1]} node(s): {v}" for k, v in sorted(by.items())), f"; {reused} taken from campaign f")
