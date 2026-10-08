#!/usr/bin/env python3
"""Item list of campaign f: everything that runs on 1, 2, 4, 8 and 16 GPUs (up to four nodes), one run each.

One line per run in queue_f/todo.txt:  id|nodes|per_node|kind|timeout_s|program arguments
  kind eng    SectorQ engine, per_node = GPUs used on every node
  kind ref    cuPauliProp on one GPU (the engine's --mode ref: one sector, in memory)
  kind cupp   cuPauliProp distributed, per_node = ranks (one GPU each) on every node
jobs/f_runner.sh executes the lines; a finished run leaves queue_f/done/<id> and logs/f/<id>.out.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (2, 4), 16: (4, 4)}          # GPUs -> (nodes, per node)
CIRC = {
    "ki":     ("--circuit ki_lab --weight 0", 6000000),
    "tfim":   ("--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20", 4000000),
    "noise":  ("--circuit ki_noisy --noise 0.005", 6000000),
    "tfimsq": ("--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10", 4000000),
    "xxz":    ("--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6", 4000000),
    "qaoa":   ("--circuit qaoa --n 100 --layers 4", 4000000),
    "hea":    ("--circuit hea --n 100 --layers 16 --amp 0.4", 4000000),
}
LADDER = {
    "ki":     ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"],
    "tfim":   ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "5e-6"],
    "noise":  ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "5e-6"],
    "tfimsq": ["1e-4", "1e-5", "3e-6"],
    "xxz":    ["1e-4", "1e-5", "1e-6"],
    "qaoa":   ["1e-4", "1e-5", "1e-6"],
    "hea":    ["1e-5", "1e-6", "1e-7"],
}
# weak scaling: cutoff = 1e-5 * (GPUs / 4)^-0.52 (4 GPUs: the 1e-5 run of the ladder)
WEAK = {1: "2e-5", 2: "1.4e-5", 8: "7e-6", 16: "4.8e-6"}
# measured or estimated seconds on 16 GPUs, for the time limits
T16 = {
    "ki":    {"1e-3": 1, "3e-4": 1, "1e-4": 1.2, "3e-5": 4, "2e-5": 7, "1.4e-5": 10, "1e-5": 14.6, "7e-6": 26, "4.8e-6": 52, "3e-6": 133},
    "tfim":  {"1e-3": 1, "3e-4": 1.5, "1e-4": 2.1, "3e-5": 6, "2e-5": 9, "1.4e-5": 13, "1e-5": 17.5, "7e-6": 32, "4.8e-6": 66, "5e-6": 62},
    "noise": {"1e-3": 1, "3e-4": 1.5, "1e-4": 2.3, "3e-5": 4, "2e-5": 6, "1.4e-5": 8, "1e-5": 11.6, "7e-6": 18, "4.8e-6": 36, "5e-6": 34},
}
MAIN = ["ki", "tfim", "noise"]
items = []


def limit(c, coef, g):
    if c not in T16:
        return 400 if g == 1 else 300
    est = max(T16[c][coef] * 16.0 / g * 1.25, 5.0)
    return int(min(3000, 1.6 * est + 90))


def eng(c, coef, g, tag="eng"):
    nodes, per = LAYOUT[g]
    args, target = CIRC[c]
    items.append(f"{tag}.{c}.{coef}.g{g}|{nodes}|{per}|eng|{limit(c, coef, g)}|{args} --coef {coef} --target {target} --trace 1 --vram-frac 0.9")


def cupp(tag, c, coef, g, extra=""):
    args, _ = CIRC[c]
    if g == 1:
        items.append(f"{tag}.{c}.{coef}.g1|1|1|ref|300|{args} --coef {coef} --vram-frac 0.9{extra}")
    else:
        nodes, per = LAYOUT[g]
        items.append(f"{tag}.{c}.{coef}.g{g}|{nodes}|{per}|cupp|300|{args} --coef {coef} --vram-frac 0.9{extra}")


# 1. SectorQ: cutoff ladder of the three large circuits on every GPU count, then the weak-scaling points
for c in MAIN:
    for coef in LADDER[c]:
        for g in (1, 2, 4, 8, 16):
            eng(c, coef, g)
    for g, coef in WEAK.items():
        eng(c, coef, g, tag="weak")
# 2. cuPauliProp on the same operation list: one GPU, distributed default, deduplication deferred to the truncations
for c in MAIN:
    for coef in LADDER[c]:
        for g in (1, 2, 4, 8, 16):
            cupp("cupp", c, coef, g)
        for g in (2, 4, 8, 16):
            cupp("defer", c, coef, g, " --cupp-tuned 1")
    # deferred with an extra deduplication every 3 and every 2 operations, where the plain deferral runs out of memory
    for coef in ("3e-5", "1e-5"):
        for g in (2, 4, 8, 16):
            cupp("defer3", c, coef, g, " --cupp-tuned 1 --cupp-dedup-every 3")
            cupp("defer2", c, coef, g, " --cupp-tuned 1 --cupp-dedup-every 2")
# 3. cuPauliProp as shipped: the original circuit, no Clifford frame
for c in ("ki", "noise"):
    for coef in ("1e-4", "1e-5"):
        for g in (1, 2, 4, 8, 16):
            cupp("orig", c, coef, g, " --frame 0")
# 4. the other four circuits: SectorQ and cuPauliProp
for c in ("tfimsq", "xxz", "qaoa", "hea"):
    for coef in LADDER[c]:
        for g in (1, 2, 4, 8, 16):
            eng(c, coef, g)
    for coef in LADDER[c]:
        for g in (1, 2, 4, 8, 16):
            cupp("cupp", c, coef, g)
        for g in (2, 4, 8, 16):
            cupp("defer", c, coef, g, " --cupp-tuned 1")

assert len({i.split("|")[0] for i in items}) == len(items)
os.makedirs(os.path.join(ROOT, "queue_f"), exist_ok=True)
open(os.path.join(ROOT, "queue_f", "todo.txt"), "w").write("\n".join(items) + "\n")
by = {}
for i in items:
    f = i.split("|")
    by[(f[3], f[1])] = by.get((f[3], f[1]), 0) + 1
print(len(items), "items;", ", ".join(f"{k[0]} on {k[1]} node(s): {v}" for k, v in sorted(by.items())))
