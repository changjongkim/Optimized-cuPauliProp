#!/usr/bin/env python3
"""Campaign g, added 2026-10-10: the circuit size (qubits and depth) and the accuracy against the exact value.

  python3 jobs/g_size_items.py        writes queue_g/todo_e.jsonl (run by the loops of campaign g like the other lists)

  acc  5x6 TFIM (30 qubits, exact value known), 4 and 6 steps: every implementation over the cutoffs, so that the time
       to reach an error against the exact value can be compared (MonoProp truncates by its own rule)
  q    qubits: TFIM on an LxL square lattice, 10 steps, L = 6 ... 64 (36 ... 4,096 qubits), cutoff 1e-4 and 1e-5
  d    depth: kicked Ising on the 127-qubit heavy-hex lattice, 4 ... 60 steps, cutoff 3e-5

As in the rest of the campaign every implementation runs in every experiment and a ladder stops at its first failure.
The ladders of q and d go up in size, not down in cutoff, so the runner cannot predict the next step: the limit of a
run is 20 minutes here, the same for all implementations. The runs that the campaign already has (L = 11 is the circuit
tfimsq, 20 steps is the kicked Ising of the sweep) are taken over, not repeated. The CPU implementations and SPD read
the operation lists results/f_ext/ir/<case>.ir.json (exported with --export-ir; up to 1,024 qubits, the list of 2,025
qubits is 260 MB). Step counts are multiples of 4: then the Clifford frame ends as Pauli gates and the list is rotations only.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
LIMIT = 1200
EXT = {"spd": ("matched", 1, 8), "orqa": ("matched", 64, 1), "monoprop": ("native", 8, 8), "julia": ("matched", 1, 64), "ppvm": ("matched", 1, 64)}
items, taken = [], {}


def add(item_id, nodes, kind, ladder, same=None, **more):
    items.append(dict(id=item_id, nodes=nodes, kind=kind, limit=LIMIT, est=300, ladder=ladder, **more))
    if same:
        taken[item_id] = same


def sectorq(item_id, g, args, target, ladder, same=None):
    add(item_id, max(1, g // 4), "eng", ladder, same, per=min(g, 4), args=f"{args} --target {target} --trace 1 --vram-frac 0.9")


def cupp(item_id, g, args, ladder, tuned=False, same=None):
    if g == 1:
        add(item_id, 1, "ref", ladder, same, per=1, args=f"{args} --vram-frac 0.9")
    else:
        add(item_id, max(1, g // 4), "cupp", ladder, same, per=min(g, 4), plugin="orig", args=f"{args} --vram-frac 0.9" + (" --cupp-tuned 1" if tuned else ""))


def ext(item_id, b, n, case, x, ladder, same=None):
    policy, rpn, threads = EXT[b]
    add(item_id, n, "ext", ladder, same, backend=b, case=case, coef=float(x), policy=policy, rpn=rpn, threads=threads, predict=True)


# 1. accuracy against the exact value: 5x6 TFIM, cutoffs from 1e-3 to 1e-7
for s in (4, 6):
    a = f"--circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps {s}"
    cuts = ("1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7", "1e-7")
    for g in (16, 4, 1):
        for x in cuts:
            sectorq(f"acc.eng.s{s}.{x}.g{g}", g, f"{a} --coef {x}", 4000000, f"acc.eng.s{s}.g{g}", same=f"exact.tfim30.s{s}.{x}.g4" if g == 4 else None)
            items[-1]["coef"] = float(x)
    for b, n in (("monoprop", 4), ("monoprop", 1), ("orqa", 4), ("spd", 1), ("julia", 1)):
        for x in cuts:
            ext(f"acc.{b}.s{s}.{x}.n{n}", b, n, f"tfim30s{s}", x, f"x.acc.{b}.s{s}.n{n}")
    for x in cuts:
        cupp(f"acc.ref.s{s}.{x}.g1", 1, f"{a} --coef {x}", f"acc.ref.s{s}")
        items[-1]["coef"] = float(x)

# 2. qubits: TFIM on an LxL lattice, 10 steps
SIZES = (6, 8, 11, 16, 23, 32, 45, 64)
for x in ("1e-4", "1e-5"):
    arg = lambda L: f"--circuit tfim_sq --lx {L} --ly {L} --theta 0.2 --phi -0.3 --steps 10 --coef {x}"
    old = lambda tag, L, suffix: f"{tag}.tfimsq.{x}.{suffix}" if L == 11 else None
    for g in (16, 4, 1):
        for L in SIZES:
            sectorq(f"q.eng.L{L}.{x}.g{g}", g, arg(L), 4000000, f"q.eng.{x}.g{g}", old("eng", L, f"g{g}"))
    for g, tuned in ((16, False), (16, True), (4, False), (4, True), (1, False)):
        tag = "defer" if tuned else ("ref" if g == 1 else "cupp")
        for L in SIZES if g == 1 else SIZES[:-1]:            # distributed: up to 2,025 qubits
            cupp(f"q.{tag}.L{L}.{x}.g{g}", g, arg(L), f"q.{tag}.{x}.g{g}", tuned, old("defer" if tuned else "cupp", L, f"g{g}"))
    which = (("monoprop", 4), ("monoprop", 1), ("orqa", 4), ("orqa", 1), ("spd", 1), ("julia", 1), ("ppvm", 1)) if x == "1e-4" else (("monoprop", 4), ("orqa", 4), ("spd", 1))
    for b, n in which:
        for L in SIZES[:6]:                                  # up to 1,024 qubits
            ext(f"q.{b}.L{L}.{x}.n{n}", b, n, "tfimsq" if L == 11 else f"tfimL{L}", x, f"x.q.{b}.{x}.n{n}", old(b, L, f"n{n}"))

# 3. depth: kicked Ising, cutoff 3e-5
STEPS = (4, 8, 12, 16, 20, 28, 40, 60)
x = "3e-5"
arg = lambda s: f"--circuit ki_lab --weight 0 --steps {s} --coef {x}"
old = lambda tag, s, suffix: f"{tag}.ki.{x}.{suffix}" if s == 20 else None
for g in (16, 4, 1):
    for s in STEPS:
        sectorq(f"d.eng.s{s}.{x}.g{g}", g, arg(s), 6000000, f"d.eng.g{g}", old("eng", s, f"g{g}"))
for g, tuned in ((16, False), (16, True), (4, False), (4, True), (1, False)):
    tag = "defer" if tuned else ("ref" if g == 1 else "cupp")
    for s in STEPS:
        cupp(f"d.{tag}.s{s}.{x}.g{g}", g, arg(s), f"d.{tag}.g{g}", tuned, old("defer" if tuned else "cupp", s, f"g{g}"))
for b, n in (("monoprop", 4), ("monoprop", 1), ("orqa", 4), ("orqa", 1), ("spd", 1), ("julia", 1)):
    for s in STEPS:
        ext(f"d.{b}.s{s}.{x}.n{n}", b, n, f"kis{s}", x, f"x.d.{b}.n{n}", old(b, s, f"n{n}"))

assert len({i["id"] for i in items}) == len(items)
missing = sorted({i["case"] for i in items if i["kind"] == "ext" and not os.path.exists(os.path.join(ROOT, "results", "f_ext", "ir", i["case"] + ".ir.json"))
                  and i["case"] not in ("ki", "tfimsq")})
assert not missing, f"operation lists missing: {missing}"
# runs the campaign already has: the same record under the new name
n_taken = 0
for new, same in taken.items():
    src, dst = os.path.join(Q, "done", same), os.path.join(Q, "done", new)
    if os.path.exists(src) and not os.path.exists(dst):
        record = json.load(open(src))
        record["reused"] = same
        json.dump(record, open(dst, "w"))
        n_taken += 1
tmp = os.path.join(Q, ".todo_e.tmp")
with open(tmp, "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
os.replace(tmp, os.path.join(Q, "todo_e.jsonl"))
by = {}
for i in items:
    key = (i["id"].split(".")[0], i["kind"], i["nodes"])
    by[key] = by.get(key, 0) + (0 if os.path.exists(os.path.join(Q, "done", i["id"])) else 1)
print(f"{len(items)} runs in queue_g/todo_e.jsonl; {n_taken} taken over now; open by (experiment, kind, nodes): {sorted(by.items())}")
