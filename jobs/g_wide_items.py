#!/usr/bin/env python3
"""Campaign g, added 2026-10-10 (2): more than 256 qubits, and a qubit sweep whose problem grows with the qubits.

  python3 jobs/g_wide_items.py        writes queue_g/todo_f.jsonl and queue_g/hold/todo_f1.jsonl

Why: the qubit sweep of jobs/g_size_items.py (experiment q) stopped for SectorQ at 529 qubits, because the partition
kernels took terms of at most 8 words (256 qubits). bin/pp_engine_w has the kernels for any width; every run of this
list names that binary ("bin"). The observable of q is Z on the centre site, so beyond the light cone of 10 steps the
number of terms no longer grows with the lattice (only the terms get wider). Experiment m therefore takes the total
magnetization, the sum of Z over all sites (--obs-all 1): one term per qubit at the start.

  val    the sum observable against the sum of the single-site runs (3x3 lattice, 3 steps, nothing truncated)
  probe  which numbers of qubits the cuPauliProp library accepts above 1,024 (2,025 was refused)
  qw     experiment q for SectorQ on 529 qubits and more (single-site observable), where q failed
  m      total magnetization of the LxL TFIM, 10 steps, L = 6 ... 64, cutoffs 1e-3 and 1e-4, every implementation

The runs of one node that need the new binary go to queue_g/hold: the one-node runners of the allocation that was
running when this list was made had loaded the runner before it knew "bin". jobs/g_release.sh moves the file into
the queue when those runners have ended. The limit of a run is 20 minutes, one run per setting, ladders stop at the
first failure, as in the rest of the campaign.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
IR = os.path.join(ROOT, "results", "f_ext", "ir")
BIN = "bin/pp_engine_w"
LIMIT = 1200
EXT = {"spd": ("matched", 1, 8), "orqa": ("matched", 64, 1), "monoprop": ("native", 8, 8), "julia": ("matched", 1, 64), "ppvm": ("matched", 1, 64)}
items = []


def add(item_id, nodes, kind, ladder=None, limit=LIMIT, est=300, **more):
    item = dict(id=item_id, nodes=nodes, kind=kind, limit=limit, est=est, **more)
    if ladder:
        item["ladder"] = ladder
    items.append(item)


def sectorq(item_id, g, args, ladder=None, target=4000000, **more):
    add(item_id, max(1, g // 4), "eng", ladder, per=min(g, 4), bin=BIN, args=f"{args} --target {target} --trace 1 --vram-frac 0.9", **more)


def cupp(item_id, g, args, ladder=None, tuned=False, **more):
    if g == 1:
        add(item_id, 1, "ref", ladder, per=1, bin=BIN, args=f"{args} --vram-frac 0.9", **more)
    else:
        add(item_id, max(1, g // 4), "cupp", ladder, per=min(g, 4), bin=BIN, plugin="orig",
            args=f"{args} --vram-frac 0.9" + (" --cupp-tuned 1" if tuned else ""), **more)


def ext(item_id, b, n, case, x, ladder):
    policy, rpn, threads = EXT[b]
    add(item_id, n, "ext", ladder, backend=b, case=case, coef=float(x), policy=policy, rpn=rpn, threads=threads, predict=True)


tfim = lambda lx, ly, steps, x: f"--circuit tfim_sq --lx {lx} --ly {ly} --theta 0.2 --phi -0.3 --steps {steps} --coef {x}"

# 0. the sum observable: 3x3 lattice, 3 steps, cutoff far below every coefficient
v = tfim(3, 3, 3, "1e-15")
sectorq("val.eng.sum", 16, v + " --obs-all 1", limit=300, est=30)
cupp("val.cupp.sum", 16, v + " --obs-all 1", limit=300, est=30)
for q in range(9):
    sectorq(f"val.eng.q{q}", 16, v + f" --obs-qubit {q}", limit=300, est=30)

# 1. the number of qubits the library accepts (4 steps, cutoff 1e-3: seconds)
for lx, ly in ((32, 33), (32, 40), (40, 40), (31, 64), (32, 64), (64, 64)):
    sectorq(f"probe.eng.n{lx * ly}", 16, tfim(lx, ly, 4, "1e-3"), limit=300, est=30)
for lx, ly in ((32, 33), (40, 40), (31, 64)):
    cupp(f"probe.cupp.n{lx * ly}", 16, tfim(lx, ly, 4, "1e-3"), limit=300, est=30)

# 2. experiment q for SectorQ above 256 qubits
WIDE = (23, 32, 45, 64)
for x in ("1e-4", "1e-5"):
    for g in (16, 4, 1):
        for L in WIDE:
            sectorq(f"qw.eng.L{L}.{x}.g{g}", g, tfim(L, L, 10, x), f"qw.eng.{x}.g{g}")

# 3. experiment m: total magnetization over the lattice size
SIZES = (6, 8, 11, 16, 23, 32, 45, 64)
for x in ("1e-3", "1e-4"):
    arg = lambda L: tfim(L, L, 10, x) + " --obs-all 1"
    for L in SIZES:
        sectorq(f"m.eng.L{L}.{x}.g16", 16, arg(L), f"m.eng.{x}.g16")
    for tag, tuned in (("cupp", False), ("defer", True)):
        for L in SIZES:
            cupp(f"m.{tag}.L{L}.{x}.g16", 16, arg(L), f"m.{tag}.{x}.g16", tuned)
    for b in ("monoprop", "orqa"):
        for L in SIZES[:6]:
            ext(f"m.{b}.L{L}.{x}.n4", b, 4, f"tfimL{L}m", x, f"x.m.{b}.{x}.n4")
    for g in (4, 1):
        for L in SIZES:
            sectorq(f"m.eng.L{L}.{x}.g{g}", g, arg(L), f"m.eng.{x}.g{g}")
    for tag, tuned in (("cupp", False), ("defer", True)):
        for L in SIZES:
            cupp(f"m.{tag}.L{L}.{x}.g4", 4, arg(L), f"m.{tag}.{x}.g4", tuned)
    for L in SIZES:
        cupp(f"m.ref.L{L}.{x}.g1", 1, arg(L), f"m.ref.{x}.g1")
    for b in ("spd", "julia", "ppvm"):
        for L in SIZES[:6]:
            ext(f"m.{b}.L{L}.{x}.n1", b, 1, f"tfimL{L}m", x, f"x.m.{b}.{x}.n1")

assert len({i["id"] for i in items}) == len(items)
old = set()
for name in os.listdir(Q):
    if name.startswith("todo") and name.endswith(".jsonl") and name != "todo_f.jsonl":
        old |= {json.loads(line)["id"] for line in open(os.path.join(Q, name)) if line.strip()}
assert not old & {i["id"] for i in items}

# operation lists of experiment m for the CPU implementations and SPD: the lists of q with one Z per qubit as observable
for L in SIZES[:6]:
    dst = os.path.join(IR, f"tfimL{L}m.ir.json")
    if os.path.exists(dst):
        continue
    ir = json.load(open(os.path.join(IR, "tfimsq.ir.json" if L == 11 else f"tfimL{L}.ir.json")))
    n = ir["qubits"]
    assert n == L * L and len(ir["processing_order"]) == 10 * (n + 2 * L * (L - 1)), (L, n, len(ir["processing_order"]))
    ir["observable"] = [dict(pauli="I" * q + "Z" + "I" * (n - q - 1), coefficient=1) for q in range(n)]
    json.dump(ir, open(dst + ".tmp", "w"))
    os.replace(dst + ".tmp", dst)

# one-node runs with the new binary wait in queue_g/hold (see the head of the file)
hold = [i for i in items if i["nodes"] == 1 and i["kind"] != "ext"]
now = [i for i in items if not (i["nodes"] == 1 and i["kind"] != "ext")]
os.makedirs(os.path.join(Q, "hold"), exist_ok=True)
released = os.path.exists(os.path.join(Q, "todo_f1.jsonl"))
for name, part in (("todo_f.jsonl", now), ("todo_f1.jsonl" if released else os.path.join("hold", "todo_f1.jsonl"), hold)):
    tmp = os.path.join(Q, ".todo_f.tmp")
    with open(tmp, "w") as f:
        for i in part:
            f.write(json.dumps(i) + "\n")
    os.replace(tmp, os.path.join(Q, name))
by = {}
for i in items:
    key = (i["id"].split(".")[0], i["kind"], i["nodes"])
    by[key] = by.get(key, 0) + 1
print(f"{len(now)} runs in queue_g/todo_f.jsonl, {len(hold)} in {'queue_g' if released else 'queue_g/hold'}/todo_f1.jsonl; by (experiment, kind, nodes): {sorted(by.items())}")
