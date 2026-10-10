#!/usr/bin/env python3
"""Run list of the scale campaign (32, 64, 128 and 256 GPUs = 8, 16, 32 and 64 nodes; 512 GPUs: scale-out only): queue_s/todo_s.jsonl, in the
item format of campaign g (jobs/g_items.py, jobs/g_ext_items.py). Run by jobs/s_run.py inside one batch job per GPU count.

  python3 jobs/s_items.py          writes the list and takes over the runs that the scaling jobs e15 already made
  python3 jobs/s_items.py --take   only takes over the runs of e15 (the list is left as it is: a runner may be reading it)

The plan follows campaign g: every performance experiment has all the implementations that run on several nodes
(SectorQ, cuPauliProp's distributed mode in both forms, the ORQA reproduction, MonoProp); the cutoff ladders go from
what everything finishes down to where each implementation fails; 40 minutes per run; one run per setting. A run that
jobs e15 already made (logs/pp_e15_g*-*.out) is not repeated: its record goes to queue_s/done with its part of the log
in logs/s. The list is ordered by importance: a batch job works through it from the top.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_s")
LOGS = os.path.join(ROOT, "logs", "s")
LIMIT = 2400
ARGS = {"ki": ("--circuit ki_lab --weight 0", 6000000), "tfim": ("--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20", 4000000),
        "noise": ("--circuit ki_noisy --noise 0.005", 6000000)}
NAME = {"ki_lab": "ki", "tfim_hh": "tfim", "ki_noisy": "noise"}
WEAK_L = {32: "3.4e-6", 64: "2.3e-6", 128: "1.6e-6", 256: "1.15e-6"}       # cutoff = 1e-5 * (GPUs/4)^-0.52
WEAK_S = {32: "1e-5", 64: "7.1e-6", 128: "4.9e-6", 256: "3.5e-6"}           # cutoff = 3e-5 * (GPUs/4)^-0.52
items = []


def add(item_id, g, kind, ladder=None, coef=None, est=600, **more):
    item = dict(id=item_id, nodes=g // 4, kind=kind, limit=LIMIT, coef=float(coef), est=est, **more)
    if ladder:
        item["ladder"] = ladder
    items.append(item)


def eng(item_id, c, x, g, ladder=None, est=600, more=""):
    a, target = ARGS[c]
    add(item_id, g, "eng", ladder, x, est, per=4, args=f"{a} --coef {x} --target {target} --trace 1 --vram-frac 0.9{more}")


def cupp(item_id, c, x, g, ladder=None, tuned=False, est=300):
    add(item_id, g, "cupp", ladder, x, est, per=4, plugin="orig", args=f"{ARGS[c][0]} --coef {x} --vram-frac 0.9" + (" --cupp-tuned 1" if tuned else ""))


def ext(item_id, b, c, x, g, ladder=None, est=900):
    rpn, threads, policy = {"orqa": (64, 1, "matched"), "monoprop": (8, 8, "native")}[b]
    add(item_id, g, "ext", ladder, x, est, backend=b, case=c, policy=policy, rpn=rpn, threads=threads, predict=True)


for g in (32, 64, 128, 256):
    n = g // 4
    # 1. the representative circuit: SectorQ over the cutoffs (strong-scaling panel and scale-out)
    for x in ("1e-4", "5e-5", "3e-5", "2e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6"):
        eng(f"eng.ki.{x}.g{g}", "ki", x, g, f"eng.ki.g{g}")
    # 2. the other two large circuits
    for c in ("tfim", "noise"):
        for x in ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6"):
            eng(f"eng.{c}.{x}.g{g}", c, x, g, f"eng.{c}.g{g}")
    # 3. weak scaling: the large load (SectorQ: made by e15) and the small load
    for c in ARGS:
        eng(f"weakL.{c}.{WEAK_L[g]}.g{g}", c, WEAK_L[g], g)
        if WEAK_S[g] != "1e-5":                     # on 32 GPUs the small load is the cutoff 1e-5 of the sweep
            eng(f"weakS.{c}.{WEAK_S[g]}.g{g}", c, WEAK_S[g], g, est=120)
    # 3b. up to 1,024 sectors per window for the largest problems: with the default of 256 there are two sectors or
    #     fewer per GPU from 128 GPUs on, and the sectors of these problems do not fit the GPU buffer
    if g >= 128:
        for c, x in (("ki", WEAK_L[g]), ("ki", "3e-6"), ("ki", "1e-6"), ("tfim", WEAK_L[g])):
            eng(f"eng10.{c}.{x}.g{g}", c, x, g, more=" --gmax 10")
    # 4. cuPauliProp's distributed mode on the same GPUs: default and deduplication deferred to the truncations
    for c in ARGS:
        for tag, tuned in (("cupp", False), ("defer", True)):
            for x in ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6"):
                cupp(f"{tag}.{c}.{x}.g{g}", c, x, g, f"{tag}.{c}.g{g}", tuned)
        if WEAK_S[g] != "1e-5":
            cupp(f"weakSc.{c}.{WEAK_S[g]}.g{g}", c, WEAK_S[g], g)
            cupp(f"weakSd.{c}.{WEAK_S[g]}.g{g}", c, WEAK_S[g], g, tuned=True)
        cupp(f"weakLc.{c}.{WEAK_L[g]}.g{g}", c, WEAK_L[g], g)
    # 5. the CPU implementations that run on several nodes, on the same nodes: the representative circuit over the cutoffs
    for b in ("monoprop", "orqa"):
        for x in ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6"):
            ext(f"{b}.ki.{x}.n{n}", b, "ki", x, g, f"x.{b}.ki.n{n}")
    # 6. ... and at the weak-scaling cutoffs (MonoProp does not take the noisy circuit; on 1-16 GPUs the ORQA reproduction
    #    ran out of memory in the large load of the two noiseless circuits)
    for b, cs in (("monoprop", ("tfim",)), ("orqa", ("tfim", "noise"))):
        for c in cs:
            if WEAK_S[g] == "1e-5":
                ext(f"{b}.{c}.1e-5.n{n}", b, c, "1e-5", g)
    for b, cs in (("monoprop", ("ki", "tfim")), ("orqa", ("ki", "tfim", "noise"))):
        for c in cs:
            if WEAK_S[g] != "1e-5":
                ext(f"weakS.{b}.{c}.{WEAK_S[g]}.g{g}", b, c, WEAK_S[g], g)
    for b, cs in (("monoprop", ("ki", "tfim")), ("orqa", ("noise",))):
        for c in cs:
            ext(f"weakL.{b}.{c}.{WEAK_L[g]}.g{g}", b, c, WEAK_L[g], g)
    # 7. reach of SectorQ below the cutoffs of the sweep (last: only when time is left)
    eng(f"eng.ki.5e-7.g{g}", "ki", "5e-7", g, f"eng.ki.g{g}")

# 512 GPUs (128 nodes): the scale-out of SectorQ only, on the representative circuit (no baselines, no weak scaling, so that
# the job is short). The default of 256 sectors per window is below the number of GPUs, so every run sets the limit:
# 1,024, or 4,096 with host chunks of 65,536 terms (a partly filled chunk per sector and node stays small).
for x, est in (("1e-5", 120), ("3e-6", 120), ("1e-6", 300)):
    eng(f"eng10.ki.{x}.g512", "ki", x, 512, est=est, more=" --gmax 10")
for x, est in (("3e-6", 120), ("1e-6", 300), ("5e-7", 900), ("3e-7", 1500)):
    eng(f"eng12.ki.{x}.g512", "ki", x, 512, "eng12.ki.g512", est, " --gmax 12 --chunk 65536")

assert len({i["id"] for i in items}) == len(items)
os.makedirs(os.path.join(Q, "done"), exist_ok=True)
os.makedirs(os.path.join(Q, "lock"), exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
if "--take" not in sys.argv:
    tmp = os.path.join(Q, ".todo_s.tmp")
    with open(tmp, "w") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    os.replace(tmp, os.path.join(Q, "todo_s.jsonl"))

# runs already made by the scaling jobs e15: one record and one log per run
ids = {i["id"] for i in items}
taken = 0
for path in sorted(glob.glob(os.path.join(ROOT, "logs", "pp_e15_g*-*.out"))):
    blocks = re.split(r"(?m)^(?====== (?:engine|cupp) nodes=)", open(path, errors="replace").read())
    for block in blocks:
        m = re.match(r"===== (engine|cupp) nodes=(\d+) gpus=(\d+) args: (.*)", block)
        if not m or "skipped: no time left" in block:
            continue
        kind, g, args = m.group(1), int(m.group(3)), m.group(4)
        opt = dict(zip(args.split()[::2], args.split()[1::2]))
        if "--chunk" in opt or opt.get("--gmax", "10") != "10":
            continue                                  # settings that are not in the list
        c, x = NAME[opt["--circuit"]], opt["--coef"]
        if kind == "engine" and "--gmax" in opt:
            item_id = f"eng10.{c}.{x}.g{g}"
        elif kind == "engine":
            item_id = f"weakL.{c}.{x}.g{g}" if x == WEAK_L.get(g) else f"eng.{c}.{x}.g{g}"
        else:
            item_id = ("defer" if opt.get("--cupp-tuned") == "1" else "cupp") + f".{c}.{x}.g{g}"
        if item_id not in ids or os.path.exists(os.path.join(Q, "done", item_id)):
            continue
        wall = re.search(r"^wall=([0-9.]+)", block, re.M)
        exp = re.search(r"^expectation=(\S+)", block, re.M)
        code = re.search(r"^exit=(\d+)", block, re.M)
        code = int(code.group(1)) if code else None
        if code == 0 and exp:
            status = "ok"
        elif code == 124:
            status = "tl"
        elif re.search(r"out of memory|distributed capacity|OUT_OF_MEMORY|bad_alloc|host allocation failed|Killed|oom-kill|Out Of Memory|FATAL", block):
            status = "oom"
        else:
            status = "fail"
        open(os.path.join(LOGS, item_id + ".out"), "w").write(block)
        json.dump(dict(status=status, exit=code, wall=float(wall.group(1)) if wall else None, expectation=float(exp.group(1)) if exp else None,
                       log=f"logs/s/{item_id}.out", reused=os.path.basename(path)), open(os.path.join(Q, "done", item_id), "w"))
        taken += 1
done = len([i for i in ids if os.path.exists(os.path.join(Q, "done", i))])
by = {}
for i in items:
    if not os.path.exists(os.path.join(Q, "done", i["id"])):
        by[(i["nodes"] * 4, i["kind"])] = by.get((i["nodes"] * 4, i["kind"]), 0) + 1
print(f"{len(items)} runs in queue_s/todo_s.jsonl; {taken} taken over from the e15 logs now, {done} done in all; open by (GPUs, kind): {sorted(by.items())}")
