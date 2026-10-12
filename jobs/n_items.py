"""The SectorQ runs of the scale campaign again with bin/pp_engine_i: queue_s/todo_a1_n.jsonl (32-512 GPUs).

  env/bin/python3 jobs/n_items.py            # print what would be written
  env/bin/python3 jobs/n_items.py --write    # write the list; mark the open runs of the old engine as superseded

The list sorts before the other lists of queue_s, so a scale job takes these runs first, the short ones before the
long ones: the cutoff sweeps of the three circuits, weak scaling, the runs with a higher sector limit that the
README compares, and the ladder of kicked Ising down to the memory limit. The four cutoffs 1e-4, 3e-5, 1e-5 and 3e-6
of kicked Ising with the default settings and the last steps of the ladder are in todo_a2_i.jsonl (jobs/i_items.py)
and are not repeated here. On 512 GPUs TFIM, noisy kicked Ising and weak scaling are run for the first time.
The SectorQ runs that the other lists still hold for the old engine (bin/pp_engine_h) are marked superseded."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = "bin/pp_engine_i"
ARGS = {"ki": "--circuit ki_lab --weight 0 --coef {x} --target 6000000 --trace 1 --vram-frac 0.9",
        "tfim": "--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef {x} --target 4000000 --trace 1 --vram-frac 0.9",
        "noise": "--circuit ki_noisy --noise 0.005 --coef {x} --target 6000000 --trace 1 --vram-frac 0.9"}
GPUS = (32, 64, 128, 256, 512)
KI_SWEEP = ("1e-4", "5e-5", "3e-5", "2e-5", "1e-5", "5e-6", "3e-6", "2e-6")
IN_A2 = ("1e-4", "3e-5", "1e-5", "3e-6")                # kicked Ising, default settings, 64-512 GPUs: i.ref.* of todo_a2_i
SWEEP3 = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6")
WEAK_L = {32: "3.4e-6", 64: "2.3e-6", 128: "1.6e-6", 256: "1.15e-6", 512: "8e-7"}       # 1e-5 * (GPUs/4)^-0.52
WEAK_S = {32: "1e-5", 64: "7.1e-6", 128: "4.9e-6", 256: "3.5e-6", 512: "2.4e-6"}        # 3e-5 * (GPUs/4)^-0.52
# the ladder of kicked Ising: (sector limit and chunk size, ((cutoff, seconds expected), ...)); its last steps are in todo_a2_i
LADDER = {32: (" --gmax 10", (("1e-6", 600), ("7e-7", 1600), ("5e-7", 3600), ("3e-7", 7000))),
          64: (" --gmax 10", (("1e-6", 300), ("7e-7", 800), ("5e-7", 1700), ("3e-7", 3600))),
          128: (" --gmax 12 --chunk 65536", (("1e-6", 200), ("7e-7", 400), ("5e-7", 900))),
          256: (" --gmax 12 --chunk 65536", (("1e-6", 100), ("7e-7", 200), ("5e-7", 400), ("3e-7", 1400))),
          512: (" --gmax 12 --chunk 65536", (("3e-6", 20), ("2e-6", 30), ("1e-6", 60), ("7e-7", 110), ("5e-7", 280), ("3e-7", 750), ("2e-7", 1800)))}


def items():
    out = []

    def add(name, c, x, g, extra="", limit=2400, est=120, ladder=None):
        i = dict(id=f"n.{name}.{x}.g{g}", nodes=g // 4, kind="eng", bin=BIN, per=4, limit=limit, coef=float(x), est=est,
                 args=ARGS[c].format(x=x) + extra)
        if ladder:
            i["ladder"] = ladder
        out.append(i)

    for g in GPUS:
        for x in KI_SWEEP:
            if g == 32 or x not in IN_A2:
                add("eng.ki", "ki", x, g)
        for c in ("tfim", "noise"):
            for x in SWEEP3:
                add(f"eng.{c}", c, x, g, ladder=f"n.eng.{c}.g{g}", est=300 if float(x) < 2.5e-6 else 120)
        for c in ("ki", "tfim", "noise"):
            add(f"weakL.{c}", c, WEAK_L[g], g, est=200)
            if WEAK_S[g] != "1e-5":                     # on 32 GPUs the small load is the cutoff 1e-5 of the sweep
                add(f"weakS.{c}", c, WEAK_S[g], g)
        if g in (128, 256):                             # the same problems with 1,024 sectors (README 4.4, sector limit)
            for c, x in (("ki", WEAK_L[g]), ("ki", "3e-6"), ("ki", "1e-6"), ("tfim", WEAK_L[g])):
                add(f"eng10.{c}", c, x, g, extra=" --gmax 10", est=300)
        extra, steps = LADDER[g]
        tag = "n10" if "--gmax 10" in extra else "n12"
        for x, est in steps:
            out.append(dict(id=f"{tag}.ki.{x}.g{g}", nodes=g // 4, kind="eng", bin=BIN, per=4, limit=7200, coef=float(x), est=est,
                            ladder=f"{tag}.ki.g{g}", args=ARGS["ki"].format(x=x) + extra))
    return out


def old_open():
    """The SectorQ runs still open in the other lists: they would run with the old engine."""
    done = set(os.listdir(os.path.join(ROOT, "queue_s", "done")))
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "queue_s", "todo*.jsonl"))):
        if os.path.basename(path) in ("todo_a1_n.jsonl", "todo_a2_i.jsonl"):
            continue
        for line in open(path):
            if line.strip():
                i = json.loads(line)
                if i["kind"] == "eng" and i.get("bin") != BIN and i["id"] not in done:
                    out.append(i["id"])
    return out


if __name__ == "__main__":
    its, old = items(), old_open()
    by = {}
    for i in its:
        by[i["nodes"] * 4] = by.get(i["nodes"] * 4, 0) + 1
    if "--write" not in sys.argv:
        for i in its:
            print(json.dumps(i))
        print(f"{len(its)} runs, by GPU count {by}; {len(old)} open runs of the old engine: {' '.join(old)}", file=sys.stderr)
        sys.exit(0)
    path = os.path.join(ROOT, "queue_s", "todo_a1_n.jsonl")
    with open(path + ".tmp", "w") as f:
        for i in its:
            f.write(json.dumps(i) + "\n")
    os.replace(path + ".tmp", path)
    for sid in old:
        with open(os.path.join(ROOT, "queue_s", "done", sid), "w") as f:
            json.dump(dict(status="superseded", note="SectorQ is run again with bin/pp_engine_i (todo_a1_n.jsonl, todo_a2_i.jsonl)"), f)
    print(f"wrote {len(its)} runs to {path}, by GPU count {by}; superseded {len(old)} open runs of the old engine")
