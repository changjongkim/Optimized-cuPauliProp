"""Runs with bin/pp_engine_i (the exchange between nodes in rounds) for the scale jobs: queue_s/todo_a2_i.jsonl.

  env/bin/python3 jobs/i_items.py            # print what would be written
  env/bin/python3 jobs/i_items.py --write    # write the list and mark the runs it replaces as superseded

The list sorts before the other lists of queue_s, so a scale job takes these runs first.
  1. Small and medium problems (cutoff 1e-4 to 3e-6) with the default settings and with the early sectors given by
     the size expected at the end of a window (--min-bucket-early-end): does a problem get slower with more GPUs?
  2. The ladder down to the memory limit with the new exchange (a node no longer holds its output and all arrivals
     at once, and the requests in flight are bounded by the number of sectors).
The runs of the old ladder that are still open on 256 and 512 GPUs are marked superseded: with the old exchange they
stop in the exchange (128 GPUs, cutoff 3e-7: no progress for 38 minutes, job 59660399) or run out of host memory."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = "bin/pp_engine_i"
BASE = "--circuit ki_lab --weight 0 --coef {x} --target 6000000 --trace 1 --vram-frac 0.9"
SMALL = ("1e-4", "3e-5", "1e-5", "3e-6")
RULES = (("ref", ""), ("e1m", " --min-bucket-early-end 1000000"), ("e2m", " --min-bucket-early-end 2000000"),
         ("e500k", " --min-bucket-early-end 500000"))
# GPU count -> (sector limit of the ladder, cutoffs of the ladder with the seconds each is expected to take)
LADDER = {128: (" --gmax 12 --chunk 65536", (("3e-7", 3000), ("2e-7", 6000))),
          256: (" --gmax 12 --chunk 65536", (("2e-7", 3600), ("1.5e-7", 6000), ("1e-7", 7000))),
          512: (" --gmax 12 --chunk 65536", (("1.5e-7", 3600), ("1e-7", 6500)))}
SUPERSEDED = ("eng12.ki.2e-7.g256", "eng12.ki.1.5e-7.g256", "eng12.ki.1e-7.g256", "eng12.ki.1.5e-7.g512", "eng12.ki.1e-7.g512")


def items():
    out = []
    for g in (64, 128, 256, 512):
        nodes = g // 4
        for tag, extra in RULES:
            for x in SMALL:
                out.append(dict(id=f"i.{tag}.ki.{x}.g{g}", nodes=nodes, kind="eng", bin=BIN, per=4, limit=300, coef=float(x), est=40,
                                args=BASE.format(x=x) + extra))
        if g in LADDER:
            extra, steps = LADDER[g]
            for x, est in steps:
                out.append(dict(id=f"i12.ki.{x}.g{g}", nodes=nodes, kind="eng", bin=BIN, per=4, limit=7200, coef=float(x), est=est,
                                ladder=f"i12.ki.g{g}", args=BASE.format(x=x) + extra))
    return out


if __name__ == "__main__":
    its = items()
    if "--write" not in sys.argv:
        for i in its:
            print(json.dumps(i))
        print(len(its), "runs", file=sys.stderr)
        sys.exit(0)
    path = os.path.join(ROOT, "queue_s", "todo_a2_i.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for i in its:
            f.write(json.dumps(i) + "\n")
    os.replace(tmp, path)
    for sid in SUPERSEDED:
        rec = os.path.join(ROOT, "queue_s", "done", sid)
        if not os.path.exists(rec):
            with open(rec, "w") as f:
                json.dump(dict(status="superseded", by=sid.replace("eng12.", "i12."), note="the old exchange stops or runs out of host memory here"), f)
    print(f"wrote {len(its)} runs to {path}; superseded: {', '.join(SUPERSEDED)}")
