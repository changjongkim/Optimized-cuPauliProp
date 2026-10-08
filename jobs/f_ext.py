#!/usr/bin/env python3
"""Campaign f, external baselines: executes the lines of queue_f/ext_todo.txt that need as many nodes as hosts given.

  python3 jobs/f_ext.py --job <interactive allocation id> --hosts <host[,host...]>

Uses bench/external.py of the worktree kcj/pauliprop-sectorq-impl (adapters, pinned sources, environments);
nothing is written there, the results go to results/f_ext/. A run is claimed with mkdir queue_f/ext_lock/<id>;
a finished run leaves queue_f/ext_done/<id> with its status. A ladder (implementation, circuit, node count) is
not continued to smaller cutoffs after a failure (timeout, out of memory, unsupported).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPL = "/pscratch/sd/s/sgkim/kcj/pauliprop-sectorq-impl"
sys.path.insert(0, os.path.join(IMPL, "bench"))
from external import run_external          # noqa: E402

Q = os.path.join(ROOT, "queue_f")
OUT = os.path.join(ROOT, "results", "f_ext")


def status_of(item_id):
    p = os.path.join(Q, "ext_done", item_id)
    return open(p).read().split()[0] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, type=int)
    ap.add_argument("--hosts", required=True)
    a = ap.parse_args()
    hosts = a.hosts.split(",")
    for d in ("ext_lock", "ext_done"):
        os.makedirs(os.path.join(Q, d), exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    items = [l.strip().split("|") for l in open(os.path.join(Q, "ext_todo.txt")) if l.strip()]
    for k, (item_id, nodes, backend, case, coef, policy, rpn, threads, tmo) in enumerate(items):
        if int(nodes) != len(hosts) or status_of(item_id):
            continue
        # earlier (larger) cutoffs of the same ladder: wait for them, stop after a failure
        ladder = [i[0] for i in items[:k] if (i[2], i[3], i[1]) == (backend, case, nodes)]
        prev = [status_of(i) for i in ladder]
        if any(s is None for s in prev):
            continue                                   # an earlier step is still open (another runner has it)
        if any(s != "success" for s in prev):
            open(os.path.join(Q, "ext_done", item_id), "w").write("skipped_after_failure\n")
            continue
        try:
            os.mkdir(os.path.join(Q, "ext_lock", item_id))
        except FileExistsError:
            continue
        open(os.path.join(Q, "ext_lock", item_id, "owner"), "w").write(f"{a.job} {a.hosts}\n")
        name = item_id
        n = 0
        while os.path.exists(os.path.join(OUT, name + ".launch.json")):     # a run cut off by the end of an allocation
            n += 1
            name = f"{item_id}.retry{n}"
        source = os.path.join(IMPL, "results", "own_perf_v1_n1", case + "_engine_r0.ir.json")
        t0 = time.time()
        try:
            r = run_external(a.job, int(nodes), backend, source, OUT, name, policy, float(coef), int(rpn), int(threads),
                             hosts=a.hosts, timeout=float(tmo), repeats=1)
            status = r["status"]
            line = {k2: r.get(k2) for k2 in ("status", "expectation", "solver_seconds", "cold_end_to_end_seconds")}
        except RuntimeError as e:                      # not enough time left in the allocation: give the run back
            os.remove(os.path.join(Q, "ext_lock", item_id, "owner"))
            os.rmdir(os.path.join(Q, "ext_lock", item_id))
            print("stopping:", str(e)[:200], flush=True)
            return
        except Exception as e:                         # adapter or launch problem: record it, keep going
            status, line = "error", {"status": "error", "message": repr(e)[:300]}
        open(os.path.join(Q, "ext_done", item_id), "w").write(status + "\n")
        with open(os.path.join(Q, "ext.log"), "a") as f:
            f.write(json.dumps(dict(line, id=item_id, job=a.job, hosts=a.hosts, seconds=round(time.time() - t0, 1),
                                    time=time.strftime("%m-%dT%H:%M:%S"))) + "\n")


if __name__ == "__main__":
    main()
