#!/usr/bin/env python3
"""One line per run of the scaling logs (logs/pp_e12_g*-*.out and logs/pp_e13_g*-*.out).

Work is the number of term checks of the integrity line (terms at the window boundaries,
summed over the run); the rate is that number per second and GPU. In job set e12 choosing the rows
took a time that grew with the number of nodes (column plan); e13 repeats the engine runs without it.
"""
import glob
import os
import re

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def runs(pattern="pp_e1[23]_g*-*.out"):
    rows = []
    for path in sorted(glob.glob(os.path.join(LOGS, pattern))):
        cur = None
        for line in open(path, errors="replace"):
            m = re.match(r"===== (engine|cupp) nodes=(\d+) gpus=(\d+) args: (.*)", line)
            if m:
                a = m.group(4).split()
                opt = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
                cur = dict(kind=m.group(1), gpus=int(m.group(3)), circuit=opt["--circuit"], coef=opt["--coef"],
                           gmax=opt.get("--gmax", "8"), chunk=opt.get("--chunk", ""), tuned=opt.get("--cupp-tuned") == "1",
                           k=opt.get("--cupp-dedup-every", ""), wall=None, checks=None, exch=None, sent=None, value=None,
                           exit=None, note="", host=None, runs=None, splits=None, wait=None, ops=None, idle=None,
                           running=None, d2h=None, h2d=None, part=None, plan=None, log=os.path.basename(path), set=os.path.basename(path)[3:6])
                rows.append(cur)
                continue
            if cur is None:
                continue
            if "skipped" in line:
                cur["note"] = "skipped"
            if "FATAL" in line or "out of memory" in line or "failed at operation" in line:
                cur["note"] = "out of memory / failed"
            m = re.match(r"expectation=(\S+)", line)
            if m:
                cur["value"] = m.group(1)
            m = re.match(r"wall=([\d.]+) s", line)
            if m:
                cur["wall"] = float(m.group(1))
                for key, name in (("ops", "operations"), ("idle", "idle"), ("d2h", "gpu->host"), ("h2d", "host->gpu"), ("part", "partition")):
                    o = re.search(name + r"=([\d.]+)", line)
                    cur[key] = float(o.group(1)) if o else None
                o = re.search(r"planning=([\d.]+)", line)
                cur["plan"] = float(o.group(1)) if o else None
            m = re.match(r"integrity: (\d+) term checks", line)
            if m:
                cur["checks"] = int(m.group(1))
            m = re.search(r"between nodes: ([\d.]+) GB sent, ([\d.]+) s per node", line)
            if m:
                cur["sent"], cur["exch"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"host memory in chunks: ([\d.]+) GiB", line)
            if m:
                cur["host"] = float(m.group(1))
            m = re.search(r"bucket_runs=(\d+) splits_on_gpu=(\d+)", line)
            if m:
                cur["runs"], cur["splits"] = int(m.group(1)), int(m.group(2))
            m = re.search(r"windows running=([\d.]+), waiting for the slowest node=([\d.]+)", line)
            if m:
                cur["running"], cur["wait"] = float(m.group(1)), float(m.group(2))
            m = re.match(r"exit=(\d+)", line)
            if m:
                cur["exit"] = int(m.group(1))
    return rows


def main():
    f = lambda v, p="%.1f": "-" if v is None else p % v
    head = ("set", "kind", "gpus", "circuit", "coef", "gmax", "chunk", "wall", "checks(B)", "M/s/GPU", "ops/GPU", "exch", "wait", "plan", "hostGiB", "runs", "splits", "exit", "value/note")
    fmt = "%-3s %-7s %4s %-9s %-7s %-4s %-6s %8s %9s %8s %8s %6s %6s %6s %7s %6s %6s %4s  %s"
    print(fmt % head)
    key = lambda r: (r["kind"] != "engine", r["set"], r["circuit"], -float(r["coef"]), r["gmax"], r["chunk"], r["tuned"], r["gpus"])
    for r in sorted(runs(), key=key):
        rate = None if not (r["wall"] and r["checks"]) else r["checks"] / r["wall"] / r["gpus"] / 1e6
        kind = "engine" if r["kind"] == "engine" else ("tuned" + r["k"] if r["tuned"] else "cupp")
        print(fmt % (r["set"], kind, r["gpus"], r["circuit"], r["coef"], r["gmax"], r["chunk"], f(r["wall"], "%.2f"),
                     f(None if r["checks"] is None else r["checks"] / 1e9, "%.2f"), f(rate, "%.2f"),
                     f(None if r["ops"] is None else r["ops"] / r["gpus"], "%.2f"), f(r["exch"]), f(r["wait"]), f(r["plan"]), f(r["host"]),
                     f(r["runs"], "%d"), f(r["splits"], "%d"), f(r["exit"], "%d"), (r["value"] or "") + " " + r["note"]))


if __name__ == "__main__":
    main()
