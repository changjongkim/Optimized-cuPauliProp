#!/usr/bin/env python3
"""Tables of campaign f: everything measured on 1, 2, 4, 8 and 16 GPUs (logs/f, results/f_ext).

  python3 analysis/campaign_f.py            progress and the tables
  python3 analysis/campaign_f.py --check    only the comparison of expectation values

A cell is the solve time in seconds, or why there is none: oom (out of memory or capacity), tl (time limit),
fail (other failure), blank (not run yet). The deferred-deduplication baseline is the fastest of the three
intervals that completed (at the truncations only, plus every 3 or every 2 operations).
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs", "f")
EXT = os.path.join(ROOT, "results", "f_ext")
Q = os.path.join(ROOT, "queue_f")
GPUS = (1, 2, 4, 8, 16)
NAMES = {"ki": "kicked Ising", "tfim": "TFIM heavy-hex", "noise": "noisy kicked Ising", "tfimsq": "TFIM 11x11",
         "xxz": "XXZ 8x8", "qaoa": "QAOA n=100", "hea": "ansatz n=100"}


def read_runs():
    runs = {}
    for path in glob.glob(os.path.join(LOGS, "*.out")):
        item = os.path.basename(path)[:-4]
        m = re.match(r"(eng|long|weak|cupp|defer|defer3|defer2|orig)\.([a-z]+)\.([0-9.e-]+)\.g(\d+)$", item)
        if not m:
            continue
        text = open(path, errors="replace").read()
        r = dict(tag=m.group(1), circuit=m.group(2), coef=m.group(3), gpus=int(m.group(4)), wall=None, value=None, exit=None, status=None)
        w = re.search(r"^wall=([\d.]+) s", text, re.M)
        v = re.search(r"^expectation=(\S+)", text, re.M)
        e = re.search(r"^exit=(\d+)", text, re.M)
        r["wall"] = float(w.group(1)) if w else None
        r["value"] = float(v.group(1)) if v else None
        r["exit"] = int(e.group(1)) if e else None
        b = re.search(r"^max_terms_at_window_boundary=(\d+)", text, re.M)
        r["boundary"] = int(b.group(1)) if b else None
        if r["exit"] is None:
            r["status"] = "running"
        elif r["exit"] == 0 and r["wall"] is not None:
            r["status"] = "ok"
        elif r["exit"] == 124:
            r["status"] = "tl"
        elif re.search(r"out of memory|capacity|FATAL", text):
            r["status"] = "oom"
        else:
            r["status"] = "fail"
        runs[item] = r
    return runs


def cell(r):
    if r is None:
        return ""
    if r["status"] == "ok":
        return ("%.2f" if r["wall"] < 10 else "%.1f") % r["wall"]
    return r["status"]


def best(rs):
    ok = [r for r in rs if r is not None and r["status"] == "ok"]
    if ok:
        return min(ok, key=lambda r: r["wall"])
    done = [r for r in rs if r is not None]
    return done[0] if done else None


def order(coefs):
    return sorted(coefs, key=lambda c: -float(c))


def main():
    runs = read_runs()
    total = sum(1 for _ in open(os.path.join(Q, "todo.txt")))
    done = len(os.listdir(os.path.join(Q, "done")))
    ext_total = sum(1 for _ in open(os.path.join(Q, "ext_todo.txt"))) if os.path.exists(os.path.join(Q, "ext_todo.txt")) else 0
    ext_done_dir = os.path.join(Q, "ext_done")
    ext_done = len(os.listdir(ext_done_dir)) if os.path.isdir(ext_done_dir) else 0
    print(f"progress: SectorQ and cuPauliProp {done} of {total}; external baselines {ext_done} of {ext_total}")

    # same answer?
    worst = 0.0
    groups = {}
    for r in runs.values():
        if r["status"] == "ok" and r["value"] is not None:
            groups.setdefault((r["circuit"], r["coef"]), []).append(r["value"])
    bad = []
    for k, v in groups.items():
        spread = max(v) - min(v)
        worst = max(worst, spread)
        if spread > 1e-10:
            bad.append((k, min(v), max(v), len(v)))
    print(f"expectation values: {len(groups)} settings with completed runs, largest spread between runs of one setting {worst:.2e}")
    for b in bad:
        print("  DIFFERENT:", b)
    if "--check" in sys.argv:
        return

    circuits = [c for c in NAMES if any(r["circuit"] == c for r in runs.values())]
    for c in circuits:
        coefs = order({r["coef"] for r in runs.values() if r["circuit"] == c and r["tag"] != "weak"})
        print(f"\n## {NAMES[c]}: seconds on 1 / 2 / 4 / 8 / 16 GPUs")
        print("%-8s %-22s %s   %s" % ("cutoff", "implementation", "  ".join("%7s" % g for g in GPUS), "boundary terms"))
        for coef in coefs:
            get = lambda tag, g: runs.get(f"{tag}.{c}.{coef}.g{g}")
            rows = [("SectorQ", [best([get("long", g), get("eng", g)]) for g in GPUS]),     # long: the same run with a longer time limit
                    ("cuPauliProp", [get("cupp", g) for g in GPUS]),
                    ("cuPauliProp, deferred", [None] + [best([get("defer", g), get("defer3", g), get("defer2", g)]) for g in GPUS[1:]]),
                    ("cuPauliProp, original", [get("orig", g) for g in GPUS])]
            for name, rs in rows:
                if all(r is None for r in rs):
                    continue
                bt = next((r["boundary"] for r in rs if r is not None and r.get("boundary")), None)
                print("%-8s %-22s %s   %s" % (coef, name, "  ".join("%7s" % cell(r) for r in rs), f"{bt:,}" if bt and name == "SectorQ" else ""))
        weak = [runs.get(f"weak.{c}.{coef}.g{g}") for g, coef in ((1, "2e-5"), (2, "1.4e-5"), (8, "7e-6"), (16, "4.8e-6"))]
        if any(weak):
            four = runs.get(f"eng.{c}.1e-5.g4")
            print("weak scaling (cutoff per GPU count: 2e-5, 1.4e-5, 1e-5, 7e-6, 4.8e-6): " + "  ".join("%7s" % cell(r) for r in (weak[0], weak[1], four, weak[2], weak[3])))

    # external baselines
    if os.path.isdir(ext_done_dir) and ext_done:
        print("\n## external baselines: solve seconds (status when there is none)")
        rows = {}
        for line in open(os.path.join(Q, "ext_todo.txt")):
            item, nodes, backend, case, coef, policy = line.strip().split("|")[:6]
            p = os.path.join(ext_done_dir, item)
            if not os.path.exists(p):
                continue
            status = open(p).read().split()[0]
            sec = val = None
            lp = sorted(glob.glob(os.path.join(EXT, item + "*.launch.json")))
            if lp:
                j = json.load(open(lp[-1]))
                sec, val = j.get("solver_seconds"), j.get("expectation")
            rows.setdefault((case, backend, policy, nodes), {})[coef] = (status, sec, val)
        for (case, backend, policy, nodes), d in sorted(rows.items()):
            cells = []
            for coef in order(d):
                status, sec, val = d[coef]
                short = {"skipped_after_failure": "-", "timeout": "tl", "success": "ok"}.get(status, status)
                cells.append(f"{coef}: " + (("%.2f" % sec) if (status == "success" and sec is not None) else short))
            print("%-6s %-9s %-8s nodes=%s  %s" % (case, backend, policy, nodes, "   ".join(cells)))


if __name__ == "__main__":
    main()
