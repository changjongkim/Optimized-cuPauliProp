#!/usr/bin/env python3
"""Tables of campaign g: everything measured on 1, 2, 4, 8 and 16 GPUs (queue_g, logs/g, results/g_ext).

  python3 analysis/campaign_g.py [circuit ...]      progress and the tables (all circuits by default)

A cell is the solve time in seconds, or why there is none: oom (out of memory), tl (time limit), tl* (not run: the
earlier steps predict more than the limit), - (not run: an easier step of the same ladder failed), blank (open).
The deferred-deduplication baseline is the fastest of its three intervals that completed (at the truncations only,
plus every 3 or every 2 operations).
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
GPUS = (1, 2, 4, 8, 16)
NAMES = {"ki": "kicked Ising", "tfim": "TFIM heavy-hex", "noise": "noisy kicked Ising", "tfimsq": "TFIM 11x11",
         "xxz": "XXZ 8x8", "qaoa": "QAOA n=100", "hea": "ansatz n=100"}
SHORT = {"oom": "oom", "tl": "tl", "fail": "fail", "predicted_tl": "tl*", "skipped_after_failure": "-"}


def load():
    items, state = [], {}
    for path in sorted(glob.glob(os.path.join(Q, "todo*.jsonl")) + glob.glob(os.path.join(Q, "later", "todo*.jsonl"))):
        items += [json.loads(l) for l in open(path) if l.strip()]
    for i in items:
        p = os.path.join(Q, "done", i["id"])
        if os.path.exists(p):
            try:
                state[i["id"]] = json.load(open(p))
            except ValueError:
                pass
    return items, state


def cell(s):
    if s is None:
        return ""
    if s["status"] == "ok":
        w = s.get("wall")
        return "?" if w is None else ("%.2f" % w if w < 10 else "%.1f" % w if w < 1000 else "%.0f" % w)
    return SHORT.get(s["status"], s["status"][:7])


def engine_facts(item_id, s):
    """Boundary terms, largest count inside a window, host memory over all nodes (GiB) of a SectorQ run."""
    if not s or s["status"] != "ok" or not s.get("log"):
        return None
    try:
        text = open(os.path.join(ROOT, s["log"]), errors="replace").read()
    except OSError:
        return None
    b = re.search(r"^max_terms_at_window_boundary=(\d+)", text, re.M)
    i = re.search(r"^max_terms_inside_a_window=(\d+)", text, re.M)
    h = re.search(r"GiB on the busiest node, ([\d.]+) GiB over all nodes", text)
    return (int(b.group(1)) if b else None, int(i.group(1)) if i else None, float(h.group(1)) if h else None)


def best(states):
    ok = [s for s in states if s and s["status"] == "ok" and s.get("wall") is not None]
    if ok:
        return min(ok, key=lambda s: s["wall"])
    known = [s for s in states if s]
    for want in ("oom", "tl", "fail", "predicted_tl", "skipped_after_failure"):
        for s in known:
            if s["status"] == want:
                return s
    return None


def terms(n):
    return "" if n is None else ("%.3g M" % (n / 1e6) if n < 1e9 else "%.3g B" % (n / 1e9))


def main():
    items, state = load()
    mine = {k: s for k, s in state.items() if not k.startswith("test.")}
    count = {}
    for s in mine.values():
        key = "taken from campaign f" if s.get("reused") else s["status"]
        count[key] = count.get(key, 0) + 1
    print("items: %d; open: %d; %s" % (len(items), sum(1 for i in items if i["id"] not in state), ", ".join(f"{k}: {v}" for k, v in sorted(count.items()))))
    # answers of all implementations with the same truncation rule
    values = {}
    for k, s in mine.items():
        p = k.split(".")
        if s["status"] == "ok" and s.get("expectation") is not None and p[0] in ("eng", "cupp", "defer", "defer2", "defer3", "spd", "orqa", "julia", "ppvm", "qiskit"):
            values.setdefault((p[1], p[2]), []).append(s["expectation"])
    spread = max([max(v) - min(v) for v in values.values() if len(v) > 1] or [0.0])
    print("expectation values: %d settings with more than one completed run, largest spread %.2e" % (sum(1 for v in values.values() if len(v) > 1), spread))
    ladders = {}
    for i in items:
        p = i["id"].split(".")
        if p[0] == "eng" and len(p) == 4:
            ladders.setdefault(p[1], [])
            if p[2] not in ladders[p[1]]:
                ladders[p[1]].append(p[2])
    want = sys.argv[1:] or list(NAMES)
    for c in want:
        if c not in ladders:
            continue
        print(f"\n## {NAMES[c]}: seconds on 1 / 2 / 4 / 8 / 16 GPUs")
        print("%-7s %-22s %s   %-9s %-9s %-10s %s" % ("cutoff", "implementation", " ".join("%7s" % g for g in GPUS), "terms", "inside", "host GiB", "expectation"))
        for coef in sorted(ladders[c], key=float, reverse=True):
            get = lambda tag, g: state.get(f"{tag}.{c}.{coef}.g{g}")
            eng = [get("eng", g) for g in GPUS]
            facts = next((f for f in (engine_facts(f"eng.{c}.{coef}.g{g}", s) for g, s in zip(GPUS, eng)) if f), (None, None, None))
            host = next((engine_facts("", s)[2] for s in reversed(eng) if engine_facts("", s) and engine_facts("", s)[2]), None)
            value = next((s["expectation"] for s in eng if s and s.get("expectation") is not None), None)
            print("%-7s %-22s %s   %-9s %-9s %-10s %s" % (coef, "SectorQ", " ".join("%7s" % cell(s) for s in eng), terms(facts[0]), terms(facts[1]),
                                                       "" if host is None else "%.1f" % host, "" if value is None else "%+.4f" % value))
            base = [get("cupp", g) for g in GPUS]
            if any(base):
                print("%-7s %-22s %s" % ("", "cuPauliProp", " ".join("%7s" % cell(s) for s in base)))
            deferred = [None] + [best([get("defer", g), get("defer3", g), get("defer2", g)]) for g in GPUS[1:]]
            if any(deferred):
                print("%-7s %-22s %s" % ("", "cuPauliProp, deferred", " ".join("%7s" % cell(s) for s in deferred)))
            orig = [get("orig", g) for g in GPUS]
            if any(orig):
                print("%-7s %-22s %s" % ("", "cuPauliProp, original", " ".join("%7s" % cell(s) for s in orig)))
    # how many times faster, where both finished
    print("\n## SectorQ against cuPauliProp on the same GPUs (cuPauliProp seconds / SectorQ seconds), largest size both finished")
    for c in want:
        if c not in ladders:
            continue
        row = []
        for g in GPUS:
            pick = None
            for coef in sorted(ladders[c], key=float):
                a, b = state.get(f"eng.{c}.{coef}.g{g}"), state.get(f"cupp.{c}.{coef}.g{g}")
                if a and b and a["status"] == "ok" and b["status"] == "ok" and a.get("wall") and b.get("wall"):
                    pick = (coef, b["wall"] / a["wall"])
                    break
            row.append("%7s" % "" if pick is None else "%s: %.1fx" % pick)
        print("%-20s %s" % (NAMES[c], "   ".join("%-13s" % r for r in row)))
    # how far each implementation got
    print("\n## smallest cutoff finished on 1 / 2 / 4 / 8 / 16 GPUs")
    for c in want:
        if c not in ladders:
            continue
        for label, tags in (("SectorQ", ("eng",)), ("cuPauliProp", ("cupp",)), ("cuPauliProp, deferred", ("defer", "defer3", "defer2"))):
            row = []
            for g in GPUS:
                done = [float(coef) for coef in ladders[c] for t in tags
                        if (state.get(f"{t}.{c}.{coef}.g{g}") or {}).get("status") == "ok"]
                still = any(f"{t}.{c}.{coef}.g{g}" not in state for coef in ladders[c] for t in tags[:1]
                            if any(i["id"] == f"{t}.{c}.{coef}.g{g}" for i in items))
                row.append(("%.0e" % min(done) if done else "") + ("+" if still else ""))
            print("%-20s %-22s %s" % (NAMES[c] if label == "SectorQ" else "", label, " ".join("%8s" % r for r in row)))
    print("(+ : the ladder is still running)")


if __name__ == "__main__":
    main()
