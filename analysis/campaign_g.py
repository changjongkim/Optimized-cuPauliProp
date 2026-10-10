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
import time

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
    extra_tables(items, state, ladders, want)


WEAK = {"L": {1: "2e-5", 2: "1.4e-5", 4: "1e-5", 8: "7e-6", 16: "4.8e-6"},
        "S": {1: "6.2e-5", 2: "4.3e-5", 4: "3e-5", 8: "2.1e-5", 16: "1.5e-5"}}
MAIN = ("ki", "tfim", "noise")
BACKENDS = (("spd", "SPD, 1 GPU", (1,)), ("orqa", "ORQA (reproduction), 64 ranks/node", (1, 2, 4)),
            ("monoprop", "MonoProp, own truncation rule", (1, 2, 4)), ("julia", "PauliPropagation.jl, 64 threads", (1,)),
            ("ppvm", "ppvm, 64 threads", (1,)), ("qiskit", "Qiskit pauli-prop, 1 thread", (1,)), ("obp", "OBP, own error budget", (1,)))
NODES_OF = {4: 1, 8: 2, 16: 4}


def ratio(a, b):
    """b seconds / a seconds where both runs completed."""
    if a and b and a["status"] == "ok" and b["status"] == "ok" and a.get("wall") and b.get("wall"):
        return b["wall"] / a["wall"]
    return None


def gpu_busy(item_id, text):
    """Mean GPU utilization (percent) during the solve, from the nvidia-smi samples of the run."""
    m = re.search(r"^solve_epoch=([\d.]+)\.\.([\d.]+)", text, re.M)
    if not m:
        return None
    t0, t1 = float(m.group(1)), float(m.group(2))
    vals = []
    for path in glob.glob(os.path.join(ROOT, "logs", "g", f"smi.{item_id}.*.csv")):
        for line in open(path, errors="replace"):
            f = line.split(",")
            if len(f) < 3:
                continue
            try:
                stamp = f[0].strip()
                t = time.mktime(time.strptime(stamp[:19], "%Y/%m/%d %H:%M:%S")) + float("0" + stamp[19:] or 0)
                if t0 <= t <= t1:
                    vals.append(float(f[2]))
            except ValueError:
                continue
    return sum(vals) / len(vals) if vals else None


def breakdown_table(state, want):
    """Where the time goes on 2, 4, 8 and 16 GPUs (runs bd: SectorQ, bdc: cuPauliProp, bdd: cuPauliProp deferred)."""
    num = lambda pat, text: (lambda m: float(m.group(1)) if m else None)(re.search(pat, text, re.M))
    pct = lambda x, wall: "" if x is None or not wall else "%3.0f%%" % (100 * x / wall)
    print("\n## where the time goes (wall seconds; shares of the wall time)")
    print("  %-20s %-6s %3s  %-9s %7s %6s  %s" % ("circuit", "cutoff", "GPU", "", "wall", "GPU%", "shares"))
    for c in MAIN:
        if c not in want:
            continue
        for coef in ("1e-4", "3e-5"):
            for g in (2, 4, 8, 16):
                for tag, name in (("bd", "SectorQ"), ("bdc", "cuPP"), ("bdd", "cuPP def")):
                    item_id = f"{tag}.{c}.{coef}.g{g}"
                    st = state.get(item_id)
                    if not st:
                        continue
                    if st["status"] != "ok" or not st.get("log"):
                        print("  %-20s %-6s %3d  %-9s %7s" % (NAMES[c], coef, g, name, cell(st)))
                        continue
                    try:
                        text = open(os.path.join(ROOT, st["log"]), errors="replace").read()
                    except OSError:
                        continue
                    wall = st.get("wall")
                    busy = gpu_busy(item_id, text)
                    if tag == "bd":
                        ops = num(r"operations=([\d.]+)", text)
                        part = num(r"partition=([\d.]+)", text)
                        cp = (num(r"host->gpu=([\d.]+)", text) or 0) + (num(r"gpu->host=([\d.]+)", text) or 0)
                        exch = num(r"GB sent, ([\d.]+) s per node", text)
                        wait = num(r"waiting for the slowest node=([\d.]+)", text)
                        waited = num(r"workers waited ([\d.]+) s", text)
                        shares = "gates %s  re-sectoring %s  copies %s  exchange %s  other nodes %s  waiting for rows %s" % (
                            pct(ops / g if ops else None, wall), pct(part / g if part else None, wall), pct(cp / g, wall), pct(exch, wall), pct(wait, wall),
                            pct(waited / g if waited is not None else None, wall))
                    else:
                        app = num(r"application=([\d.]+)", text)
                        mpi = num(r"cupp mpi, seconds per rank: .*total=([\d.]+)", text)
                        coll = num(r"collective=([\d.]+)", text)
                        per = num(r"cupp mpi allgather: calls/rank=\d+ \(([\d.]+) per operation\)", text)
                        shares = "inside MPI %s (collective %s)  allgather calls per operation %s" % (pct(mpi, wall), pct(coll, wall), "" if per is None else "%.1f" % per)
                    print("  %-20s %-6s %3d  %-9s %7s %6s  %s" % (NAMES[c], coef, g, name, cell(st), "" if busy is None else "%.0f" % busy, shares))


def extra_tables(items, state, ladders, want):
    ids = {i["id"] for i in items}
    get = state.get

    # external baselines: the sweep
    print("\n## external baselines: solve seconds per cutoff (CPU implementations: 1, 2, 4 nodes stand for 4, 8, 16 GPUs)")
    for c in want:
        if c not in ladders:
            continue
        coefs = set()                      # cutoffs of the external ladders of this circuit (a cutoff may contain a dot: 6.2e-5)
        for i in items:
            m = re.match(r"[a-z]+\.%s\.(.+)\.n\d+$" % c, i["id"])
            if m and i.get("kind") == "ext" and i.get("ladder"):
                coefs.add(m.group(1))
        coefs = sorted(coefs, key=float, reverse=True)
        if not coefs:
            continue
        print(f"\n{NAMES[c]}\n%-46s %s" % ("", " ".join("%7s" % x for x in coefs)))
        for g in (1, 4, 8, 16):
            print("%-46s %s" % (f"SectorQ, {g} GPU" + ("s" if g > 1 else ""), " ".join(
                "%7s" % cell(get(f"eng.{c}.{x}.g{g}") or get(f"weakL.{c}.{x}.g{g}") or get(f"weakS.{c}.{x}.g{g}")) for x in coefs)))
        for b, label, node_counts in BACKENDS:
            for n in node_counts:
                if not any(f"{b}.{c}.{x}.n{n}" in ids for x in coefs):
                    continue
                name = label + (f", {n} node" + ("s" if n > 1 else "") if len(node_counts) > 1 else "")
                print("%-46s %s" % (name, " ".join("%7s" % cell(get(f"{b}.{c}.{x}.n{n}")) for x in coefs)))
    print("\n## SectorQ against the external baselines on the same nodes (their seconds / SectorQ seconds), largest size both finished")
    for c in want:
        if c not in ladders:
            continue
        out = []
        for b, label, node_counts in BACKENDS:
            for n in node_counts:
                g = 1 if b == "spd" else 4 * n
                pick = None
                for x in sorted(ladders[c], key=float):
                    r = ratio(get(f"eng.{c}.{x}.g{g}"), get(f"{b}.{c}.{x}.n{n}"))
                    if r:
                        pick = f"{b} n{n} ({x}): {r:.1f}x"
                        break
                if pick:
                    out.append(pick)
        if out:
            print("%-20s %s" % (NAMES[c], "   ".join(out)))

    # weak scaling
    print("\n## weak scaling: seconds on 1 / 2 / 4 / 8 / 16 GPUs, cutoff lowered with the GPU count (L: 1e-5 on 4 GPUs, S: 3e-5 on 4 GPUs)")
    for load in ("L", "S"):
        for c in MAIN:
            if c not in want:
                continue
            def pick(tag_weak, tag_sweep, g):
                x = WEAK[load][g]
                return get(f"{tag_weak}.{c}.{x}.g{g}") or get(f"{tag_sweep}.{c}.{x}.g{g}")
            eng = [pick(f"weak{load}", "eng", g) for g in GPUS]
            rows = [("SectorQ", eng), ("cuPauliProp", [pick(f"weak{load}c", "cupp", g) for g in GPUS]),
                    ("cuPauliProp, deferred", [None] + [pick(f"weak{load}d", "defer", g) for g in GPUS[1:]])]
            for b in ("orqa", "monoprop"):
                r = []
                for g in GPUS:
                    x = WEAK[load][g]
                    r.append(get(f"weak{load}.{b}.{c}.{x}.g{g}") or (get(f"{b}.{c}.{x}.n{NODES_OF[g]}") if g in NODES_OF else None))
                rows.append((b + " (16, 32 cores; 1, 2, 4 nodes)", r))
            print(f"\n{NAMES[c]}, load {load}: cutoffs " + " / ".join(WEAK[load][g] for g in GPUS))
            for name, r in rows:
                if any(r):
                    print("  %-34s %s" % (name, " ".join("%7s" % cell(x) for x in r)))
            facts = [engine_facts("", x) for x in eng]
            if any(facts):
                per = ["" if not f or not f[0] else "%.1f" % (f[0] / g / 1e6) for f, g in zip(facts, GPUS)]
                print("  %-34s %s" % ("boundary terms per GPU (millions)", " ".join("%7s" % x for x in per)))

    breakdown_table(state, want)

    # less GPU memory, design parameters
    print("\n## SectorQ with less GPU memory, cutoff 1e-5 (fraction of the 40 GB; default 0.9): seconds on 1 GPU and on 4 GPUs")
    for c in MAIN:
        if c not in want:
            continue
        for g in (1, 4):
            row = [("0.9", get(f"eng.{c}.1e-5.g{g}"))] + [(f, get(f"vram.{c}.1e-5.g{g}.f{f}")) for f in ("0.45", "0.2", "0.1")]
            print("  %-20s %2d GPU  %s" % (NAMES[c], g, "   ".join(f"{f}: {cell(x) or 'open'}" for f, x in row)))
    print("\n## design parameters, cutoff 1e-5: seconds on 4 and 16 GPUs (desired terms per sector; sectors per GPU, default 2)")
    for c in MAIN:
        if c not in want:
            continue
        for g in (4, 16):
            base = get(f"eng.{c}.1e-5.g{g}")
            tg = sorted(((int(i["id"].rsplit(".t", 1)[1]), get(i["id"])) for i in items if i["id"].startswith(f"tgt.{c}.1e-5.g{g}.t")))
            pg = sorted(((int(i["id"].rsplit(".p", 1)[1]), get(i["id"])) for i in items if i["id"].startswith(f"pg.{c}.1e-5.g{g}.p")))
            print("  %-20s %2d GPUs  default: %s   target %s   per GPU %s" % (NAMES[c], g, cell(base) or "open",
                  "  ".join(f"{t / 1e6:g}M: {cell(x) or 'open'}" for t, x in tg), "  ".join(f"{k}: {cell(x) or 'open'}" for k, x in pg)))

    # the engine before and after the change of 2026-10-10 (rows chosen while the window runs)
    old_dir = os.path.join(Q, "done_g")
    if os.path.isdir(old_dir):
        pairs = []
        for name in os.listdir(old_dir):
            new = get(name)
            try:
                old = json.load(open(os.path.join(old_dir, name)))
            except ValueError:
                continue
            r = ratio(new, old)
            if r and name.startswith("eng."):
                pairs.append((name, old["wall"], new["wall"], r))
        if pairs:
            print("\n## engine before and after rows are chosen while the window runs (multi-GPU sweep runs measured with both): seconds, old / new")
            for c in want:
                for g in (2, 4, 8, 16):
                    row = sorted((p for p in pairs if p[0].split(".")[1] == c and p[0].endswith(f".g{g}")), key=lambda p: -float(p[0].split(".")[2]))
                    if row:
                        print("  %-20s %2d GPUs  %s" % (NAMES.get(c, c), g, "  ".join(f"{p[0].split('.')[2]}: {p[1]:.3g}/{p[2]:.3g} ({p[3]:.2f}x)" for p in row)))

    # convergence: kicked Ising at other rotation angles
    conv = sorted({i["id"].split(".")[2] for i in items if i["id"].startswith("conv.eng.")}, key=lambda t: float(t[1:]))
    if conv and any(get(i["id"]) for i in items if i["id"].startswith("conv.")):
        print("\n## convergence: kicked Ising at other rotation angles, expectation value by cutoff (SectorQ, 16 GPUs; the value of the smallest finished cutoff last)")
        rungs = sorted({i["id"].split(".")[3] for i in items if i["id"].startswith("conv.eng.")}, key=float, reverse=True)
        print("  %-8s %s" % ("angle/pi", " ".join("%9s" % x for x in rungs)))
        for t in conv:
            vals = []
            for x in rungs:
                s = get(f"conv.eng.{t}.{x}.g16") or get(f"conv.eng.{t}.{x}.g1")
                vals.append("%+.5f" % s["expectation"] if s and s["status"] == "ok" and s.get("expectation") is not None else (cell(s) if s else ""))
            print("  %-8s %s" % (t[1:], " ".join("%9s" % v for v in vals)))
        print("  how far each gets (smallest cutoff finished): " + "; ".join(
            f"{label} " + ", ".join(f"{t[1:]}: " + (min((x for x in rungs if (get(f'conv.{tag}.{t}.{x}.g{g}') or {}).get('status') == 'ok'), key=float, default='-')) for t in conv)
            for label, tag, g in (("SectorQ 1 GPU", "eng", 1), ("cuPauliProp 1 GPU", "cupp", 1), ("cuPauliProp 16 GPUs", "cupp", 16))))
    more_tables(items, state)


EXACT = {4: 0.802867136639, 6: 0.790133273060}      # 5x6 TFIM, state vector (analysis/exact_sv.py)


def more_tables(items, state):
    """Checkpoint and resume, the exact-value check, baselines given more resources than they use, core counts."""
    get = state.get
    ids = [i["id"] for i in items]
    value = lambda s: "" if not s or s.get("expectation") is None else "%.12f" % s["expectation"]

    if any(i.startswith("ckpt") for i in ids):
        print("\n## checkpoint and resume, kicked Ising, cutoff 1e-5: the run is stopped at a checkpoint and resumed from the file")
        for g in (4, 16):
            whole, w, r = get(f"eng.ki.1e-5.g{g}"), get(f"ckptw.ki.1e-5.g{g}"), get(f"ckptr.ki.1e-5.g{g}")
            if not (w or r):
                continue
            note = ""
            if r and r.get("log"):
                try:
                    m = re.search(r"^resumed at operation (\d+) with (\d+) terms", open(os.path.join(ROOT, r["log"]), errors="replace").read(), re.M)
                    note = "resumed at operation %s with %s terms" % (m.group(1), terms(int(m.group(2)))) if m else ""
                except OSError:
                    pass
            same = whole and r and whole.get("expectation") is not None and r.get("expectation") is not None and abs(whole["expectation"] - r["expectation"]) < 1e-10
            print("  %2d GPUs  whole run: %s s   resumed part: %s s   value whole %s, resumed %s (%s)   %s" % (
                g, cell(whole) or "open", cell(r) or "open", value(whole), value(r), "same" if same else "DIFFERENT" if whole and r and r["status"] == "ok" else "open", note))

    rows = sorted((i for i in ids if i.startswith("exact.tfim30.")), key=lambda i: (int(i.split(".")[2][1:]), -float(i.split(".s")[1].split(".", 1)[1].rsplit(".g", 1)[0])))
    if rows:
        print("\n## exact-value check: 5x6 TFIM (30 qubits), SectorQ on 4 GPUs; error against the state-vector value")
        print("  %-5s %-7s %8s  %-15s %-9s %s" % ("steps", "cutoff", "seconds", "value", "error", "terms"))
        for i in rows:
            steps = int(i.split(".")[2][1:])
            coef = i.split(f".s{steps}.", 1)[1].rsplit(".g", 1)[0]
            s = get(i)
            facts = engine_facts(i, s) or (None, None, None)
            err = "" if not s or s.get("expectation") is None else "%.5f" % abs(s["expectation"] - EXACT[steps])
            print("  %-5d %-7s %8s  %-15s %-9s %s" % (steps, coef, cell(s) or "open", value(s), err, terms(facts[0])))

    giv = [i for i in ids if i.startswith("giv.")]
    if giv:
        print("\n## baselines that run on one node or one GPU, given more: seconds (own: 1 node, SPD 1 GPU; then what they were given)")
        groups = {}
        for i in giv:
            m = re.match(r"giv\.([a-z]+)\.([a-z]+)\.(.+)\.([ng])(\d+)$", i)
            groups.setdefault(m.group(1, 2, 3), []).append((m.group(4), int(m.group(5)), i))
        for (b, c, x), members in groups.items():
            own = get(f"{b}.{c}.{x}.n1")
            print("  %-9s %-18s %-6s own: %-7s %s" % (b, NAMES.get(c, c), x, cell(own) or "open", "  ".join(
                "%d %s: %s" % (n, "nodes" if unit == "n" else "GPUs", cell(get(i)) or "open") for unit, n, i in sorted(members, key=lambda t: t[1]))))

    thr = [i for i in ids if i.startswith("thr.")]
    if thr:
        print("\n## CPU baselines by core count on one node: seconds (64 cores is the setting of the sweep; Qiskit pauli-prop: 1 thread there)")
        groups = {}
        for i in thr:
            m = re.match(r"thr\.([a-z]+)\.([a-z]+)\.(.+)\.c(\d+)$", i)
            groups.setdefault(m.group(1, 2, 3), []).append((int(m.group(4)), i))
        for (b, c, x), members in groups.items():
            base = get(f"{b}.{c}.{x}.n1")
            cells = ["%d: %s" % (n, cell(get(i)) or "open") for n, i in sorted(members)]
            cells.insert(0, "1: %s" % (cell(base) or "open")) if b == "qiskit" else cells.append("64: %s" % (cell(base) or "open"))
            print("  %-9s %-18s %-6s %s" % (b, NAMES.get(c, c), x, "  ".join(cells)))


if __name__ == "__main__":
    main()
