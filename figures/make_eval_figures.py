#!/usr/bin/env python3
"""Background and evaluation figures of the README.

Run times, expectation values and time breakdowns are read from the run logs in ../logs
(blocks that start with "===== engine|cupp nodes=... args: ..."). The numbers of the
background figure and of the exact-value check come from earlier logs with other formats
and are listed below with their source.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")

OURS, BASE, TUNED, INK, GRID = "#1B9E8F", "#8A8780", "#C9B99B", "#1F2328", "#DDDDD5"
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.edgecolor": "#BDBDB5", "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False, "savefig.dpi": 200,
})


# ---------------------------------------------------------------------------------------
# log reader
# ---------------------------------------------------------------------------------------
def parse(name):
    """Returns the runs of one log as a list of dicts."""
    path = os.path.join(LOGS, name)
    if not os.path.exists(path):
        return []
    runs, cur = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"===== (engine|cupp) nodes=(\d+) gpus=(\d+) args: (.*)", line)
        if m:
            args = m.group(4).split()
            opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
            gpus = int(m.group(3))
            if m.group(1) == "engine" and "--gpus" in opt:
                gpus = int(opt["--gpus"]) * int(m.group(2))
            cur = dict(kind=m.group(1), nodes=int(m.group(2)), gpus=gpus, opt=opt,
                       circuit=opt.get("--circuit"), coef=float(opt.get("--coef", "1e-4")),
                       tuned=opt.get("--cupp-tuned") == "1", vram=float(opt.get("--vram-frac", "0.9")),
                       wall=None, value=None, exit=None, fatal=False, parts={}, log=name)
            runs.append(cur)
            continue
        if cur is None:
            continue
        if "FATAL" in line or "failed at operation" in line or "out of memory" in line:
            cur["fatal"] = True
        m = re.match(r"expectation=(\S+)", line)
        if m:
            cur["value"] = float(m.group(1))
        m = re.match(r"wall=([\d.]+) s", line)
        if m:
            cur["wall"] = float(m.group(1))
            for k, v in re.findall(r"(operations|partition|host->gpu|gpu->host|idle|planning)=([\d.]+)", line):
                cur["parts"][k] = float(v)
            w = re.search(r"summed over (\d+) GPU workers", line)
            if w:
                cur["parts"]["workers"] = int(w.group(1))
        m = re.search(r"between nodes: ([\d.]+) GB sent, ([\d.]+) s per node", line)
        if m:
            cur["parts"]["exchange"] = float(m.group(2))
        m = re.match(r"max_terms_at_window_boundary=(\d+)", line)
        if m:
            cur["boundary"] = int(m.group(1))
        m = re.match(r"exit=(\d+)", line)
        if m:
            cur["exit"] = int(m.group(1))
    return runs


def ok(r):
    return r is not None and r["wall"] is not None and r["exit"] == 0 and not r["fatal"]


def find(runs, kind, circuit, coef, gpus=16, tuned=False, last=True, **opt):
    hits = [r for r in runs if r["kind"] == kind and r["circuit"] == circuit and abs(r["coef"] / coef - 1) < 1e-9
            and r["gpus"] == gpus and r["tuned"] == tuned and all(r["opt"].get(k) == v for k, v in opt.items())]
    done = [r for r in hits if ok(r)]
    if tuned and done:          # the hand-tuned baseline: the fastest deduplication interval that fits
        return min(done, key=lambda r: r["wall"])
    pool = done or hits
    return (pool[-1] if last else pool[0]) if pool else None


def secs(v):
    return f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")


def times(b, e):
    x = b / e
    return f"{x:.1f}×" if x < 10 else f"{x:.0f}×"


def clean(ax, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------
# background: cuPauliProp's distributed mode (logs/pp_prof1-59470347.out, pp_prof2-59470887.out,
# pp_prof4n-59470888.out; kicked Ising, cutoff 1e-4, 22.5 million terms, original circuit)
# ---------------------------------------------------------------------------------------
def background():
    gpus = ["1", "4", "8", "16"]
    panels = [
        ("bg_scaling.png", [2.92, 32.9, 64.8, 71.2], "Time (s)", "{:.0f}"),
        ("bg_gatecost.png", [0.05, 5.9, 22.5, 29.8], "Fixed Cost per Gate (ms)", "{:.1f}"),
        ("bg_capacity.png", [96.0, 95.8, 103.7, 87.9], "Terms per GPU\nat the Failure Point (millions)", "{:.0f}"),
    ]
    for name, vals, ylabel, fmt in panels:
        fig, ax = plt.subplots(figsize=(4.3, 3.5))
        ax.bar(gpus, vals, width=0.62, color=BASE)
        for x, v in zip(gpus, vals):
            ax.text(x, v + max(vals) * 0.02, ("%.2f" % v) if v < 1 else fmt.format(v), ha="center", va="bottom", fontsize=13)
        ax.set_ylim(0, max(vals) * 1.16)
        ax.set_xlabel("GPUs (cuPauliProp distributed)")
        ax.set_ylabel(ylabel)
        clean(ax)
        save(fig, name)


# ---------------------------------------------------------------------------------------
# speed against cuPauliProp's distributed mode, 16 GPUs
# ---------------------------------------------------------------------------------------
MEDIUM = [("ki_lab", 1e-4, "Kicked\nIsing", {}), ("tfim_hh", 1e-4, "TFIM\nheavy-hex", {}),
          ("tfim_sq", 1e-4, "TFIM\n11×11", {"--lx": "11"}), ("xxz", 1e-3, "XXZ\n6×6", {}),
          ("qaoa", 1e-4, "QAOA", {}), ("hea", 1e-5, "Ansatz", {}), ("ki_noisy", 1e-4, "Noisy\nKicked\nIsing", {})]
LARGE = [("ki_lab", 1e-5, "Kicked\nIsing", {}), ("tfim_hh", 1e-5, "TFIM\nheavy-hex", {}),
         ("tfim_sq", 1e-5, "TFIM\n11×11", {"--lx": "11"}), ("ki_noisy", 1e-5, "Noisy\nKicked\nIsing", {})]


def triple(runs_first, runs_all, c, coef, opt):
    """(engine, tuned baseline, default baseline) for one setting; prefers runs on the same nodes."""
    out = []
    for kind, tuned in (("engine", False), ("cupp", True), ("cupp", False)):
        r = find(runs_first, kind, c, coef, tuned=tuned, **opt)
        if not ok(r):
            r2 = find(runs_all, kind, c, coef, tuned=tuned, **opt)
            r = r2 if (ok(r2) or r is None) else r
        out.append(r)
    return out


def speed(run5, allruns):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 3.9), gridspec_kw={"width_ratios": [7, 4.2], "wspace": 0.14})
    table = []
    have_tuned = any(r["tuned"] and ok(r) for r in allruns)
    for ax, sets, title in ((axes[0], MEDIUM, "Fits One GPU"), (axes[1], LARGE, "Beyond One GPU (up to 1.4 billion terms)")):
        w = 0.25 if have_tuned else 0.38
        for i, (c, coef, label, opt) in enumerate(sets):
            e, t, d = triple(run5, allruns, c, coef, opt)
            table.append((label.replace("\n", " "), coef, e, t, d))
            series = [(e, OURS)] + ([(t, TUNED)] if have_tuned else []) + [(d, BASE)]
            for j, (r, col) in enumerate(series):
                x = i + (j - (len(series) - 1) / 2) * w
                if ok(r):
                    ax.bar(x, r["wall"], width=w * 0.94, color=col)
                    if j > 0 and ok(e):   # multiples: the tuned label grows to the left, the default label to the right
                        left = have_tuned and j == 1
                        ax.text(x + (w * 0.47 if left else -w * 0.47), r["wall"] * 1.12, times(r["wall"], e["wall"]),
                                ha="right" if left else "left", va="bottom", fontsize=10.5)
                elif r is not None:
                    ax.text(x, 0.075, "out of\nmemory" if r["fatal"] else "time\nlimit", ha="center", va="bottom", fontsize=9.5, rotation=90, color=col)
        ax.set_yscale("log")
        ax.set_ylim(0.06, 400)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s[2] for s in sets], fontsize=11.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=13.5, pad=8)
        clean(ax)
    axes[0].set_ylabel("Time on 16 GPUs (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OURS)] + ([plt.Rectangle((0, 0), 1, 1, color=TUNED)] if have_tuned else []) + [plt.Rectangle((0, 0), 1, 1, color=BASE)]
    names = ["This work"] + (["cuPauliProp distributed, hand-tuned"] if have_tuned else []) + ["cuPauliProp distributed"]
    fig.legend(handles, names, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_speed.png")
    return table


# ---------------------------------------------------------------------------------------
# number of GPUs and reach (kicked Ising)
# ---------------------------------------------------------------------------------------
def gpus_and_reach(run4, run5, allruns):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.7), gridspec_kw={"width_ratios": [5, 3.4], "wspace": 0.22})
    ax = axes[0]
    counts = [1, 2, 4, 8, 16]
    rows = []
    w = 0.38
    for i, g in enumerate(counts):
        e = find(allruns, "engine", "ki_lab", 1e-5, gpus=g)
        d = find(run4 + run5, "cupp", "ki_lab", 1e-5, gpus=g)
        rows.append((g, e, d))
        if ok(e):
            ax.bar(i - w / 2, e["wall"], width=w * 0.94, color=OURS)
            ax.text(i - w / 2, e["wall"] * 1.1, secs(e["wall"]), ha="center", va="bottom", fontsize=11.5)
        if ok(d):
            ax.bar(i + w / 2, d["wall"], width=w * 0.94, color=BASE)
            ax.text(i + w / 2, d["wall"] * 1.1, secs(d["wall"]) + (" (%s)" % times(d["wall"], e["wall"]) if ok(e) else ""), ha="center", va="bottom", fontsize=11.5)
        elif d is not None or g == 1:   # one GPU: fails at 96 million terms (logs/pp_prof2-59470887.out)
            ax.text(i + w / 2, 6.3, "out of memory", ha="center", va="bottom", fontsize=10.5, rotation=90, color=BASE)
        else:
            ax.text(i + w / 2, 6.3, "not run", ha="center", va="bottom", fontsize=10.5, rotation=90, color=BASE)
    ax.set_yscale("log")
    ax.set_ylim(5.5, 700)
    ax.set_xlim(-0.6, len(counts) - 0.3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(g) for g in counts])
    ax.set_xlabel("GPUs (kicked Ising, cutoff 1e-5, up to 1.4 billion terms)")
    ax.set_ylabel("Time (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    clean(ax)

    ax = axes[1]
    reach = []
    for i, coef in enumerate([1e-5, 3e-6, 1e-6]):
        e = find(allruns, "engine", "ki_lab", coef)
        cands = [r for r in allruns if r["kind"] == "cupp" and r["circuit"] == "ki_lab" and abs(r["coef"] / coef - 1) < 1e-9 and r["gpus"] == 16 and not r["tuned"]]
        d = next((r for r in cands if ok(r)), cands[-1] if cands else None)
        reach.append((coef, e, d))
        if ok(e):
            ax.bar(i - w / 2, e["wall"], width=w * 0.94, color=OURS)
            ax.text(i - w / 2, e["wall"] * 1.1, secs(e["wall"]), ha="center", va="bottom", fontsize=11.5)
        elif e is not None and e["exit"] == 124:
            ax.text(i - w / 2, 6.3, "over 25 minutes", ha="center", va="bottom", fontsize=10.5, rotation=90, color=OURS)
        if ok(d):
            ax.bar(i + w / 2, d["wall"], width=w * 0.94, color=BASE)
            ax.text(i + w / 2, d["wall"] * 1.1, secs(d["wall"]) + (" (%s)" % times(d["wall"], e["wall"]) if ok(e) else ""), ha="center", va="bottom", fontsize=11.5)
        elif d is not None:
            ax.text(i + w / 2, 6.3, "out of memory" if d["fatal"] else "time limit", ha="center", va="bottom", fontsize=10.5, rotation=90, color=BASE)
    ax.set_yscale("log")
    ax.set_ylim(5.5, 700)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1e-5", "3e-6", "1e-6"])
    ax.set_xlabel("Cutoff (kicked Ising, 16 GPUs)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    clean(ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=BASE)], ["This work", "cuPauliProp distributed"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_gpus.png")
    return rows, reach


# ---------------------------------------------------------------------------------------
# exact-value check: 5x6 square-lattice TFIM, 30 qubits (logs/pp_e2b-59475514.out,
# logs/inter_run2.out, logs/inter_run3.out; exact values from analysis/exact_sv.py)
# ---------------------------------------------------------------------------------------
EXACT = {4: 0.802867136639, 6: 0.790133273060}
PP = {4: {1e-3: 0.78343, 1e-4: 0.79018, 1e-5: 0.79905, 1e-6: 0.80155, 1e-7: 0.80280},
      6: {1e-3: 0.72858, 1e-4: 0.74777, 1e-5: 0.78088, 3e-6: 0.77863, 1e-6: 0.78341}}


def accuracy():
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for steps, col, mk in ((4, OURS, "o"), (6, "#0E5A52", "s")):
        cs = sorted(PP[steps], reverse=True)
        err = [abs(PP[steps][c] - EXACT[steps]) for c in cs]
        ax.plot(cs, err, color=col, marker=mk, markersize=8, linewidth=2.2, label=f"{steps} steps")
        ax.text(cs[-1] / 1.25, err[-1], f"{err[-1]:.5f}" if err[-1] < 1e-3 else f"{err[-1]:.4f}", ha="left", va="center", fontsize=11.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-3, 2.2e-8)
    ax.set_ylim(3e-5, 0.2)
    ax.set_xlabel("Cutoff (smaller keeps more terms)")
    ax.set_ylabel("Error against the Exact Value")
    ax.grid(color=GRID, linewidth=1.0)
    ax.tick_params(length=0)
    ax.legend(loc="lower left", fontsize=12.5)
    save(fig, "eval_accuracy.png")


# ---------------------------------------------------------------------------------------
# where the time goes, 16 GPUs
# ---------------------------------------------------------------------------------------
def breakdown(run4):
    sets = [("ki_lab", 1e-5, "Kicked Ising, 1e-5"), ("tfim_hh", 1e-5, "TFIM heavy-hex, 1e-5"),
            ("ki_noisy", 1e-5, "Noisy Kicked Ising, 1e-5"), ("ki_lab", 3e-6, "Kicked Ising, 3e-6")]
    parts = [("Gates on GPU", "operations", OURS), ("Exchange Between Nodes", "exchange", "#C0432F"),
             ("GPU to Host", "gpu->host", "#F5B553"), ("Host to GPU", "host->gpu", "#F9D79A"),
             ("Partition", "partition", "#7FC8BE"), ("Idle GPU", "idle", "#B9BDC4"), ("Other", "other", "#E3E4E8")]
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    out = []
    ylabels = []
    for i, (c, coef, label) in enumerate(sets):
        r = find(run4, "engine", c, coef)
        if not ok(r) or "workers" not in r["parts"]:
            continue
        p, g = r["parts"], r["parts"]["workers"]
        share = {k: p.get(k, 0.0) / g / r["wall"] for k in ("operations", "gpu->host", "host->gpu", "partition", "idle")}
        share["exchange"] = p.get("exchange", 0.0) / r["wall"]
        share["other"] = max(0.0, 1.0 - sum(share.values()))
        out.append((label, r["wall"], share))
        left = 0.0
        for name, key, col in parts:
            ax.barh(i, share[key] * 100, left=left, color=col, height=0.62, label=name if i == 0 else None)
            if share[key] >= 0.06:
                ax.text(left + share[key] * 50, i, f"{share[key] * 100:.0f}%", ha="center", va="center", fontsize=11.5,
                        color="white" if key in ("operations", "exchange") else INK)
            left += share[key] * 100
        ylabels.append(f"{label}\n({secs(r['wall'])} s)")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the Run Time on 16 GPUs (%)")
    ax.tick_params(length=0)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.42, 1.36), fontsize=11.5, columnspacing=1.4, handlelength=1.2)
    save(fig, "eval_breakdown.png")
    return out


def main():
    run4, run5, e3 = parse("inter_run4.out"), parse("inter_run5.out"), parse("pp_e3-59475516.out")
    run6 = parse("inter_run6.out")
    e2a, e2b, run2, run3 = parse("pp_e2a-59475513.out"), parse("pp_e2b-59475514.out"), parse("inter_run2.out"), parse("inter_run3.out")
    allruns = e2a + e2b + e3 + run2 + run3 + run4 + run5 + run6
    background()
    table = speed(run5 + run6, e3 + run3 + run4 + run5 + run6)
    rows, reach = gpus_and_reach(run4, run5, run4 + run5 + run6)
    accuracy()
    parts = breakdown(run4)
    if "--print" in sys.argv:
        print("\n# speed, 16 GPUs: setting | engine | tuned | default | value")
        for label, coef, e, t, d in table:
            f = lambda r: ("-" if r is None else (("%.2f" % r["wall"]) + (" [%s%s]" % (r["log"][:10], " k=" + r["opt"]["--cupp-dedup-every"] if "--cupp-dedup-every" in r["opt"] else "")) if ok(r) else ("OOM" if r["fatal"] else "exit %s" % r["exit"])))
            v = e["value"] if ok(e) else None
            agree = [r["value"] == v for r in (t, d) if ok(r)]
            print(f"{label:22s} {coef:g} | {f(e)} | {f(t)} | {f(d)} | {v} same={agree}")
        print("\n# GPUs: engine | cuPauliProp")
        for g, e, d in rows:
            print(g, secs(e["wall"]) if ok(e) else e and e["exit"], secs(d["wall"]) if ok(d) else (d and ("OOM" if d["fatal"] else d["exit"])))
        print("\n# reach: cutoff | engine | cuPauliProp")
        for coef, e, d in reach:
            print(coef, (secs(e["wall"]), e["value"], e.get("boundary")) if ok(e) else e and e["exit"], secs(d["wall"]) if ok(d) else (d and ("OOM" if d["fatal"] else d["exit"])))
        print("\n# breakdown")
        for label, wall, share in parts:
            print(label, wall, {k: round(v * 100, 1) for k, v in share.items()})


if __name__ == "__main__":
    main()
