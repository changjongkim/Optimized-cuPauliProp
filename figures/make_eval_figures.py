#!/usr/bin/env python3
"""Background and evaluation figures of the README.

Run times, expectation values and time breakdowns are read from the run logs in ../logs
(blocks that start with "===== engine|cupp nodes=... args: ..."). The numbers of the
background figure and of the exact-value check come from earlier logs with other formats
and are listed below with their source.
"""
import glob
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

BIG_LOG = "pp_e10-59483161.out"
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
        m = re.search(r"windows running=([\d.]+), waiting for the slowest node=([\d.]+)", line)
        if m:
            cur["parts"]["running"], cur["parts"]["waiting"] = float(m.group(1)), float(m.group(2))
        m = re.match(r"solve_epoch=([\d.]+)\.\.([\d.]+)", line)
        if m:
            cur["epoch"] = (float(m.group(1)), float(m.group(2)))
        if line.startswith("cupp phases"):
            cur["phases"] = {k: (float(a), float(b)) for k, a, b in re.findall(r"(\w+)=([\d.]+) \(([\d.]+)\)", line)}
        m = re.match(r"cupp mpi, seconds per rank: collective=([\d.]+) point_to_point=([\d.]+) total=([\d.]+)", line)
        if m:
            cur["mpi"] = {"collective": float(m.group(1)), "p2p": float(m.group(2)), "total": float(m.group(3))}
        m = re.match(r"cupp mpi (\w+): calls/rank=(\d+) \(([\d.]+) per operation\) seconds/rank=([\d.]+) GB/rank=([\d.]+)", line)
        if m:
            cur.setdefault("mpi_kinds", {})[m.group(1)] = dict(calls=int(m.group(2)), per_op=float(m.group(3)), seconds=float(m.group(4)), gb=float(m.group(5)))
        m = re.match(r"max_terms_at_window_boundary=(\d+)", line)
        if m:
            cur["boundary"] = int(m.group(1))
        m = re.match(r"integrity: (\d+) term checks", line)
        if m:
            cur["checks"] = int(m.group(1))
        m = re.match(r"exit=(\d+)", line)
        if m:
            cur["exit"] = int(m.group(1))
    return runs


# options that mark a run as a variant of the engine (a default switched off or changed)
VARIANT = ("--early-per-node", "--share-above", "--defer-dedup", "--min-bucket", "--growth-cap", "--busy-window", "--min-bucket-early")


def ok(r):
    return r is not None and r["wall"] is not None and r["exit"] == 0 and not r["fatal"]


def find(runs, kind, circuit, coef, gpus=16, tuned=False, last=True, **opt):
    hits = [r for r in runs if r["kind"] == kind and r["circuit"] == circuit and abs(r["coef"] / coef - 1) < 1e-9
            and r["gpus"] == gpus and r["tuned"] == tuned and all(r["opt"].get(k) == v for k, v in opt.items())
            and not any(k in r["opt"] for k in VARIANT)]
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
LARGE3 = [("ki_lab", 1e-5, "Kicked\nIsing", {}), ("tfim_hh", 1e-5, "TFIM\nheavy-hex", {}), ("ki_noisy", 1e-5, "Noisy\nKicked\nIsing", {})]
LARGE = [("ki_lab", 1e-5, "Kicked\nIsing", {}), ("tfim_hh", 1e-5, "TFIM\nheavy-hex", {}),
         ("tfim_sq", 1e-5, "TFIM\n11×11", {"--lx": "11"}), ("ki_noisy", 1e-5, "Noisy\nKicked\nIsing", {})]


def triple(runs_first, runs_all, c, coef, opt, gpus=16):
    """(engine, tuned baseline, default baseline) for one setting; prefers runs on the same nodes."""
    out = []
    for kind, tuned in (("engine", False), ("cupp", True), ("cupp", False)):
        r = find(runs_first, kind, c, coef, gpus=gpus, tuned=tuned, **opt)
        if not ok(r):
            r2 = find(runs_all, kind, c, coef, gpus=gpus, tuned=tuned, **opt)
            r = r2 if (ok(r2) or r is None) else r
        out.append(r)
    return out


def speed(same16, all16, big):
    """Three panels: fits one GPU (16 GPUs); beyond one GPU on 16 and on 32 GPUs of the same nodes."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.9), gridspec_kw={"width_ratios": [7, 3.3, 3.3], "wspace": 0.13})
    table = []
    panels = ((axes[0], MEDIUM, "Fits One GPU, 16 GPUs", same16, all16, 16),
              (axes[1], LARGE3, "Beyond One GPU, 16 GPUs", big, big + all16, 16),
              (axes[2], LARGE3, "Beyond One GPU, 32 GPUs", big, big, 32))
    for ax, sets, title, first, pool, g in panels:
        w = 0.25
        for i, (c, coef, label, opt) in enumerate(sets):
            e, t, d = triple(first, pool, c, coef, opt, gpus=g)
            table.append((label.replace("\n", " "), coef, g, e, t, d))
            for j, (r, col) in enumerate(((e, OURS), (t, TUNED), (d, BASE))):
                x = i + (j - 1) * w
                if ok(r):
                    ax.bar(x, r["wall"], width=w * 0.94, color=col)
                    if j > 0 and ok(e):   # multiples: the tuned label grows to the left, the default label to the right
                        ax.text(x + (w * 0.47 if j == 1 else -w * 0.47), r["wall"] * 1.12, times(r["wall"], e["wall"]),
                                ha="right" if j == 1 else "left", va="bottom", fontsize=10.5)
                elif r is not None:
                    ax.text(x, 0.075, "out of\nmemory" if r["fatal"] else "time\nlimit", ha="center", va="bottom", fontsize=9.5, rotation=90, color=col)
        ax.set_yscale("log")
        ax.set_ylim(0.06, 400)
        ax.set_xlim(-0.6, len(sets) - 0.4)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s[2] for s in sets], fontsize=11.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=13.5, pad=8)
        clean(ax)
    axes[0].set_ylabel("Time (s)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, TUNED, BASE)]
    fig.legend(handles, ["SectorQ", "cuPauliProp distributed, dedup deferred", "cuPauliProp distributed"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_speed.png")
    return table


# ---------------------------------------------------------------------------------------
# number of GPUs and reach (kicked Ising)
# ---------------------------------------------------------------------------------------
def gpus_and_reach(run4, run5, allruns, big):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 3.8), gridspec_kw={"width_ratios": [6.4, 3.0], "wspace": 0.2})
    ax = axes[0]
    counts = [1, 2, 4, 8, 16, 32]
    rows = []
    w = 0.26
    for i, g in enumerate(counts):
        e = find(run5 + big, "engine", "ki_lab", 1e-5, gpus=g)                       # one GPU: batch 5; the rest: final job
        d = find(run4 + big, "cupp", "ki_lab", 1e-5, gpus=g)
        t = find(run5 + big, "cupp", "ki_lab", 1e-5, gpus=g, tuned=True)
        rows.append((g, e, t, d))
        for j, (r, col) in enumerate(((e, OURS), (t, TUNED), (d, BASE))):
            x = i + (j - 1) * w
            if ok(r):
                ax.bar(x, r["wall"], width=w * 0.94, color=col)
                label = secs(r["wall"]) if j == 0 else times(r["wall"], e["wall"])
                # the engine's time and the tuned multiple grow to the left, the default multiple to the right
                ax.text(x + (-w * 0.47 if j == 2 else w * 0.47), r["wall"] * 1.1, label,
                        ha="left" if j == 2 else "right", va="bottom", fontsize=10.5)
            elif r is not None or (g == 1 and j == 2):   # one GPU, default: fails at 96 million terms (logs/pp_prof2-59470887.out)
                ax.text(x, 6.3, "out of memory", ha="center", va="bottom", fontsize=9.5, rotation=90, color=col)
    ax.set_yscale("log")
    ax.set_ylim(5.5, 700)
    ax.set_xlim(-0.6, len(counts) - 0.4)
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
        w = 0.38
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
            ax.text(i + w / 2, d["wall"] * 1.1, times(d["wall"], e["wall"]) if ok(e) else secs(d["wall"]), ha="center", va="bottom", fontsize=11.5)
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
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, TUNED, BASE)], ["SectorQ", "cuPauliProp distributed, dedup deferred", "cuPauliProp distributed"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_gpus.png")
    return rows, reach


# ---------------------------------------------------------------------------------------
# scaling on 4-128 GPUs, 40 GB nodes at 36 GB per GPU: engine from job set e13 (default settings),
# cuPauliProp from job set e12
# ---------------------------------------------------------------------------------------
SCALE3 = [("ki_lab", "Kicked Ising", 3e-6), ("tfim_hh", "TFIM heavy-hex", 5e-6), ("ki_noisy", "Noisy Kicked Ising", 5e-6)]
WEAK_CUTOFF = {4: 1e-5, 8: 7e-6, 16: 4.8e-6, 32: 3.4e-6, 64: 2.3e-6, 128: 1.6e-6}


def cut(v):
    """A cutoff as 3e-6 instead of 3e-06."""
    return f"{v:g}".replace("e-0", "e-")


def many(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(LOGS, pattern))):
        out += parse(os.path.basename(path))
    return out


def default_run(runs, circuit, coef, gpus):
    """The engine with its default sector limit and chunk size."""
    hits = [r for r in runs if r["kind"] == "engine" and r["circuit"] == circuit and abs(r["coef"] / coef - 1) < 1e-9
            and r["gpus"] == gpus and "--gmax" not in r["opt"] and "--chunk" not in r["opt"] and ok(r)]
    return hits[-1] if hits else None


def scaling(e13):
    counts = [g for g in (4, 8, 16, 32, 64, 128) if default_run(e13, "ki_lab", 3e-6, g)]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.9), gridspec_kw={"wspace": 0.2, "hspace": 0.5})
    strong, weak = [], []
    for j, (c, title, coef) in enumerate(SCALE3):
        # strong scaling: the same problem on every GPU count
        ax = axes[0][j]
        rs = [default_run(e13, c, coef, g) for g in counts]
        t0, g0 = rs[0]["wall"], counts[0]
        for i, (g, r) in enumerate(zip(counts, rs)):
            if r is None:
                continue
            ax.bar(i, r["wall"], width=0.62, color=OURS)
            ideal = t0 * g0 / g
            ax.plot([i - 0.38, i + 0.38], [ideal, ideal], color=INK, linewidth=1.6)
            ax.text(i, r["wall"] * 1.12, secs(r["wall"]) + " s" if i == 0 else times(t0, r["wall"]), ha="center", va="bottom", fontsize=12)
            strong.append((title, coef, g, r))
        ax.set_yscale("log")
        ax.set_ylim(t0 / (counts[-1] / g0) / 2.2, t0 * 2.6)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(g) for g in counts])
        ax.set_xlabel(f"GPUs (cutoff {cut(coef)})")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            ax.set_ylabel("Time (s)")
        clean(ax)
        # weak scaling: the cutoff is lowered with the GPU count, the rate is terms per second and GPU
        ax = axes[1][j]
        rs = [default_run(e13, c, WEAK_CUTOFF[g], g) for g in counts]
        rate = [None if r is None else r["checks"] / r["wall"] / g for g, r in zip(counts, rs)]
        for i, (g, r) in enumerate(zip(counts, rs)):
            if r is None:
                continue
            pct = 100 * rate[i] / rate[0]
            ax.bar(i, pct, width=0.62, color=OURS)
            ax.text(i, 103, f"{pct:.0f}%", ha="center", va="bottom", fontsize=12 if len(counts) <= 5 else 11)      # all labels in one row above the ideal line
            weak.append((title, g, r, rate[i], pct))
        ax.axhline(100, color=INK, linewidth=1.6)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{g}\n{cut(WEAK_CUTOFF[g])}" for g in counts], fontsize=11 if len(counts) <= 5 else 9.5)
        ax.set_xlabel("GPUs and Cutoff of the Run")
        if j == 0:
            ax.set_ylabel("Terms per Second per GPU\n(% of 4 GPUs)")
        clean(ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Line2D([0], [0], color=INK, linewidth=1.6)], ["SectorQ", "Ideal scaling from 4 GPUs"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    for ax, label in ((axes[0][0], "Strong Scaling"), (axes[1][0], "Weak Scaling")):
        box = ax.get_position()
        fig.text(0.028, (box.y0 + box.y1) / 2, label, fontsize=13.5, fontweight="bold", rotation=90, ha="center", va="center")
    save(fig, "eval_scaling.png")
    return strong, weak


def scaling_baselines(e13, e12):
    """Cutoff 1e-5 on 16 GPUs and more: the engine against both cuPauliProp modes on the same kind of node."""
    counts = [g for g in (16, 32, 64, 128) if default_run(e13, "ki_lab", 1e-5, g)]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8), gridspec_kw={"wspace": 0.16})
    rows = []
    w = 0.27
    for j, (c, title, _) in enumerate(SCALE3):
        ax = axes[j]
        for i, g in enumerate(counts):
            e = default_run(e13, c, 1e-5, g)
            t = find(e12, "cupp", c, 1e-5, gpus=g, tuned=True)
            d = find(e12, "cupp", c, 1e-5, gpus=g)
            rows.append((title, g, e, t, d))
            for k, (r, col) in enumerate(((e, OURS), (t, TUNED), (d, BASE))):
                if not ok(r):
                    continue
                x = i + (k - 1) * w
                ax.bar(x, r["wall"], width=w * 0.94, color=col)
                # the engine's time and the tuned multiple grow to the left, the default multiple to the right
                ax.text(x + (-w * 0.47 if k == 2 else w * 0.47), r["wall"] * 1.1, secs(r["wall"]) if k == 0 else times(r["wall"], e["wall"]),
                        ha="left" if k == 2 else "right", va="bottom", fontsize=10.5)
        ax.set_yscale("log")
        ax.set_ylim(4.5, 220)
        ax.set_xlim(-0.6, len(counts) - 0.4)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(g) for g in counts])
        ax.set_xlabel("GPUs (cutoff 1e-5)")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            ax.set_ylabel("Time (s)")
        else:
            ax.set_yticklabels([])
        clean(ax)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, TUNED, BASE)], ["SectorQ", "cuPauliProp distributed, dedup deferred", "cuPauliProp distributed"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), fontsize=12.5, columnspacing=2.2, handlelength=1.4)
    save(fig, "eval_scaling_base.png")
    return rows


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
def breakdown(big):
    sets = [("ki_lab", 1e-5, 16, "Kicked Ising, 16 GPUs"), ("ki_lab", 1e-5, 32, "Kicked Ising, 32 GPUs"),
            ("tfim_hh", 1e-5, 32, "TFIM heavy-hex, 32 GPUs"), ("ki_noisy", 1e-5, 32, "Noisy Kicked Ising, 32 GPUs")]
    parts = [("Gates on GPU", "operations", OURS), ("Exchange Between Nodes", "exchange", "#C0432F"),
             ("GPU-Host Copies", "copies", "#F5B553"), ("Partition", "partition", "#7FC8BE"),
             ("Idle GPU", "idle", "#B9BDC4"), ("Waiting for Other Nodes", "waiting", "#8E96A3"), ("Planning and Other", "other", "#E3E4E8")]
    fig, ax = plt.subplots(figsize=(10.8, 2.9))
    out, ylabels = [], []
    for c, coef, g, label in sets:
        r = find(big, "engine", c, coef, gpus=g)
        if not ok(r) or "workers" not in r["parts"]:
            continue
        p, n = r["parts"], r["parts"]["workers"]
        share = {"operations": p.get("operations", 0.0) / n, "copies": (p.get("gpu->host", 0.0) + p.get("host->gpu", 0.0)) / n,
                 "partition": p.get("partition", 0.0) / n, "idle": p.get("idle", 0.0) / n,
                 "waiting": p.get("waiting", 0.0), "exchange": p.get("exchange", 0.0)}
        share = {k: v / r["wall"] for k, v in share.items()}
        share["other"] = max(0.0, 1.0 - sum(share.values()))
        out.append((label, r["wall"], share))
        left, i = 0.0, len(ylabels)
        for name, key, col in parts:
            ax.barh(i, share[key] * 100, left=left, color=col, height=0.62, label=name if i == 0 else None)
            if share[key] >= 0.06:
                ax.text(left + share[key] * 50, i, f"{share[key] * 100:.0f}%", ha="center", va="center", fontsize=11.5,
                        color="white" if key in ("operations", "exchange", "waiting") else INK)
            left += share[key] * 100
        ylabels.append(f"{label}\n({secs(r['wall'])} s)")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the Run Time (%)")
    ax.tick_params(length=0)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.42, 1.36), fontsize=11.5, columnspacing=1.4, handlelength=1.2)
    save(fig, "eval_breakdown.png")
    return out


# ---------------------------------------------------------------------------------------
# where the time goes on 16 GPUs: SectorQ next to cuPauliProp's distributed mode (logs/u7, jobs/u7_breakdown.sh)
# ---------------------------------------------------------------------------------------
def gpu_busy(prefix, window):
    """Mean GPU utilization (%) of all GPUs of all nodes inside the timed part of a run (nvidia-smi samples)."""
    if not window:
        return None
    vals = []
    for path in glob.glob(prefix + ".*.csv"):
        for line in open(path, errors="replace"):
            f = [x.strip() for x in line.split(",")]
            try:
                ts = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()   # local time, as on the nodes
                if window[0] <= ts <= window[1]:
                    vals.append(float(f[2]))
            except (ValueError, IndexError):
                continue
    return sum(vals) / len(vals) if vals else None


def breakdown_base(folder="u7"):
    """Stacked bars of the run time on 16 GPUs: SectorQ, cuPauliProp distributed (default), and cuPauliProp with
    the deduplication deferred to the truncations. cuPauliProp: seconds per rank from the profiling plugin
    (every MPI call of the library) and from timers around the library calls; the time inside the library calls
    and outside MPI is split by the sampled GPU utilization into the part with a busy GPU and the rest."""
    sets = [("ki.1e-4", "Kicked Ising, Cutoff 1e-4"), ("ki.1e-5", "Kicked Ising, Cutoff 1e-5"),
            ("tfim.1e-5", "TFIM Heavy-Hex, Cutoff 1e-5"), ("noise.1e-5", "Noisy Kicked Ising, Cutoff 1e-5")]
    parts = [("Gates on GPU", "gpu", OURS), ("Library on the Host, GPU Idle", "host", "#8E96A3"),
             ("Collective Calls Among All GPUs", "collective", "#5B3A8C"), ("Moving Terms Between GPUs", "moving", "#C0432F"),
             ("GPU-Host Copies and Partition", "copies", "#F5B553"), ("Idle or Waiting GPU", "idle", "#CFD3D9"),
             ("Other", "other", "#ECEDEF")]

    def one(kind, name):
        runs = parse(os.path.join(folder, f"{kind}.{name}.out"))
        r = runs[-1] if runs else None
        if not ok(r):
            return None
        wall = r["wall"]
        busy = gpu_busy(os.path.join(LOGS, folder, f"smi.{kind}.{name}"), r.get("epoch"))
        if kind == "eng":
            # time inside the single-GPU library calls of the workers: the GPU works for all of it (the sampled
            # utilization agrees), so it is not split
            p, n = r["parts"], r["parts"]["workers"]
            sec = {"gpu": p.get("operations", 0.0) / n, "host": 0.0, "collective": 0.0, "moving": p.get("exchange", 0.0),
                   "copies": (p.get("gpu->host", 0.0) + p.get("host->gpu", 0.0) + p.get("partition", 0.0)) / n,
                   "idle": p.get("idle", 0.0) / n + p.get("waiting", 0.0)}
        else:
            if "phases" not in r or "mpi" not in r:
                return None
            inside = sum(a for a, _ in r["phases"].values())
            local = inside - sum(b for _, b in r["phases"].values())      # inside the library calls, outside MPI
            gpu = min(local, (busy or 0.0) / 100.0 * wall)
            sec = {"gpu": gpu, "host": local - gpu, "collective": r["mpi"]["collective"], "moving": r["mpi"]["p2p"],
                   "copies": 0.0, "idle": 0.0}
        sec["other"] = max(0.0, wall - sum(sec.values()))
        return dict(wall=wall, sec=sec, share={k: v / wall for k, v in sec.items()}, busy=busy, run=r)

    rows = []           # (group title or None, label, data)
    for name, title in sets:
        group = [("SectorQ", one("eng", name)), ("cuPauliProp", one("cupp", name)),
                 ("cuPauliProp, Merge Deferred", one("cupp", name + ".defer"))]
        group = [(lab, d) for lab, d in group if d]
        for j, (lab, d) in enumerate(group):
            rows.append((title if j == 0 else None, lab, d))
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(10.8, 0.52 * len(rows) + 0.5 * len(sets) + 1.3))
    y, ticks, labels, out, seen = 0.0, [], [], [], set()
    for title, lab, d in rows:
        if title:
            y += 0.75 if ticks else 0.0
            ax.text(0, y - 0.62, title, ha="left", va="center", fontsize=12.5, fontweight="bold")
        left = 0.0
        for pname, key, col in parts:
            v = d["share"][key] * 100
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=col, height=0.62, label=pname if pname not in seen else None)
            seen.add(pname)
            if v >= 6:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=11.5,
                        color="white" if key in ("gpu", "host", "collective", "moving") else INK)
            left += v
        ticks.append(y)
        labels.append(f"{lab} ({secs(d['wall'])} s)")
        out.append((title, lab, d))
        y += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(ticks[-1] + 0.55, ticks[0] - 1.0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the Run Time (%)")
    ax.tick_params(length=0)
    handles, names = ax.get_legend_handles_labels()
    order = [names.index(pn) for pn, _, _ in parts if pn in names]
    ax.legend([handles[i] for i in order], [names[i] for i in order], loc="lower center", ncol=4, bbox_to_anchor=(0.40, 1.0),
              fontsize=11.5, columnspacing=1.3, handlelength=1.2)
    save(fig, "eval_breakdown_base.png")
    return out


def main():
    if "--base-only" in sys.argv:
        for title, lab, d in breakdown_base():
            r = d["run"]
            print(f"{title or '':32s} {lab:28s} {d['wall']:7.2f} s  " + "  ".join(f"{k}={v * 100:.0f}%" for k, v in d["share"].items())
                  + (f"  gpu_busy={d['busy']:.0f}%" if d["busy"] is not None else ""))
            for k, v in sorted(r.get("mpi_kinds", {}).items()):
                print(f"{'':62s} mpi {k}: {v['per_op']:.2f} calls per operation, {v['seconds']:.2f} s, {v['gb']:.2f} GB per rank")
            if "phases" in r:
                print(f"{'':62s} phases: " + ", ".join(f"{k} {a:.2f} s ({b:.2f} in MPI)" for k, (a, b) in r["phases"].items()))
        return
    run4, run5, e3 = parse("inter_run4.out"), parse("inter_run5.out"), parse("pp_e3-59475516.out")
    run6 = parse("inter_run6.out")
    big = parse(BIG_LOG)        # 16 and 32 GPUs of the same eight nodes: engine and both baselines
    e2a, e2b, run2, run3 = parse("pp_e2a-59475513.out"), parse("pp_e2b-59475514.out"), parse("inter_run2.out"), parse("inter_run3.out")
    allruns = e2a + e2b + e3 + run2 + run3 + run4 + run5 + run6
    background()
    table = speed(run5 + run6 + [r for r in big if r["kind"] == "engine"], e3 + run3 + run4 + run5 + run6, big)
    rows, reach = gpus_and_reach(run4, run5, run4 + run5 + run6, big)
    e12, e13 = many("pp_e12_g*-*.out"), many("pp_e13_g*-*.out")
    strong, weak = scaling(e13)
    base = scaling_baselines(e13, e12)
    accuracy()
    parts = breakdown(big)
    breakdown_base()
    if "--print" in sys.argv:
        print("\n# speed, 16 GPUs: setting | engine | tuned | default | value")
        for label, coef, g, e, t, d in table:
            f = lambda r: ("-" if r is None else (("%.2f" % r["wall"]) + (" [%s%s]" % (r["log"][:10], " k=" + r["opt"]["--cupp-dedup-every"] if "--cupp-dedup-every" in r["opt"] else "")) if ok(r) else ("OOM" if r["fatal"] else "exit %s" % r["exit"])))
            v = e["value"] if ok(e) else None
            agree = [r["value"] == v for r in (t, d) if ok(r)]
            print(f"{label:22s} {coef:g} {g:2d} GPUs | {f(e)} | {f(t)} | {f(d)} | {v} same={agree}")
        print("\n# GPUs: engine | cuPauliProp")
        for g, e, t, d in rows:
            f2 = lambda r: ("%.2f [%s]" % (r["wall"], r["log"][:10])) if ok(r) else (r and ("OOM" if r["fatal"] else r["exit"]))
            print(g, f2(e), "| tuned", f2(t), "| default", f2(d))
        print("\n# reach: cutoff | engine | cuPauliProp")
        for coef, e, d in reach:
            print(coef, (secs(e["wall"]), e["value"], e.get("boundary")) if ok(e) else e and e["exit"], secs(d["wall"]) if ok(d) else (d and ("OOM" if d["fatal"] else d["exit"])))
        print("\n# strong scaling: circuit | cutoff | GPUs | time | gate time per GPU | exchange | host GiB")
        for title, coef, g, r in strong:
            print(f"{title:20s} {coef:g} {g:4d} {r['wall']:8.1f} {r['parts']['operations'] / g:8.1f} {r['parts'].get('exchange', 0):6.1f}")
        print("\n# weak scaling: circuit | GPUs | cutoff | time | work (1e9 terms) | rate (1e6 terms/s/GPU) | % of 4 GPUs")
        for title, g, r, rate, pct in weak:
            print(f"{title:20s} {g:4d} {r['coef']:g} {r['wall']:8.1f} {r['checks'] / 1e9:8.2f} {rate / 1e6:6.2f} {pct:5.0f}%")
        print("\n# cutoff 1e-5 on 16 GPUs and more: circuit | GPUs | engine | tuned | default")
        for title, g, e, t, d in base:
            f3 = lambda r: ("%.1f" % r["wall"]) if ok(r) else "-"
            print(f"{title:20s} {g:4d} {f3(e)} | {f3(t)} | {f3(d)}")
        print("\n# breakdown")
        for label, wall, share in parts:
            print(label, wall, {k: round(v * 100, 1) for k, v in share.items()})


if __name__ == "__main__":
    main()
