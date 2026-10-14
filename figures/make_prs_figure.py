"""SectorQ against paulistrings-rs under the same truncation rule (a truncation after every operation), seven circuits on
16 GPUs: figures/eval_prs.png (README) and figures/deck/d_prs.png (slides).

  ../env/bin/python3 make_prs_figure.py            # both figures
  ../env/bin/python3 make_prs_figure.py --table    # the README table as well

Sources: queue_p/done/p.<mode>.<circuit>.<cutoff>.json (jobs/p_drive.py) and logs/p/sq.e1[.<circuit>].g<GPUs>.<cutoff>.out (jobs/p_sq.sh)."""
import json
import os
import re
import sys

import matplotlib.pyplot as plt

import make_eval_figures as mf

ROOT = os.path.dirname(mf.HERE)
PRS = ("#6B8E23", "D")                      # paulistrings-rs: one color in every figure
LADDER = {"ki": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), "tfim": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"),
          "noise": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"), "tfimsq": ("1e-4", "3e-5", "1e-5", "3e-6"),
          "xxz": ("1e-4", "3e-5", "1e-5", "3e-6"), "qaoa": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"),
          "hea": ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7")}
TITLE = {"ki": "Kicked Ising", "tfim": "TFIM Heavy-Hex", "noise": "Noisy Kicked Ising", "tfimsq": "TFIM 11×11", "xxz": "XXZ 8×8",
         "qaoa": "QAOA", "hea": "Ansatz"}
MODE = {1: "g1", 4: "g4", 16: "mg16"}


def prs(c, g, x):
    try:
        return json.load(open(os.path.join(ROOT, "queue_p", "done", f"p.{MODE[g]}.{c}.{x}.json")))
    except (OSError, ValueError):
        return None


def ours(c, g, x):
    name = f"sq.e1.g{g}.{x}.out" if c == "ki" else f"sq.e1.{c}.g{g}.{x}.out"
    try:
        t = open(os.path.join(ROOT, "logs", "p", name)).read()
    except OSError:
        return None
    w, e, n = re.search(r"^wall=([\d.]+) s", t, re.M), re.search(r"^expectation=([-\d.e]+)", t, re.M), re.search(r"max_terms_at_window_boundary=(\d+)", t)
    return dict(wall=float(w.group(1)), expectation=float(e.group(1)), terms=int(n.group(1)) if n else 0) if w and e and "exit=0" in t else None


def okp(r):
    return r is not None and r.get("status") == "ok"


def draw(axes, size, g=16):
    for j, c in enumerate(LADDER):
        ax, cuts = axes[j], LADDER[c]
        a = [ours(c, g, x) for x in cuts]
        b = [prs(c, g, x) for x in cuts]
        n = max(i + 1 for i in range(len(cuts)) if a[i] or okp(b[i]))
        for recs, color, marker, name, lw in ((a, mf.SQ["ours"][0], "o", "SectorQ", 3.6), ([r if okp(r) else None for r in b], PRS[0], PRS[1], "paulistrings-rs", 2.8)):
            pts = [(i, r["wall"]) for i, r in enumerate(recs[:n]) if r]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, linewidth=lw, markersize=8.5, label=name if j == 0 else None,
                    zorder=3 if name == "SectorQ" else 2)
        for i in range(n):
            if a[i] and okp(b[i]):
                ax.annotate(mf.times(b[i]["wall"], a[i]["wall"]), (i, b[i]["wall"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=size - 3.5)
        mf.sq_axes(ax, None, None, size=size)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{x}\n({mf.count(r['terms']).replace(' ', '')})" if r else x for x, r in zip(cuts[:n], a[:n])], fontsize=size - 4.5)
        ax.set_xlim(-0.45, n - 0.55)
        ax.set_ylim(0.3, 3000)
        ax.set_title(f"({'abcdefg'[j]}) {TITLE[c]}", fontsize=size - 1.5, pad=6)


def table():
    """Markdown rows for the README: 16 GPUs per circuit, then the ranges on 1 and 4 GPUs."""
    word = {"oom": "메모리 부족", "tl": "40분 초과", "fail": "내부 오류"}
    print("| 회로 | cutoff | 구간 경계의 항 수 | SectorQ | paulistrings-rs | 배수 | 기대값의 차이 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for c, cuts in LADDER.items():
        for x in cuts:
            a, b = ours(c, 16, x), prs(c, 16, x)
            if a is None and not okp(b):
                continue
            pb = f"{b['wall']:.3g}초" if okp(b) else word.get((b or {}).get("status"), "–")
            fac = f"{b['wall'] / a['wall']:.1f}배" if a and okp(b) else "–"
            dif = f"{abs(a['expectation'] - b['expectation']):.0e}" if a and okp(b) else "–"
            print(f"| {TITLE[c]} | {x} | {a['terms'] / 1e8:.3g}억 | {a['wall']:.3g}초 | {pb} | {fac} | {dif} |" if a else f"| {TITLE[c]} | {x} | – | – | {pb} | – | – |")


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    for name, figsize, size, wspace, hspace in (("eval_prs.png", (19.5, 8.6), 15, 0.16, 0.62), ("deck/d_prs.png", (17.44, 5.8), 13, 0.14, 0.75)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 12, wspace=wspace * 6, hspace=hspace)
        axes = [fig.add_subplot(gs[0, k * 4:(k + 1) * 4]) for k in range(3)] + [fig.add_subplot(gs[1, k * 3:(k + 1) * 3]) for k in range(4)]
        draw(axes, size)
        axes[0].set_ylabel("Simulation Time (s)", fontsize=size, fontweight="bold")
        axes[3].set_ylabel("Simulation Time (s)", fontsize=size, fontweight="bold")
        fig.supxlabel(mf.xl_cut(), fontsize=size, fontweight="bold", y=0.0 if name.startswith("eval") else -0.04)
        handles, names = axes[0].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="none")); names.append("N×: paulistrings-rs time / SectorQ time, 16 GPUs, same truncation rule")
        fig.legend(handles, names, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02 if name.startswith("eval") else 1.06), fontsize=size - 1.5, frameon=False,
                   columnspacing=1.6, handlelength=2.4)
        if name.startswith("deck"):
            fig.savefig(os.path.join(mf.HERE, name), dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.08)
            print("wrote", name)
        else:
            mf.save(fig, name)
