#!/usr/bin/env python3
"""Explanatory figures of the SectorQ presentation deck (figures/deck/n01 ... n16).

  make_deck_figures.py             writes the 16 figures to figures/deck/
  make_deck_figures.py n03 n11     only the figures whose names start with these

Every number in the figures is a measured value entered by hand below. The coefficients
and Pauli strings inside the schematics (0.31 x X I Z, 0.20 . XZ, ...) are illustrations.

Each figure is checked while it is written: a text that leaves its box or overlaps
another text is reported on stderr as "CHECK <figure>: ...".
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.transforms import Bbox

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "deck")

# one colour per entity, the same in every figure
SECTORQ, CUPP, CUPP_DEFER = "#1f77b4", "#d62728", "#8c564b"
CUPP_ORIGINAL = "#E8A0A0"
ORQA, MONOPROP, SPD = "#800080", "#FFA500", "#008000"
BOX, EDGE = "#F2F4F7", "#8E96A3"
INK, SUB = "#1B1F24", "#5B6470"
BAD, GOOD = "#C0432F", "#1E7B4F"
GRID = "#E3E6EB"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5, "text.color": INK,
    "axes.labelsize": 12.5, "axes.labelcolor": INK, "axes.titlesize": 13.5, "axes.titleweight": "normal",
    "axes.titlepad": 10, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": EDGE, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 11.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 200,          # text is measured at the resolution the files are written with
})


def tint(color, a=0.10):
    """The colour mixed with white (a = share of the colour)."""
    r, g, b = to_rgb(color)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def T(s, size=11.5, weight="normal", color=INK):
    """One line of a stack. s is a string or a list of segments (see rich)."""
    return (s, size, weight, color)


# ---------------------------------------------------------------------------------------
# text with mixed weights, placed by its cap height so that all segments share a baseline
# ---------------------------------------------------------------------------------------
def _width_pt(fig, s, size, weight):
    r = fig.canvas.get_renderer()
    w = []
    for probe in ("x" + s + "x", "xx"):          # the probe keeps leading and trailing spaces
        t = fig.text(0, 0, probe, fontsize=size, fontweight=weight)
        w.append(t.get_window_extent(r).width)
        t.remove()
    return (w[0] - w[1]) * 72.0 / fig.dpi


def rich(ax, x, y, segs, size=11.5, weight="normal", color=INK, ha="center", dy=None, z=6):
    """One line at (x, y); y is the middle of the capital letters unless dy (points) is given.

    segs: a string, or a list whose items are a string or (string, weight[, colour])."""
    if isinstance(segs, str):
        segs = [(segs, weight, color)]
    parts = []
    for seg in segs:
        if isinstance(seg, str):
            seg = (seg,)
        s = seg[0]
        w = seg[1] if len(seg) > 1 else weight
        col = seg[2] if len(seg) > 2 else color
        parts.append((s, w, col, _width_pt(ax.figure, s, size, w)))
    total = sum(p[3] for p in parts)
    x0 = {"left": 0.0, "center": -total / 2.0, "right": -total}[ha]
    if dy is None:
        dy = -0.365 * size
    arts = []
    for s, w, col, width in parts:
        arts.append(ax.annotate(s, xy=(x, y), xytext=(x0, dy), textcoords="offset points", fontsize=size,
                                fontweight=w, color=col, ha="left", va="baseline", zorder=z, annotation_clip=False))
        arts[-1]._line = arts[0]              # the segments of one line touch each other by design
        x0 += width
    return arts


# ---------------------------------------------------------------------------------------
# checks: text inside its box, no two texts on top of each other
# ---------------------------------------------------------------------------------------
def _overlaps(texts, r, name):
    n = 0
    ext = [(t, t.get_window_extent(r)) for t in texts]
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            a, b = ext[i][1], ext[j][1]
            same = getattr(ext[i][0], "_line", 1) is getattr(ext[j][0], "_line", 2)
            e = 4 if same else 1               # pixels
            if a.x0 < b.x1 - e and b.x0 < a.x1 - e and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                n += 1
                print(f"CHECK {name}: texts overlap: {ext[i][0].get_text()!r} and {ext[j][0].get_text()!r}", file=sys.stderr)
    return n


def _chart_texts(fig):
    out = list(fig.texts)
    for leg in fig.legends:
        out += leg.get_texts()
    for ax in fig.axes:
        out += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            out += ax.get_legend().get_texts()
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if tick.label1.get_visible() and lo <= tick.get_loc() <= hi:
                    out.append(tick.label1)
    return [t for t in out if t.get_visible() and t.get_text().strip()]


SIZES = {}


def _write(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    h, w = imread(path).shape[:2]
    SIZES[name] = (w, h)


def save_chart(fig, name):
    fig.canvas.draw()
    _overlaps(_chart_texts(fig), fig.canvas.get_renderer(), name)
    _write(fig, name)


# ---------------------------------------------------------------------------------------
# schematics: a canvas in inches (1 unit = 1 inch) that is cut to its content when saved
# ---------------------------------------------------------------------------------------
class Canvas:
    W, H = 14.0, 8.0

    def __init__(self):
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.axis("off")
        self.fits = []

    def box(self, x, y, w, h, fc=BOX, ec=EDGE, lw=1.3, r=0.07, ls="-", z=2):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
        return (x, y, w, h)

    def rect(self, x, y, w, h, fc="white", ec=EDGE, lw=1.1, z=3):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
        return (x, y, w, h)

    def text(self, x, y, s, size=11.5, weight="normal", color=INK, ha="center", inside=None, z=6):
        arts = rich(self.ax, x, y, s, size=size, weight=weight, color=color, ha=ha, z=z)
        if inside is not None:
            self.fits += [(a, inside) for a in arts]
        return arts

    def stack(self, x, cy, items, ha="center", inside=None, lead=1.45):
        """Lines stacked around cy. An item is T(...) or a number (a gap in inches)."""
        hs = [it if isinstance(it, (int, float)) else it[1] * lead / 72.0 for it in items]
        y = cy + sum(hs) / 2.0
        for it, h in zip(items, hs):
            if not isinstance(it, (int, float)):
                self.text(x, y - h / 2.0, it[0], size=it[1], weight=it[2], color=it[3], ha=ha, inside=inside)
            y -= h

    def arrow(self, x0, y0, x1, y1, color=INK, lw=1.6, both=False, head=14, z=5):
        assert math.hypot(x1 - x0, y1 - y0) >= 0.3, "arrow shorter than 0.3 in"
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if both else "-|>",
                                          mutation_scale=head, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))

    def line(self, xs, ys, color=INK, lw=1.6, ls="-", z=4):
        self.ax.add_line(Line2D(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt"))

    def bits(self, x, y, values, s=0.46, fills=None, size=12.5, weights=None, edges=None):
        """A row of bit cells; returns the centre x of every cell."""
        xs = []
        for i, v in enumerate(values):
            fc = fills[i] if fills else "white"
            ec, lw = edges[i] if edges else (EDGE, 1.1)
            self.rect(x + i * s, y, s, s, fc=fc, ec=ec, lw=lw, z=3 if lw < 1.5 else 4)
            self.text(x + (i + 0.5) * s, y + s / 2.0, str(v), size=size, weight=weights[i] if weights else "normal")
            xs.append(x + (i + 0.5) * s)
        return xs

    def save(self, name):
        fig, ax = self.fig, self.ax
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        dpi = fig.dpi
        for t, (x, y, w, h) in self.fits:           # text inside its box, 0.06 in from the border
            e = t.get_window_extent(r)
            out = max(x + 0.06 - e.x0 / dpi, e.x1 / dpi - (x + w - 0.06), y + 0.06 - e.y0 / dpi, e.y1 / dpi - (y + h - 0.06))
            if out > 0:
                print(f"CHECK {name}: {t.get_text()!r} is {out:.2f} in outside its box", file=sys.stderr)
        _overlaps([t for t in ax.texts if t.get_text().strip()], r, name)
        ext = []
        for a in ax.get_children():
            if isinstance(a, Spine) or a is ax.patch or not a.get_visible():
                continue
            if isinstance(a, Text) and not a.get_text().strip():
                continue
            if isinstance(a, (Text, Patch, Line2D)):
                ext.append(a.get_window_extent(r))
        bb = Bbox.union(ext).transformed(ax.transData.inverted())
        m = 0.03                                    # room for the line width of the outermost borders
        ax.set_xlim(bb.x0 - m, bb.x1 + m)
        ax.set_ylim(bb.y0 - m, bb.y1 + m)
        fig.set_size_inches(bb.width + 2 * m, bb.height + 2 * m)
        _write(fig, name)


# ---------------------------------------------------------------------------------------
# chart helpers
# ---------------------------------------------------------------------------------------
def words(v):
    """22500000 -> '22.5 million'."""
    for unit, word in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if v >= unit:
            return f"{v / unit:g} {word}"
    return f"{v:g}"


def count_label(v, _pos=None):
    return words(v) if v >= 1e6 else f"{int(round(v)):,}"


def plain(v, _pos=None):
    return f"{v:g}"


def log_axis(axis, ticks, fmt=plain, minor=()):
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_locator(FixedLocator(list(minor)))
    axis.set_minor_formatter(NullFormatter())


def above(ax, x, y, s, dy=4, size=11.5, weight="normal", color=INK):
    return ax.annotate(s, xy=(x, y), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom",
                       fontsize=size, fontweight=weight, color=color, annotation_clip=False)


# ---------------------------------------------------------------------------------------
# n01  what is computed
# ---------------------------------------------------------------------------------------
def n01():
    c = Canvas()
    y0, h = 2.3, 2.75
    ty, yc, yt = y0 + h - 0.5, y0 + h / 2.0, y0 + 1.02
    A = c.box(0.0, y0, 2.3, h)
    c.text(1.15, ty, "Quantum circuit", 14, "bold", inside=A)
    c.stack(1.15, yt, [T("127 qubits,", 12.5), T("20 steps of gates", 12.5)], inside=A)
    B = c.box(2.92, y0, 4.5, h)
    c.text(5.17, ty, "Observable as a list of terms", 14, "bold", inside=B)
    c.stack(3.14, yt, [T("•  each term = coefficient × Pauli string", 12.5),
                       T("•  gates are applied from the last to the first", 12.5),
                       T("•  terms with small coefficients are dropped", 12.5)], ha="left", inside=B, lead=1.85)
    C = c.box(8.04, y0, 2.3, h)
    c.text(9.19, ty, "Expectation value", 14, "bold", inside=C)
    c.stack(9.19, yt, [T("one number,", 12.5), T(["e.g. ", ("0.2286", "bold")], 12.5)], inside=C)
    c.arrow(2.38, yc, 2.84, yc)
    c.arrow(7.5, yc, 7.96, yc)
    rows = (("State vector", r"stores $\mathregular{2^{n}}$ amplitudes — about 50 qubits at most"),
            ("Pauli propagation", "stores the kept terms — cost set by the number of terms, not by the qubit count"))
    for i, (name, what) in enumerate(rows):
        y = 1.03 - 1.03 * i
        R = c.box(0.0, y, 10.34, 0.82)
        c.text(0.3, y + 0.41, name, 13, "bold", ha="left", inside=R)
        c.text(2.6, y + 0.41, what, 12.5, ha="left", inside=R)
    c.save("n01_overview")


# ---------------------------------------------------------------------------------------
# n02  more kept terms give a more accurate answer
# ---------------------------------------------------------------------------------------
def n02():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 6.1), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.9, bottom=0.13, wspace=0.31)

    cut = ["1e-4", "1e-5", "3e-6", "1e-6"]
    terms = [22.5e6, 1.4e9, 12.6e9, 100.1e9]
    value = [0.153, 0.229, 0.289, 0.322]
    a.bar(range(4), terms, width=0.56, color=SUB)
    a.set_yscale("log")
    a.set_ylim(1e6, 3e12)
    log_axis(a.yaxis, [1e6, 1e7, 1e8, 1e9, 1e10, 1e11], count_label)
    a.grid(axis="y")
    a.set_xlim(-0.6, 3.6)
    a.set_xticks(range(4), cut)
    a.tick_params(axis="x", length=0)
    for i in range(4):
        above(a, i, terms[i], words(terms[i]), dy=4)
        rich(a, i, terms[i], ["value ", (f"{value[i]:.3f}", "bold")], dy=23)
    rich(a, -0.5, 9e11, "value = expectation value at that cutoff", ha="left", color=SUB)
    a.set_xlabel("Cutoff")
    a.set_ylabel("Terms (largest count)")
    a.set_title("(a) 127-qubit kicked Ising")

    cut2 = ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6", "3e-7", "1e-7"]
    err = [1.9e-2, 5.9e-3, 1.3e-2, 1.0e-2, 3.8e-3, 4.8e-4, 1.3e-3, 3.5e-4, 7.2e-5]
    b.plot(range(9), err, color=SUB, lw=2, marker="o", ms=7, mfc=SUB, mec="white", mew=1.2, clip_on=False)
    b.set_yscale("log")
    b.set_ylim(3e-5, 6e-2)
    log_axis(b.yaxis, [1e-4, 1e-3, 1e-2], plain)
    b.grid(axis="y")
    b.set_xlim(-0.5, 8.5)
    b.set_xticks(range(9), cut2)
    b.annotate("0.019", xy=(0, err[0]), xytext=(-5, 9), textcoords="offset points", ha="left", va="bottom", fontsize=11.5)
    b.annotate("0.000072", xy=(8, err[8]), xytext=(9, -10), textcoords="offset points", ha="right", va="top", fontsize=11.5)
    b.set_xlabel("Cutoff")
    b.set_ylabel("Error against the exact value")
    b.set_title("(b) 30-qubit TFIM, 4 steps (exact value known)")
    save_chart(fig, "n02_accuracy_terms")


# ---------------------------------------------------------------------------------------
# n03  truncating before merging gives a wrong answer
# ---------------------------------------------------------------------------------------
def n03():
    c = Canvas()
    yu, yl, hb = 4.2, 2.25, 1.6               # centres of the upper and the lower row, box height
    for i, (yc, coef) in enumerate(((yu, "0.7e-4"), (yl, "0.6e-4"))):
        G = c.box(0.0, yc - hb / 2, 2.3, hb)
        c.text(1.15, yc + 0.36, f"GPU {i}", 13, "bold", inside=G)
        c.text(1.15, yc - 0.24, [(coef, "bold"), " × X Z Y …"], 12.5, inside=G)
    c.text(1.15, 1.08, ["cutoff = ", ("1e-4", "bold")], 12.5)
    # both pieces go to both paths
    c.line([2.3, 2.62], [yu, yu])
    c.line([2.3, 2.62], [yl, yl])
    c.line([2.62, 2.62], [yl, yu])
    c.arrow(2.62, yu, 3.17, yu)
    c.arrow(2.62, yl, 3.17, yl)

    U = c.box(3.25, yu - hb / 2, 4.3, hb)
    c.text(5.4, yu + 0.45, "Truncate on each GPU", 13, "bold", inside=U)
    c.stack(5.4, yu - 0.22, [T(["0.7e-4 < cutoff  →  ", ("dropped", "normal", BAD)], 12),
                             T(["0.6e-4 < cutoff  →  ", ("dropped", "normal", BAD)], 12)], inside=U, lead=1.6)
    L = c.box(3.25, yl - hb / 2, 4.3, hb)
    c.text(5.4, yl + 0.45, "Merge first, then truncate", 13, "bold", inside=L)
    c.stack(5.4, yl - 0.22, [T(["0.7e-4 + 0.6e-4 = ", ("1.3e-4", "bold")], 12),
                             T(["1.3e-4 ≥ cutoff  →  ", ("kept", "normal", GOOD)], 12)], inside=L, lead=1.6)
    c.arrow(7.63, yu, 8.12, yu)
    c.arrow(7.63, yl, 8.12, yl)
    RU = c.box(8.2, yu - hb / 2, 1.9, hb, fc="white", ec=BAD, lw=1.8)
    c.stack(9.15, yu, [T("✗  wrong", 13.5, "bold", BAD), 0.1, T("term lost", 12)], inside=RU)
    RL = c.box(8.2, yl - hb / 2, 1.9, hb, fc="white", ec=GOOD, lw=1.8)
    c.stack(9.15, yl, [T("✓  correct", 13.5, "bold", GOOD), 0.1, T("term kept", 12)], inside=RL)

    F = c.box(0.0, 0.0, 10.1, 0.7)
    c.text(5.05, 0.35, ["Kicked Ising, 22.5 million terms: computed separately on 4 / 16 GPUs → ",
                        ("0.1675 / 0.189", "bold"), "; correct value ", ("0.1531", "bold")], 11.5, inside=F)
    c.save("n03_merge_before_truncate")


# ---------------------------------------------------------------------------------------
# n04  where existing implementations run
# ---------------------------------------------------------------------------------------
def n04():
    c = Canvas()
    cw, gap = 3.2, 0.3
    cols = [
        ("One device", [(SPD, "white", [T("SPD", 12.5, "bold"), 0.06, T("1 GPU")]),
                        (EDGE, "white", [T("PauliPropagation.jl, ppvm,", 11.5, "bold"), T("Qiskit pauli-prop", 11.5, "bold"),
                                         0.06, T("1 node of CPU cores")])],
         ["limit: memory of one device"]),
        ("Many CPU nodes", [(ORQA, "white", [T("ORQA", 12.5, "bold")]),
                            (MONOPROP, "white", [T("MonoProp", 12.5, "bold")])],
         ["no GPU; communication", "at every branching gate"]),
        ("Many GPUs", [(CUPP, "white", [T("cuPauliProp", 12.5, "bold"), T("distributed mode", 12.5, "bold"), 0.06,
                                        T("all GPUs coordinate"), T("at every gate")]),
                       (SECTORQ, tint(SECTORQ, 0.13), [T("SectorQ", 12.5, "bold"), 0.06,
                                                       T("terms are exchanged only"), T("at window boundaries")])],
         ["this work targets this column"]),
    ]
    for k, (head, boxes, foot) in enumerate(cols):
        x = k * (cw + gap)
        P = c.box(x, 0.85, cw, 4.5)
        c.text(x + cw / 2, 4.98, head, 14, "bold", inside=P)
        for j, (ec, fc, lines) in enumerate(boxes):
            y = 2.85 - 1.8 * j
            E = c.box(x + 0.2, y, cw - 0.4, 1.6, fc=fc, ec=ec, lw=2.0)
            c.stack(x + cw / 2, y + 0.8, lines, inside=E)
        c.stack(x + cw / 2, 0.42, [T(s) for s in foot])
    c.save("n04_positioning")


# ---------------------------------------------------------------------------------------
# n05  how a term is stored and split
# ---------------------------------------------------------------------------------------
def n05():
    c = Canvas()
    s = 0.46
    xx, xz = 3.75, 3.75 + 3 * s + 0.4          # left edge of the x cells and of the z cells
    sel = tint(SECTORQ, 0.22)
    ya, yb = 4.55, 2.65                        # centres of the two upper rows

    A = c.box(0.0, ya - 0.7, 2.7, 1.4)
    c.stack(1.35, ya, [T("Term", 13, "bold"), 0.07, T("0.31 × X I Z", 12.5, "bold"), 0.04, T("coefficient × Pauli string")], inside=A)
    c.arrow(2.78, ya, 3.4, ya)
    cx = c.bits(xx, ya - s / 2, [1, 0, 0], s) + c.bits(xz, ya - s / 2, [0, 0, 1], s, fills=[sel] * 3)
    c.text(xx + 1.5 * s, ya + 0.92, "x bits", 12)
    c.text(xz + 1.5 * s, ya + 0.92, "z bits", 12)
    for i, x in enumerate(cx):
        c.text(x, ya + 0.47, str(i % 3 + 1), 11, color=SUB)
    c.text(xx - 0.12, ya + 0.47, "qubit", 11, color=SUB, ha="right")
    c.text((xx + 3 * s + xz) / 2, ya, "|", 14, color=SUB)
    K = c.box(7.45, ya - 0.7, 2.45, 1.4, fc="white")
    c.stack(8.675, ya, [T("(x, z) of one qubit", 11.5, "normal", SUB), 0.07,
                        T("I = (0,0)    X = (1,0)"), T("Z = (0,1)    Y = (1,1)")], inside=K)

    B = c.box(0.0, yb - 0.7, 2.7, 1.4)
    c.stack(1.35, yb, [T("Row r", 13, "bold"), 0.07, T("parity of the bits"), T("selected by a mask")], inside=B)
    c.arrow(1.35, ya - 0.78, 1.35, yb + 0.78)
    c.arrow(2.78, yb, 3.4, yb)
    c.bits(xx, yb - s / 2, [0, 0, 0], s)
    c.bits(xz, yb - s / 2, [1, 1, 1], s, fills=[sel] * 3)
    c.text((xx + 3 * s + xz) / 2, yb, "|", 14, color=SUB)
    c.text((xx + xz + 3 * s) / 2, yb + 0.5, "mask: 1 = bit selected", 11, color=SUB)
    c.arrow(xz + 3 * s + 0.1, yb, 7.37, yb)
    R = c.box(7.45, yb - 0.7, 2.45, 1.4, fc=tint(SECTORQ, 0.13), ec=SECTORQ, lw=1.8)
    c.stack(8.675, yb, [T("r = 1", 14, "bold"), 0.07, T("selected bits 0, 0, 1"), T("0 ⊕ 0 ⊕ 1 = 1")], inside=R)

    S = c.box(0.0, 0.0, 4.7, 1.25)
    c.stack(2.35, 0.625, [T("Sector", 13, "bold"), 0.07, T("the terms whose rows have the same values")], inside=S)
    W = c.box(5.2, 0.0, 4.7, 1.25)
    c.stack(7.55, 0.625, [T("Window", 13, "bold"), 0.07, T("consecutive gates that keep these rows unchanged")], inside=W)
    c.arrow(1.35, yb - 0.78, 1.35, 1.33)
    c.arrow(4.78, 0.625, 5.12, 0.625)
    c.save("n05_term_bits")


# ---------------------------------------------------------------------------------------
# n06  the three things one gate does
# ---------------------------------------------------------------------------------------
def n06():
    c = Canvas()
    pw, gap, H = 3.1, 0.5, 5.05
    ym = 2.5                                   # centre of the small diagrams

    def panel(k, title, desc):
        x = k * (pw + gap)
        P = c.box(x, 0.0, pw, H)
        c.text(x + pw / 2, H - 0.42, title, 14, "bold", inside=P)
        c.stack(x + pw / 2, H - 1.15, [T(d, 12) for d in desc], inside=P)
        return x, P

    def chip(x, y, w, s, ec=EDGE, color=INK, h=0.52):
        R = c.box(x, y - h / 2, w, h, fc="white", ec=ec, lw=1.2, r=0.05, z=3)
        c.text(x + w / 2, y, s, 12.5, color=color, inside=R)

    x, P = panel(0, "Branch", ["a rotation with generator Q", "turns term P into two terms"])
    chip(x + 0.2, ym, 0.55, "P")
    chip(x + 1.3, ym + 0.52, 1.6, "cos θ · P")
    chip(x + 1.3, ym - 0.52, 1.6, "sin θ · (P ⊕ Q)")
    c.arrow(x + 0.81, ym + 0.07, x + 1.24, ym + 0.46)
    c.arrow(x + 0.81, ym - 0.07, x + 1.24, ym - 0.46)
    c.stack(x + pw / 2, 0.75, [T("example", 12, "normal", SUB), T("Z  →  cos θ · Z + sin θ · Y", 12)], inside=P)

    x, P = panel(1, "Merge", ["two terms with the same Pauli", "string add their coefficients"])
    chip(x + 0.2, ym + 0.52, 1.2, "0.20 · XZ")
    chip(x + 0.2, ym - 0.52, 1.2, "0.15 · XZ")
    c.text(x + 0.8, ym, "+", 14)
    chip(x + 1.8, ym, 1.15, "0.35 · XZ")
    c.arrow(x + 1.46, ym + 0.46, x + 1.76, ym + 0.1)
    c.arrow(x + 1.46, ym - 0.46, x + 1.76, ym - 0.1)
    c.stack(x + pw / 2, 0.75, [T("example", 12, "normal", SUB), T(["0.20 + 0.15 = ", ("0.35", "bold")], 12)], inside=P)

    x, P = panel(2, "Truncate", ["terms with |coefficient|", "≤ cutoff are dropped"])
    for j, (s, keep) in enumerate((("0.35 · XZ", True), ("0.12 · YI", True), ("0.004 · ZX", False))):
        y = ym + 0.64 - 0.64 * j
        chip(x + 0.25, y, 1.45, s, ec=EDGE if keep else BAD)
        c.text(x + 1.88, y, "kept" if keep else "dropped", 12, color=GOOD if keep else BAD, ha="left", inside=P)
        if not keep:                           # crossed out
            c.line([x + 0.33, x + 1.62], [y - 0.2, y + 0.2], color=BAD, lw=1.6, z=7)
            c.line([x + 0.33, x + 1.62], [y + 0.2, y - 0.2], color=BAD, lw=1.6, z=7)
    c.stack(x + pw / 2, 0.75, [T("example", 12, "normal", SUB), T(["cutoff = ", ("0.01", "bold")], 12)], inside=P)

    for k in (0, 1):
        x = k * (pw + gap) + pw
        c.arrow(x + 0.07, H / 2, x + gap - 0.07, H / 2)
    c.save("n06_ops")


# ---------------------------------------------------------------------------------------
# n07  existing distributed execution gives up either communication or memory
# ---------------------------------------------------------------------------------------
def n07():
    c = Canvas()
    y0, h, bw = 1.55, 3.4, 4.2
    yc = y0 + h / 2
    A = c.box(0.0, y0, bw, h, fc="white", ec=CUPP, lw=2.2)
    c.stack(bw / 2, yc, [T("Merge at every gate,", 14, "bold"), T("across all GPUs", 14, "bold"), 0.06,
                         T("gives up communication", 12.5, "normal", BAD), 0.3,
                         T([("22", "bold"), " all-GPU calls per gate on every GPU"], 12.5), 0.16,
                         T("22.5 million terms:", 12.5),
                         T([("2.9 s", "bold"), " on 1 GPU → ", ("71 s", "bold"), " on 16 GPUs"], 12.5)], inside=A)
    x1 = 10.0 - bw
    B = c.box(x1, y0, bw, h, fc="white", ec=CUPP_DEFER, lw=2.2)
    c.stack(x1 + bw / 2, yc, [T("Merge only", 14, "bold"), T("before truncation", 14, "bold"), 0.06,
                              T("gives up memory", 12.5, "normal", BAD), 0.3,
                              T("unmerged copies stay in GPU memory", 12.5), 0.16,
                              T([("2 of 7", "bold"), " circuits do not fit 16 GPUs"], 12.5),
                              T("at the largest setting", 12.5)], inside=B)
    c.arrow(bw + 0.12, yc, x1 - 0.12, yc, both=True)
    c.stack(5.0, yc + 0.52, [T("one must be", 12.5), T("given up", 12.5)])
    S = c.box(0.0, 0.0, 10.0, 1.2, fc=tint(SECTORQ, 0.13), ec=SECTORQ, lw=2.2)
    c.stack(5.0, 0.6, [T("SectorQ: merge at every gate, inside a sector", 14, "bold"), 0.06,
                       T("no terms move between GPUs during a window", 12.5)], inside=S)
    c.save("n07_tradeoff")


# ---------------------------------------------------------------------------------------
# n08  a rotation leaves some parities unchanged
# ---------------------------------------------------------------------------------------
def n08():
    c = Canvas()
    s = 0.5
    xx = 4.05
    xz = xx + 3 * s + 0.45
    xr = xz + 3 * s + 1.45                      # centre of the row-value column
    keep = tint(GOOD, 0.16)
    yp, yq, yn = 3.85, 3.05, 2.05
    c.text(xx + 1.5 * s, 4.95, "x bits", 12)
    c.text(xz + 1.5 * s, 4.95, "z bits", 12)
    c.stack(xr, 4.82, [T("row r ="), T("parity of the z bits")])
    for i in range(3):
        c.text(xx + (i + 0.5) * s, 4.52, str(i + 1), 11, color=SUB)
        c.text(xz + (i + 0.5) * s, 4.52, str(i + 1), 11, color=SUB)
    c.text(xx - 0.15, 4.52, "qubit", 11, color=SUB, ha="right")

    rows = ((yp, "P", ["term", "Z I I"], [0, 0, 0], [1, 0, 0], ["r(P) = ", ("1", "bold", GOOD)]),
            (yq, "Q", ["generator of an X rotation", "on qubit 1:  X I I"], [1, 0, 0], [0, 0, 0], ["r(Q) = ", ("0", "bold")]),
            (yn, "P ⊕ Q", ["new term", "Y I I"], [1, 0, 0], [1, 0, 0], ["r(P ⊕ Q) = ", ("1", "bold", GOOD)]))
    for y, name, desc, xb, zb, rv in rows:
        c.text(0.0, y, name, 13, "bold", ha="left")
        c.stack(1.0, y, [T(d) for d in desc], ha="left")
        changed = name == "P ⊕ Q"
        c.bits(xx, y - s / 2, xb, s, edges=[(INK, 2.2) if changed and i == 0 else (EDGE, 1.1) for i in range(3)],
               weights=["bold" if changed and i == 0 else "normal" for i in range(3)])
        c.bits(xz, y - s / 2, zb, s, fills=[keep if name != "Q" else "white"] * 3)
        c.text((xx + 3 * s + xz) / 2, y, "|", 14, color=SUB)
        c.text(xr, y, rv, 12)
    c.text(xx - 0.3, yq, "⊕", 14)
    c.text(xx - 0.3, yn, "=", 14)
    c.line([xx - 0.5, xr + 0.95], [(yq + yn) / 2, (yq + yn) / 2], color=EDGE, lw=1.2)

    c.arrow(xx + 0.5 * s, yn - 0.72, xx + 0.5 * s, yn - s / 2 - 0.06, lw=1.4, head=12)
    c.text(xx + 0.5 * s, yn - 0.9, "only this x bit changed", 11.5)
    c.text(xz + 1.5 * s + 0.35, yn - 0.9, "z bits unchanged", 11.5, color=GOOD)

    F = c.box(0.0, 0.0, xr + 1.0, 0.72)
    c.text((xr + 1.0) / 2, 0.36, ["If ", ("r(Q) = 0", "bold"), " then ", ("r(P ⊕ Q) = r(P)", "bold"),
                                  ": the new term has the same row value."], 12, inside=F)
    c.save("n08_parity")


# ---------------------------------------------------------------------------------------
# n09  a hash key moves terms, a conserved-parity key does not
# ---------------------------------------------------------------------------------------
def n09():
    c = Canvas()
    pw, gap, gw, gg = 4.3, 0.6, 1.0, 0.08
    yb, gh = 1.0, 3.35

    def gpus(x0):
        xs = []
        for i in range(4):
            x = x0 + (pw - 4 * gw - 3 * gg) / 2 + i * (gw + gg)
            G = c.box(x, yb, gw, gh)
            c.text(x + gw / 2, yb + gh - 0.3, f"GPU {i}", 12, "bold", inside=G)
            xs.append(x)
        return xs

    def chip(xc, y, w, s):
        R = c.box(xc - w / 2, y - 0.23, w, 0.46, fc="white", r=0.05, z=4, lw=1.2)
        c.text(xc, y, s, 12, inside=R)

    c.text(pw / 2, yb + gh + 0.4, "Key = hash of the whole Pauli string", 12.5, "bold")
    xs = gpus(0.0)
    ym = yb + 1.5
    chip(xs[1] + gw / 2, ym, 0.5, "P")
    chip(xs[3] + gw / 2, ym, 0.8, "P ⊕ Q")
    c.arrow(xs[1] + gw / 2 + 0.31, ym, xs[3] + gw / 2 - 0.46, ym, color=BAD, lw=2.0, z=7)
    c.stack(pw / 2, 0.5, [T("the new term hashes to another GPU:", 11.5, "normal", BAD),
                          T("must be sent at every branching gate", 11.5, "normal", BAD)])

    x0 = pw + gap
    c.text(x0 + pw / 2, yb + gh + 0.4, "Key = parities the window's gates keep", 12.5, "bold")
    xs = gpus(x0)
    S = c.box(xs[1] + 0.07, yb + 0.14, gw - 0.14, 2.5, fc=tint(SECTORQ, 0.13), ec=SECTORQ, lw=1.6, z=3)
    xc = xs[1] + gw / 2
    c.text(xc, yb + 2.34, "sector", 11.5, inside=S)
    chip(xc, yb + 1.82, 0.5, "P")
    c.arrow(xc, yb + 1.52, xc, yb + 0.92, color=GOOD, lw=2.0, z=7)
    chip(xc, yb + 0.6, 0.74, "P ⊕ Q")
    c.stack(x0 + pw / 2, 0.5, [T("the new term has the same key:", 11.5, "normal", GOOD),
                               T("stays; its merge partner is in the same sector", 11.5, "normal", GOOD)])
    c.save("n09_keys")


# ---------------------------------------------------------------------------------------
# n10  the rule on a time axis
# ---------------------------------------------------------------------------------------
def n10():
    c = Canvas()
    ww, bw, g = 2.95, 0.55, 0.08
    yb, bh = 1.8, 2.85
    lane_h, pitch = 0.36, 0.44
    names = ("Window k", "Window k+1", "Window k+2")
    total = 3 * ww + 2 * bw + 4 * g
    c.text(total / 2, yb + bh + 0.9, ["Kicked Ising: ", ("2,540", "bold"), " gates  →  ", ("20", "bold"), " windows"], 13)
    for k in range(3):
        x = k * (ww + bw + 2 * g)
        W = c.box(x, yb, ww, bh, fc="white")
        c.text(x + ww / 2, yb + bh + 0.3, names[k], 12.5, "bold")
        c.stack(x + ww / 2, yb + bh - 0.52, [T("each GPU applies the window's", 11), T("gates to its own sectors:", 11),
                                             T("branch, merge, truncate", 11)], inside=W, lead=1.38)
        for i in range(4):
            y = yb + 0.14 + (3 - i) * pitch
            L = c.box(x + 0.12, y, ww - 0.24, lane_h, fc=tint(SECTORQ, 0.16), ec=SECTORQ, lw=1.3, r=0.04, z=3)
            c.text(x + 0.24, y + lane_h / 2, f"GPU {i}", 11, ha="left", inside=L)
            c.arrow(x + 0.95, y + lane_h / 2, x + ww - 0.26, y + lane_h / 2, color=SECTORQ, lw=1.3, head=11)
        c.text(x + ww / 2, yb - 0.28, "no terms move between GPUs", 11.5)
        if k < 2:
            xb = x + ww + g
            c.box(xb, yb, bw, bh, fc=BOX, ec=EDGE)
            c.arrow(xb + bw / 2, yb + 0.2, xb + bw / 2, yb + 0.14 + 3 * pitch + lane_h - 0.06, color=SUB, both=True, lw=1.4, head=12)
            c.arrow(xb + bw / 2, 1.2, xb + bw / 2, yb - 0.08, color=SUB, lw=1.3, head=11)
            c.stack(xb + bw / 2, 0.92, [T("boundary: re-split by the next rows", 11), T("and exchange once", 11)], lead=1.38)
    c.arrow(0.0, 0.3, total, 0.3, color=SUB, lw=1.3, head=12)
    c.text(total, 0.1, "time", 11, color=SUB, ha="right")
    c.save("n10_rule_timeline")


# ---------------------------------------------------------------------------------------
# n11  windows end where the operator is smallest
# ---------------------------------------------------------------------------------------
def n11():
    terms = [1, 2, 10, 48, 2582, 2146372, 35930513, 108287276, 151203327, 158924060, 146839005, 133005210,
             120522709, 110316483, 95647971, 81236389, 90049862, 103334041, 92371508, 77582584]
    sectors = {4: 1, 5: 4, 6: 16, 7: 64, 8: 128, 9: 64, 10: 64}
    sectors.update({k: 32 for k in range(11, 20)})
    peak = 1407075677
    fig, ax = plt.subplots(figsize=(10, 5.3))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.9, bottom=0.13)
    ax.bar(range(20), [t - 0.5 for t in terms], width=0.72, color=SECTORQ, bottom=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3e10)
    log_axis(ax.yaxis, [1, 1e2, 1e4, 1e6, 1e8, 1e10], count_label, minor=[1e1, 1e3, 1e5, 1e7, 1e9])
    ax.grid(axis="y")
    ax.set_xlim(-0.7, 19.7)
    ax.set_xticks(range(20))
    ax.tick_params(axis="x", length=0)
    for k, n in sectors.items():
        above(ax, k, terms[k], str(n), dy=3, size=11)
    ax.axhline(peak, color=INK, lw=1.3, ls=(0, (5, 3)))
    rich(ax, -0.5, peak, ["largest count inside a window: ", ("1.41 billion", "bold")], ha="left", dy=6)
    ax.annotate("number above a bar =\nsectors in that window", xy=(-0.5, 6e7), ha="left", va="center", fontsize=11.5,
                color=SUB, linespacing=1.35)
    ax.set_xlabel("Window")
    ax.set_ylabel("Terms at the start of the window")
    ax.set_title("Kicked Ising, cutoff 1e-5, 16 GPUs")
    save_chart(fig, "n11_window_terms")


# ---------------------------------------------------------------------------------------
# n12  Clifford gates are not applied to the data
# ---------------------------------------------------------------------------------------
def n12():
    gates, ops, clifford = 5420, 2541, 2880
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.15)
    ax.bar(0, gates - clifford, width=0.5, color=EDGE)
    ax.bar(0, clifford, bottom=gates - clifford, width=0.5, color="#D3D8DF", edgecolor=EDGE, linewidth=1.0)
    ax.bar(1, ops, width=0.5, color=SECTORQ)
    above(ax, 0, gates, f"{gates:,}", size=12.5, weight="bold")
    above(ax, 1, ops, f"{ops:,}", size=12.5, weight="bold")
    xb = 0.31
    ax.plot([xb - 0.03, xb, xb, xb - 0.03], [gates - clifford, gates - clifford, gates, gates], color=INK, lw=1.3)
    ym = gates - clifford / 2
    ax.plot([xb, xb + 0.04], [ym, ym], color=INK, lw=1.3)
    rich(ax, xb + 0.07, ym, [("2,880", "bold"), " Clifford gates"], ha="left", dy=14)
    rich(ax, xb + 0.07, ym, "absorbed into the generators", ha="left", dy=-4)
    rich(ax, xb + 0.07, ym, "of the following rotations", ha="left", dy=-22)
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(0, 6200)
    ax.set_xticks([0, 1], ["Gates in\nthe circuit", "Operations applied\nto the terms"])
    ax.tick_params(axis="x", length=0, labelsize=12)
    ax.set_yticks(range(0, 6001, 1000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v):,}"))
    ax.grid(axis="y")
    ax.set_ylabel("Count (gates or operations)")
    ax.set_title("127-qubit kicked Ising circuit")
    save_chart(fig, "n12_frame")


# ---------------------------------------------------------------------------------------
# n13  run time changes little with the tuning values
# ---------------------------------------------------------------------------------------
def n13():
    style = {16: dict(ls="-", marker="o"), 4: dict(ls=(0, (5, 2.5)), marker="s"), 1: dict(ls=(0, (1.2, 1.8)), marker="^")}
    panels = [
        ("(a) Target terms per sector (million)", ["1.5", "3", "6\n(default)", "12"], 2,
         {4: ["51.1", "47.0", "46.7", "44.6"], 16: ["14.7", "12.9", "12.2", "11.7"]}),
        ("(b) Sectors per GPU", ["1", "2\n(default)", "4", "8"], 1,
         {4: ["44.5", "46.7", "46.8", "46.7"], 16: ["12.3", "12.2", "12.6", "13.6"]}),
        ("(c) GPU memory allowed (GB)", ["36\n(default)", "18", "8", "4"], 0,
         {1: ["168", "177", "219", "274"], 4: ["46.7", "45.7", "55.6", "71.1"]}),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 6.0), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.8, bottom=0.17, wspace=0.07)
    for ax, (xlabel, ticks, default, series) in zip(axs, panels):
        ax.axvspan(default - 0.36, default + 0.36, color=BOX, zorder=0)
        for gpus, shown in series.items():      # the labels are the measured values as reported
            ys = [float(v) for v in shown]
            ax.plot(range(4), ys, color=SECTORQ, lw=2, ms=7.5, mfc=SECTORQ, mec="white", mew=1.2, **style[gpus])
            for i, y in enumerate(ys):
                above(ax, i, y, shown[i], dy=7)
        ax.set_yscale("log")
        ax.set_ylim(8, 450)
        log_axis(ax.yaxis, [10, 20, 50, 100, 200, 400], plain)
        ax.grid(axis="y")
        ax.set_xlim(-0.5, 3.5)
        ax.set_xticks(range(4), ticks)
        ax.set_xlabel(xlabel, labelpad=8)
    axs[0].set_ylabel("Run time (s)")
    for ax in axs[1:]:
        ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=SECTORQ, lw=2, ms=7.5, mfc=SECTORQ, mec="white", mew=1.2, **style[g]) for g in (16, 4, 1)]
    fig.legend(handles, ["SectorQ, 16 GPUs", "SectorQ, 4 GPUs", "SectorQ, 1 GPU"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.53, 0.9), handlelength=3.2, columnspacing=2.5)
    fig.suptitle("Kicked Ising, cutoff 1e-5", fontsize=13.5, x=0.53, y=0.95)
    save_chart(fig, "n13_knobs")


# ---------------------------------------------------------------------------------------
# n14  which element gives which gain
# ---------------------------------------------------------------------------------------
def n14():
    labels = ["SectorQ\n(conserved-\nparity\nsectors)", "cuPauliProp,\ndedup only\nbefore\ntruncation",
              "cuPauliProp,\nClifford gates\nnot applied\nto the data", "cuPauliProp,\noriginal\ncircuit"]
    face = [SECTORQ, CUPP_DEFER, CUPP, CUPP_ORIGINAL]
    edge = [SECTORQ, CUPP_DEFER, CUPP, CUPP]
    panels = [("Cutoff 1e-4 (at most 22.5 million terms)", [0.81, 6.5, 28.3, 59.7]),
              ("Cutoff 1e-5 (at most 1.4 billion terms)", [12.2, 48.8, 54.3, 102.9])]
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 6.2), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.21, wspace=0.06)
    for ax, (title, ys) in zip(axs, panels):
        ax.bar(range(4), [y - 0.3 for y in ys], bottom=0.3, width=0.58, color=face, edgecolor=edge, linewidth=1.4)
        for i, y in enumerate(ys):
            above(ax, i, y, f"{y:g} s", dy=4)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 300)
        log_axis(ax.yaxis, [1, 10, 100], plain, minor=[0.5, 2, 5, 20, 50, 200])
        ax.grid(axis="y")
        ax.set_xlim(-0.6, 3.6)
        ax.set_xticks(range(4), labels)
        ax.tick_params(axis="x", length=0, labelsize=11, pad=6)
        ax.set_title(title, fontsize=13)
    axs[0].set_ylabel("Run time (s)")
    axs[1].tick_params(axis="y", which="both", length=0)
    fig.suptitle("Kicked Ising, 16 GPUs", fontsize=14, x=0.53, y=0.955)
    save_chart(fig, "n14_staircase")


# ---------------------------------------------------------------------------------------
# n15  the platform and what ran where
# ---------------------------------------------------------------------------------------
def n15():
    c = Canvas()
    y0, h = 1.55, 3.6

    def node(x, y, w, hh, gpus=True):
        c.rect(x, y, w, hh, fc="white", ec=EDGE, lw=1.0, z=3)
        if gpus:
            g = w / 6.5
            for i in range(4):
                c.rect(x + g * (0.5 + 1.5 * i), y + hh * 0.25, g, hh * 0.5, fc=EDGE, ec=EDGE, lw=0.5, z=4)

    A = c.box(0.0, y0, 3.9, h)
    c.text(1.95, y0 + h - 0.38, "One node", 13.5, "bold", inside=A)
    for i in range(4):
        G = c.rect(0.25 + 0.87 * i, y0 + 2.2, 0.79, 0.58, fc="white", ec=EDGE, lw=1.3)
        c.text(0.25 + 0.87 * i + 0.395, y0 + 2.49, "GPU", 11.5, inside=G)
    c.stack(1.95, y0 + 1.05, [T("4 × NVIDIA A100"), T("(40 GB; 36 GB used per GPU)"), 0.1,
                              T("64 CPU cores"), 0.1, T("256 GiB host memory")], inside=A)

    xm, w = 4.3, 2.9
    M = c.box(xm, y0, w, h)
    c.stack(xm + w / 2, y0 + h - 0.6, [T("1–4 nodes", 13.5, "bold"), T("(1–16 GPUs)", 13.5, "bold")], inside=M)
    for i in range(4):
        node(xm + 0.2 + 0.64 * i, y0 + 1.55, 0.58, 0.42)
    c.stack(xm + w / 2, y0 + 0.72, [T("SectorQ and every"), T("compared implementation")], inside=M)

    xr = xm + w + 0.4
    R = c.box(xr, y0, w, h)
    c.stack(xr + w / 2, y0 + h - 0.6, [T("8–32 nodes", 13.5, "bold"), T("(32–128 GPUs)", 13.5, "bold")], inside=R)
    for j in range(4):
        for i in range(8):
            node(xr + 0.23 + 0.31 * i, y0 + 1.25 + 0.27 * j, 0.27, 0.2, gpus=False)
    c.stack(xr + w / 2, y0 + 0.72, [T("SectorQ and cuPauliProp"), T("distributed mode")], inside=R)

    F = c.box(0.0, 0.0, xr + w, 1.2)
    c.stack((xr + w) / 2, 0.6, [T("7 circuits (64–127 qubits), a 30-qubit circuit with a known exact value,"),
                                T("lattices of 36–1,024 qubits; one run per setting; 40-minute limit per run")], inside=F)
    c.save("n15_setup")


# ---------------------------------------------------------------------------------------
# n16  small problems gain little from more GPUs
# ---------------------------------------------------------------------------------------
def n16():
    labels = ["1e-3\n(31 thousand)", "1e-4\n(2.33 million)", "3e-5\n(20.7 million)", "1e-5\n(159 million)", "3e-6\n(1.5 billion)"]
    gain = [1.3, 2.1, 9.0, 14, 14]
    shown = ["1.3x", "2.1x", "9.0x", "14x", "14x"]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.2)
    ax.bar(range(5), gain, width=0.56, color=SECTORQ)
    for i in range(5):
        above(ax, i, gain[i], shown[i], dy=3, size=12)
    ax.axhline(16, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax.annotate("16 GPUs: ideal", xy=(-0.5, 16), xytext=(0, 5), textcoords="offset points", ha="left", va="bottom", fontsize=11.5)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xticks(range(5), labels)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y")
    ax.set_xlabel("Cutoff (terms)", labelpad=8)
    ax.set_ylabel("Speedup over 1 GPU (times)")
    ax.set_title("Kicked Ising: SectorQ on 16 GPUs")
    save_chart(fig, "n16_gain_by_size")


FIGURES = [n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11, n12, n13, n14, n15, n16]

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    for f in FIGURES:
        if not want or any(f.__name__.startswith(w) for w in want):
            f()
    for name in sorted(SIZES):
        w, h = SIZES[name]
        print(f"{name}.png  {w} x {h} px  ({w / h:.2f}:1)")
