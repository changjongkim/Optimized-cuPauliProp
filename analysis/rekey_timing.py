#!/usr/bin/env python3
"""Do re-keys happen where the operator is small?

Usage: rekey_timing.py <profile.csv> [...]   (written by pp_engine --profile)

For each run: the number of terms at the window boundaries the engine chose, against the
largest number of terms inside the windows and against the best boundaries it could have
chosen (the smallest population within half a window around each boundary).
"""
import sys
import numpy as np

for f in sys.argv[1:]:
    d = np.genfromtxt(f, delimiter=",", names=True)
    pop = d["terms_before"].astype(float)
    ws = np.where(d["window_start"] == 1)[0]
    ws = ws[pop[ws] > 1e4]                      # boundaries that matter
    if len(ws) < 3:
        print(f"{f.split('/')[-1]}: too few large windows"); continue
    lens = np.diff(ws)
    half = int(max(1, np.median(lens) // 2))
    at, best, peak = [], [], []
    for i, b in enumerate(ws[:-1]):
        lo, hi = max(0, b - half), min(len(pop), b + half)
        seg = pop[lo:hi]
        at.append(pop[b]); best.append(seg[seg > 0].min()); peak.append(pop[b:ws[i + 1]].max())
    at, best, peak = map(np.array, (at, best, peak))
    print(f"{f.split('/')[-1]:14s} windows={len(ws):3d} median length={int(np.median(lens)):4d} ops | terms at chosen boundaries: "
          f"sum {at.sum():.3e} | at the best nearby points: sum {best.sum():.3e} ({at.sum() / best.sum():.2f}x more moved than needed) | "
          f"peak inside a window / boundary: median {np.median(peak / at):.1f}x")
