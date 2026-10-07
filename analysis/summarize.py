#!/usr/bin/env python3
"""Summarize per-gate, per-rank logs written by ki_prof.

Usage: summarize.py <result_dir> [tag ...]

For each run (tag) it reports where wall time goes (GPU-side work vs time inside MPI
calls), the per-gate synchronization floor, how many bytes move between ranks, the
load balance across ranks, and the memory the library requests per live term.
"""
import csv
import glob
import os
import sys

import numpy as np

KINDS = ["barrier", "wait", "test", "send", "isend", "recv", "irecv",
         "bcast", "allred", "allredmin", "minloc", "allgather"]
COLLECTIVE = ["barrier", "bcast", "allred", "allredmin", "minloc", "allgather"]
P2P = ["wait", "test", "send", "isend", "recv", "irecv"]
BYTES_PER_TERM = 40.0  # 127 qubits: 2 x 2 x 8 B of X/Z bits + 8 B coefficient


def load(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    data = np.array([[float(x) for x in r[:len(hdr)]] for r in rows[1:] if len(r) >= len(hdr)])
    return hdr, data


def load_run(rdir, tag):
    files = sorted(glob.glob(os.path.join(rdir, f"{tag}.rank*.csv")))
    if not files:
        return None
    hdr, first = load(files[0])
    n = min(len(load(f)[1]) for f in files)
    arr = np.stack([load(f)[1][:n] for f in files])  # ranks x records x columns
    return hdr, arr


def col(hdr, arr, name):
    return arr[:, :, hdr.index(name)]


def gpu_util(rdir, tag):
    path = os.path.join(rdir, f"{tag}.gpu.csv")
    if not os.path.exists(path):
        return None
    util = {}
    with open(path) as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 4:
                continue
            try:
                util.setdefault(int(p[1]), []).append(float(p[2]))
            except ValueError:
                pass
    return {g: (np.mean(v), len(v)) for g, v in sorted(util.items())}


def summarize(rdir, tag):
    run = load_run(rdir, tag)
    if run is None:
        print(f"## {tag}: no data")
        return
    hdr, arr = run
    R, N, _ = arr.shape
    prof = "wait_time" in hdr
    kind = col(hdr, arr, "kind")[0]
    nin = col(hdr, arr, "nin")
    nout = col(hdr, arr, "nout")
    tprep = col(hdr, arr, "t_prep")
    tcomp = col(hdr, arr, "t_comp")
    gin = nin.sum(0)
    gout = nout.sum(0)
    step = (tprep + tcomp).mean(0)          # mean over ranks, per record
    total = step.sum()
    print(f"## {tag}: ranks={R} records={N} total step time (mean over ranks)={total:.3f} s")
    print(f"   global terms: max in={gin.max():.3e}  final={gout[-1]:.3e}")
    for k, name in [(0, "X rotation"), (1, "ZZ rotation"), (2, "explicit dedup"), (4, "explicit trunc"), (3, "trace")]:
        m = kind == k
        if m.any():
            print(f"   {name:15s}: n={m.sum():5d}  time={step[m].sum():8.3f} s  mean={1e3 * step[m].mean():8.3f} ms")

    if prof:
        t = {k: col(hdr, arr, f"{k}_time").mean(0) for k in KINDS}
        c = {k: col(hdr, arr, f"{k}_calls").mean(0) for k in KINDS}
        b = {k: col(hdr, arr, f"{k}_bytes").sum(0) for k in KINDS}
        tcoll = sum(t[k] for k in COLLECTIVE)
        tp2p = sum(t[k] for k in P2P)
        tmpi = tcoll + tp2p
        print(f"   time inside MPI calls: {tmpi.sum():.3f} s = {100 * tmpi.sum() / total:.1f}% of step time "
              f"(collectives {100 * tcoll.sum() / total:.1f}%, point-to-point {100 * tp2p.sum() / total:.1f}%)")
        gates = kind <= 1
        print("   MPI calls per gate per rank: " + ", ".join(
            f"{k}={c[k][gates].mean():.2f}" for k in KINDS if c[k][gates].sum() > 0))
        print("   MPI time share by call:      " + ", ".join(
            f"{k}={100 * t[k].sum() / total:.1f}%" for k in KINDS if t[k].sum() > 0))
        sent = b["isend"] + b["send"]
        print(f"   bytes sent between ranks: {sent.sum():.3e} "
              f"(= {sent.sum() / BYTES_PER_TERM:.3e} term-equivalents); "
              f"terms produced: {gout[gates].sum():.3e}; "
              f"sent/produced = {sent.sum() / BYTES_PER_TERM / max(gout[gates].sum(), 1):.2f}")

    # behaviour by expansion size
    print("   by global input terms:        gates   ms/gate   Mterms/s" + ("   MPI%   coll%   sent/out" if prof else ""))
    edges = [0, 1e2, 1e4, 1e5, 1e6, 1e7, 1e8, 1e12]
    gmask = kind <= 1
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = gmask & (gin >= lo) & (gin < hi)
        if not m.any():
            continue
        line = (f"     [{lo:7.0e},{hi:7.0e})   {m.sum():6d}  {1e3 * step[m].mean():8.3f}  "
                f"{gin[m].sum() / step[m].sum() / 1e6:9.2f}")
        if prof:
            line += (f"  {100 * tmpi[m].sum() / step[m].sum():5.1f}  {100 * tcoll[m].sum() / step[m].sum():5.1f}"
                     f"  {sent[m].sum() / BYTES_PER_TERM / max(gout[m].sum(), 1):8.2f}")
        print(line)

    if R > 1:
        big = gmask & (gin > 1e5)
        if big.any():
            imb = nin[:, big].max(0) / np.maximum(nin[:, big].mean(0), 1)
            print(f"   load balance (gates with >1e5 global terms): max/mean local terms "
                  f"median={np.median(imb):.3f} worst={imb.max():.3f}")
        # spread of per-rank compute time on the same gate
        sp = (tcomp.max(0) - tcomp.min(0))[gmask]
        print(f"   per-gate compute time spread across ranks: mean={1e3 * sp.mean():.3f} ms "
              f"({100 * sp.sum() / tcomp.mean(0)[gmask].sum():.1f}% of compute time)")

    # memory requested relative to live data
    rp = col(hdr, arr, "req_pauli")
    rc = col(hdr, arr, "req_coef")
    rw = col(hdr, arr, "req_ws")
    i = int(np.argmax(gin * gmask))
    live = nin[:, i] * BYTES_PER_TERM
    need = live + rp[:, i] + rc[:, i] + rw[:, i]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(live > 0, need / live, np.nan)
    print(f"   memory at the largest gate: live={live.sum() / 2**30:.2f} GiB, "
          f"output={(rp[:, i] + rc[:, i]).sum() / 2**30:.2f} GiB, workspace={rw[:, i].sum() / 2**30:.2f} GiB, "
          f"(live+output+workspace)/live = {np.nanmean(ratio):.2f}x")

    u = gpu_util(rdir, tag)
    if u:
        print("   GPU utilization (nvidia-smi, 200 ms samples): " + ", ".join(
            f"gpu{g}={m:.0f}% (n={n})" for g, (m, n) in u.items()))
    print()


def main():
    rdir = sys.argv[1]
    tags = sys.argv[2:]
    if not tags:
        tags = sorted({os.path.basename(f).split(".rank")[0] for f in glob.glob(os.path.join(rdir, "*.rank*.csv"))})
    for tag in tags:
        summarize(rdir, tag)


if __name__ == "__main__":
    main()
