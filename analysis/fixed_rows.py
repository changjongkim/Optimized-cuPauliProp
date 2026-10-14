#!/usr/bin/env python3
"""Where must a partition of the Pauli strings move terms? Window rows against rows fixed for the whole run.

  env/bin/python3 analysis/fixed_rows.py [--draws 20] [--g 4]

Uses only the generators of the circuits (the circuit files that SectorQ exports, results/f_ext/ir and the files of
the first implementation), no terms. For each of the seven circuits of the evaluation:

  windows   SectorQ's rule on the processing order (the operations left by the Clifford frame): a window grows while
            the generators of its branching rotations span at most n dimensions (rows valid for the window: 2n - rank
            >= n). The engine also ends a window right after a truncation, so its count can differ a little.
  fixed     g rows (2^g parts) fixed for the whole run, on the gates as the circuit gives them (no frame, as an
            implementation that applies every gate to its terms): a branching rotation or a Clifford rotation with
            generator Q moves terms when some row r has r.Q = 1 (plain parity of the bits); an H on qubit q when a row
            has x_q != z_q; a CZ on (a, b) when a row has z_a or z_b. Rows: random over all 2n bits, over the z bits,
            over the x bits (mean over draws), and the rows valid for the first window of the run, kept fixed.
  whole     the dimension of the rows that the whole processing order conserves; such a row has the value of the
            observable on every term, so it cannot divide the terms.
Prints one table (markdown) and writes analysis/fixed_rows.json.
"""
import argparse
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPL = "/pscratch/sd/s/sgkim/kcj/pauliprop-sectorq-impl/results/own_perf_v1_n1"
CIRCUITS = [("Kicked Ising", os.path.join(ROOT, "results/f_ext/ir/kis20.ir.json")),
            ("TFIM heavy-hex", os.path.join(IMPL, "tfim_engine_r0.ir.json")),
            ("Noisy kicked Ising", os.path.join(IMPL, "noise_engine_r0.ir.json")),
            ("TFIM 11x11", os.path.join(ROOT, "results/f_ext/ir/tfimsq.ir.json")),
            ("XXZ 8x8", os.path.join(ROOT, "results/f_ext/ir/xxz.ir.json")),
            ("QAOA", os.path.join(ROOT, "results/f_ext/ir/qaoa.ir.json")),
            ("Ansatz", os.path.join(ROOT, "results/f_ext/ir/hea.ir.json"))]


def vec(pauli, n):
    """Bit vector of a Pauli string as an int: bit q = x_q, bit n + q = z_q."""
    v = 0
    for q, c in enumerate(pauli):
        if c in "XY":
            v |= 1 << q
        if c in "ZY":
            v |= 1 << (n + q)
    return v


def parity(a):
    return bin(a).count("1") & 1


class Span:
    """GF(2) span of bit vectors (reduced basis keyed by the leading bit)."""
    def __init__(self):
        self.b = {}

    def reduce(self, v):
        while v:
            h = v.bit_length() - 1
            if h not in self.b:
                return v
            v ^= self.b[h]
        return 0

    def add(self, v):
        v = self.reduce(v)
        if v:
            self.b[v.bit_length() - 1] = v
            return True
        return False

    def rank(self):
        return len(self.b)


def complement(span, n2):
    """A basis of the rows r with r.v = 0 for every v in the span (plain parity)."""
    rows = []
    vecs = list(span.b.values())
    # solve by elimination on the transposed system: rows orthogonal to all basis vectors
    m = len(vecs)
    piv, mat = [], [v for v in vecs]
    used = []
    order = []
    for i in range(m):
        v = mat[i]
        for p, col in zip(used, order):
            if (v >> col) & 1:
                v ^= mat[p]
        if v == 0:
            continue
        col = (v & -v).bit_length() - 1
        mat[i] = v
        for j in range(len(used)):
            if (mat[used[j]] >> col) & 1:
                mat[used[j]] ^= v
        used.append(i)
        order.append(col)
    pivots = dict(zip(order, used))
    for free in range(n2):
        if free in pivots:
            continue
        r = 1 << free
        for col, i in pivots.items():
            if (mat[i] >> free) & 1:
                r |= 1 << col
        rows.append(r)
    return rows


def constraints(o, n):
    """Vectors c with r.c = 0 for every row r that the operation leaves valid (plain parity of the bits)."""
    if o["type"] == "rotation":
        return [vec(o["pauli"], n)]                      # branching or Clifford: P -> P xor Q keeps r iff r.Q = 0
    if o["type"] == "clifford" and o["gate"] == "H":
        q = o["qubits"][0]
        return [(1 << q) | (1 << (n + q))]               # H swaps x_q and z_q: r keeps its value iff r_xq = r_zq
    if o["type"] == "clifford" and o["gate"] == "CZ":
        a, b = o["qubits"]
        return [1 << (n + a), 1 << (n + b)]              # CZ adds x_b to z_a and x_a to z_b
    return []                                            # X, Pauli noise: no string changes


def moves(ops, rows, n):
    return sum(1 for cs in ops if any(parity(r & c) for r in rows for c in cs))


def analyse(path, draws, g, rng):
    ir = json.load(open(path))
    n = ir["qubits"]
    n2 = 2 * n
    # the operations SectorQ applies (after the Clifford frame where the circuit file has one), in processing order
    frame_ops = [cs for cs in (constraints(o, n) for o in ir["processing_order"]) if cs]
    windows, span, first, total = 1, Span(), None, Span()
    for cs in frame_ops:
        trial = Span()
        trial.b = dict(span.b)
        for c in cs:
            trial.add(c)
            total.add(c)
        if trial.rank() > n:                         # fewer than n valid rows left: the window ends before this operation
            first = first or span
            windows += 1
            span = Span()
            for c in cs:
                span.add(c)
        else:
            span = trial
    first = first or span
    first_ops = sum(1 for _ in frame_ops)
    # rows of the first window kept for the whole run, on the same operations
    firstrows = complement(first, n2)
    rng.shuffle(firstrows)
    fixed_first = moves(frame_ops, firstrows[:g], n)
    # the gates as the circuit gives them (no frame), processed backwards: what an implementation that applies every
    # gate to its terms sees
    gates = [cs for cs in (constraints(o, n) for o in reversed(ir["constructed_state_order"])) if cs]
    mean = lambda make: sum(moves(gates, [make() for _ in range(g)], n) for _ in range(draws)) / draws
    return dict(qubits=n, gates=len(gates), operations=first_ops, windows=windows, whole_rows=n2 - total.rank(),
                frame=bool(ir.get("frame")),
                fixed_random=mean(lambda: rng.getrandbits(n2) or 1), fixed_z=mean(lambda: (rng.getrandbits(n) or 1) << n),
                fixed_x=mean(lambda: rng.getrandbits(n) or 1), fixed_first=fixed_first)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--g", type=int, default=4, help="rows, 2^g parts")
    a = ap.parse_args()
    rng = random.Random(7)
    out = {}
    print(f"| Circuit | Qubits | Gates | Operations SectorQ applies (frame) | Windows (SectorQ rule) | Operations that move terms with the rows of the first window kept | Gates that move terms, {1 << a.g} fixed parts, no frame: random / z / x rows | Rows conserved by the whole run |")
    print("|---|---:|---:|---:|---:|---:|---|---:|")
    for name, path in CIRCUITS:
        r = analyse(path, a.draws, a.g, rng)
        out[name] = r
        print(f"| {name} | {r['qubits']} | {r['gates']:,} | {r['operations']:,}{'' if r['frame'] else ' (no frame)'} | {r['windows']} | {r['fixed_first']:,} | "
              f"{r['fixed_random']:,.0f} / {r['fixed_z']:,.0f} / {r['fixed_x']:,.0f} | {r['whole_rows']} |", flush=True)
    json.dump(out, open(os.path.join(ROOT, "analysis", "fixed_rows.json"), "w"), indent=1)
