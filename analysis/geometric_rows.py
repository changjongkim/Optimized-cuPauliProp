#!/usr/bin/env python3
"""How many terms must move at a re-key if ownership rows are Z parities over lattice regions?

A ZZ(pi/2) layer maps z_a -> z_a + sum over neighbours b of (x_a + x_b) (mod 2). If a row is
the Z parity over a region T before and after the layer, a term changes owner exactly when
an odd number of boundary edges of T have x_a != x_b. So rows over regions whose boundary
lies where X/Y letters are rare move few terms, while random-subset rows move about half
the terms per row. This script measures, on dumped populations taken right before a ZZ
block, the moved fraction and the load balance of both kinds of rows.

Usage: geometric_rows.py <ibm_eagle.hpp> <xz dump> [...]
"""
import re
import sys

import numpy as np

NQ = 127
rng = np.random.default_rng(11)


def edges_from_header(path):
    txt = open(path).read()
    pairs = re.findall(r"\{\s*(\d+),\s*(\d+)\}", txt)
    return sorted({tuple(sorted((int(a), int(b)))) for a, b in pairs})


def gf2_nullity(M):
    M = M.copy() % 2
    rank = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((r for r in range(rank, rows) if M[r, c]), None)
        if piv is None:
            continue
        M[[rank, piv]] = M[[piv, rank]]
        for r in range(rows):
            if r != rank and M[r, c]:
                M[r] ^= M[rank]
        rank += 1
    return cols - rank


def bits(words):
    out = np.empty((words.shape[0], NQ), dtype=np.uint8)
    for q in range(NQ):
        out[:, q] = ((words[:, q // 64] >> np.uint64(q % 64)) & np.uint64(1)).astype(np.uint8)
    return out


def main():
    edges = edges_from_header(sys.argv[1])
    adj = [[] for _ in range(NQ)]
    A = np.zeros((NQ, NQ), dtype=np.uint8)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        A[a, b] = A[b, a] = 1
    L = (A + np.diag(A.sum(1) % 2)) % 2
    print(f"lattice: {NQ} qubits, {len(edges)} edges; GF(2) nullity of the Laplacian = {gf2_nullity(L.astype(np.uint8))} "
          f"(number of Z parities conserved by the whole circuit)")

    for f in sys.argv[2:]:
        a = np.fromfile(f, dtype=np.uint64).reshape(-1, 4)
        if a.shape[0] > 300000:
            a = a[rng.choice(a.shape[0], 300000, replace=False)]
        X, Z = bits(a[:, 0:2]), bits(a[:, 2:4])
        n = X.shape[0]
        # Z bits after the ZZ layer
        Zn = Z.copy()
        for q, b in edges:
            d = X[:, q] ^ X[:, b]
            Zn[:, q] ^= d; Zn[:, b] ^= d
        active = np.where((X | Z).mean(0) > 0.01)[0]
        pe = {e: float((X[:, e[0]] ^ X[:, e[1]]).mean()) for e in edges}

        def row_stats(T):
            mask = np.zeros(NQ, dtype=bool); mask[list(T)] = True
            old = Z[:, mask].sum(1) & 1
            new = Zn[:, mask].sum(1) & 1
            return old, new

        # candidate regions: breadth-first balls around active qubits, and unions of two balls
        cands = []
        for seed in active:
            T = {int(seed)}; frontier = [int(seed)]
            for _ in range(12):
                nxt = [b for v in frontier for b in adj[v] if b not in T]
                if not nxt:
                    break
                T.update(nxt); frontier = nxt
                cands.append(frozenset(T))
        cands = list(set(cands))
        scored = []
        for T in cands:
            old, new = row_stats(T)
            p_new = new.mean()
            moved = float((old != new).mean())
            scored.append((moved, abs(p_new - 0.5), T, old, new))
        usable = [s for s in scored if s[1] < 0.1]            # parity within 40-60%
        usable.sort(key=lambda s: s[0])

        def joint(rows):
            g = len(rows)
            o_old = sum((r[3].astype(np.int64) << i) for i, r in enumerate(rows))
            o_new = sum((r[4].astype(np.int64) << i) for i, r in enumerate(rows))
            cnt = np.bincount(o_new, minlength=1 << g)
            return float((o_old != o_new).mean()), cnt.max() / cnt.mean()

        line = []
        for g in (2, 4):
            # greedy: add the usable region that keeps balance best among the lowest-movement ones
            chosen = []
            pool = usable[:60]
            while len(chosen) < g and pool:
                best = None
                for s in pool:
                    if any(s[2] == c[2] for c in chosen):
                        continue
                    mv, bal = joint(chosen + [s])
                    key = (bal > 1.15, mv, bal)
                    if best is None or key < best[0]:
                        best = (key, s, mv, bal)
                if best is None:
                    break
                chosen.append(best[1])
            if len(chosen) == g:
                mv, bal = joint(chosen)
                line.append(f"R={1 << g}: region rows move {100 * mv:.0f}% (balance {bal:.2f})")
            else:
                line.append(f"R={1 << g}: not enough usable regions")
            # random-subset rows for comparison
            rows = []
            for _ in range(g):
                T = [q for q in range(NQ) if rng.random() < 0.5]
                old, new = row_stats(T)
                rows.append((0, 0, frozenset(T), old, new))
            mv, bal = joint(rows)
            line[-1] += f", random rows move {100 * mv:.0f}% (balance {bal:.2f})"
        qe = sorted(pe.values())
        print(f"{f.split('/')[-1].split('.xz')[0]}: sample {n}, active qubits {len(active)}, "
              f"edges with x_a!=x_b in <5% of terms: {sum(v < 0.05 for v in qe)}/{len(edges)}, "
              f"usable regions {len(usable)}/{len(cands)}")
        print("   " + " | ".join(line))


if __name__ == "__main__":
    main()
