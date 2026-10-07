#!/usr/bin/env python3
"""Sparse Z-parity rows (1-3 qubits) for ownership: terms moved per re-key and load balance.

A row over a qubit set T moves a term at the ZZ layer iff sum_{a in T} (L x)_a is odd, where L is
the lattice Laplacian mod 2. Rows over a few qubits whose neighbourhoods rarely hold X/Y letters
move few terms. Also reports how many qubits are touched by any term (register width needed).

Usage: sparse_rows.py <ibm_eagle.hpp> <xz dump> [...]
"""
import itertools
import re
import sys

import numpy as np

NQ = 127
rng = np.random.default_rng(5)
txt = open(sys.argv[1]).read()
edges = sorted({tuple(sorted((int(a), int(b)))) for a, b in re.findall(r"\{\s*(\d+),\s*(\d+)\}", txt)})


def bits(words):
    out = np.empty((words.shape[0], NQ), dtype=np.uint8)
    for q in range(NQ):
        out[:, q] = ((words[:, q // 64] >> np.uint64(q % 64)) & np.uint64(1)).astype(np.uint8)
    return out


for f in sys.argv[2:]:
    a = np.fromfile(f, dtype=np.uint64).reshape(-1, 4)
    full_n = a.shape[0]
    anyX = np.bitwise_or.reduce(a[:, 0:2], axis=0); anyZ = np.bitwise_or.reduce(a[:, 2:4], axis=0)
    touched = sum(bin(int(anyX[w] | anyZ[w])).count("1") for w in range(2))
    if a.shape[0] > 300000:
        a = a[rng.choice(a.shape[0], 300000, replace=False)]
    X, Z = bits(a[:, 0:2]), bits(a[:, 2:4])
    LX = np.zeros_like(X)                      # (L x)_a = sum over neighbours b of (x_a + x_b)
    for q, b in edges:
        d = X[:, q] ^ X[:, b]
        LX[:, q] ^= d; LX[:, b] ^= d
    Zn = Z ^ LX                                # Z bits after the ZZ layer
    pz = Zn.mean(0); pm = LX.mean(0)
    good = [q for q in np.argsort(pm) if 0.25 < pz[q] < 0.75][:24]
    out = []
    for g in (2, 3, 4):
        best = None
        for combo in itertools.combinations(good[:14], g):
            o_old = sum(Z[:, q].astype(np.int64) << i for i, q in enumerate(combo))
            o_new = sum(Zn[:, q].astype(np.int64) << i for i, q in enumerate(combo))
            cnt = np.bincount(o_new, minlength=1 << g)
            bal = cnt.max() / cnt.mean(); mv = float((o_old != o_new).mean())
            key = (bal > 1.25, mv)
            if best is None or key < best[0]:
                best = (key, mv, bal, combo)
        out.append(f"R={1 << g}: move {100 * best[1]:.1f}% (balance {best[2]:.2f})")
    print(f"{f.split('/')[-1].split('.xz')[0]}: terms {full_n}, qubits touched by any term {touched}/127, "
          f"single-qubit rows with 25-75% parity: {int(((pz > 0.25) & (pz < 0.75)).sum())}; quietest of them move "
          f"{100 * pm[good[0]]:.1f}%, {100 * pm[good[1]]:.1f}%, {100 * pm[good[2]]:.1f}% of terms")
    print("   best sparse rows | " + " | ".join(out))
