#!/usr/bin/env python3
"""Load balance of Z-parity ownership rows chosen from the data, for 16 to 1024 buckets.

Usage: balance_rows.py <xz dump file> [...]. Uses a 400k-term sample of each dump."""
import sys
import numpy as np

NQ = 127
rng = np.random.default_rng(3)
for f in sys.argv[1:]:
    a = np.fromfile(f, dtype=np.uint64).reshape(-1, 4)[:, 2:4]
    if a.shape[0] > 400000:
        a = a[rng.choice(a.shape[0], 400000, replace=False)]
    Z = np.empty((a.shape[0], NQ), dtype=np.float32)
    for q in range(NQ):
        Z[:, q] = ((a[:, q // 64] >> np.uint64(q % 64)) & np.uint64(1)).astype(np.float32)
    C = 32
    picks = (rng.random((C * 10, NQ)) < 0.5).astype(np.float32)
    par = (Z @ picks.T).astype(np.int64) & 1
    out = []
    for g in (4, 6, 8, 10):
        R = 1 << g
        vals = []
        for c in range(C):
            owner = (par[:, c * 10:c * 10 + g] << np.arange(g)).sum(1)
            cnt = np.bincount(owner, minlength=R)
            vals.append(cnt.max() / cnt.mean())
        vals = np.array(vals)
        out.append(f"R={R}: median {np.median(vals):.2f}, best of 8 {vals[:8].min():.2f}, best of 32 {vals.min():.2f}")
    print(f.split('/')[-1].split('.xz')[0], "| " + " | ".join(out))
