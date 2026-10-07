#!/usr/bin/env python3
"""Population statistics of dumped Pauli expansions and the balance of candidate ownership maps.

Usage: ownership.py <dump_dir> <tag>

For every snapshot it reports the support of the operator, how many bits are usable for
partitioning, and the load balance (largest bucket / mean bucket) for R = 4, 16, 64, 256
buckets under: a generic hash of the whole string, parities of single bits, parities of the
Z bits over lattice regions, and parities of random subsets of Z bits.
"""
import glob
import os
import re
import sys

import numpy as np

NQ = 127


def load(path):
    a = np.fromfile(path, dtype=np.uint64).reshape(-1, 4)
    return a[:, 0:2], a[:, 2:4]


def bit(words, q):
    return ((words[:, q // 64] >> np.uint64(q % 64)) & np.uint64(1)).astype(np.uint8)


def bits_matrix(words):
    out = np.empty((words.shape[0], NQ), dtype=np.uint8)
    for q in range(NQ):
        out[:, q] = bit(words, q)
    return out


def balance(owner, nbuckets):
    cnt = np.bincount(owner, minlength=nbuckets)
    return cnt.max() / cnt.mean(), (cnt == 0).sum()


def mix64(x, z):
    h = np.zeros(x.shape[0], dtype=np.uint64)
    for w in (x[:, 0], x[:, 1], z[:, 0], z[:, 1]):
        h ^= w + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2))
    h ^= h >> np.uint64(30); h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27); h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


def rows_to_owner(rows):
    owner = np.zeros(rows[0].shape[0], dtype=np.int64)
    for i, r in enumerate(rows):
        owner |= r.astype(np.int64) << i
    return owner


def main():
    ddir, tag = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(1)
    files = sorted(glob.glob(os.path.join(ddir, f"{tag}.g*.xz.bin")))
    for f in files:
        m = re.search(r"\.g(\d+)\.n(\d+)\.s(\d+)\.", f)
        applied, n, stride = int(m.group(1)), int(m.group(2)), int(m.group(3))
        x, z = load(f)
        if x.shape[0] < 1000:
            print(f"gate {applied}: {n} terms (too few for statistics)")
            continue
        X, Z = bits_matrix(x), bits_matrix(z)
        nonI = (X | Z).mean(0)
        px, pz = X.mean(0), Z.mean(0)
        weight = (X | Z).sum(1)
        support = np.where(nonI > 0.01)[0]
        usable_z = int(((pz > 0.2) & (pz < 0.8)).sum())
        usable_x = int(((px > 0.2) & (px < 0.8)).sum())
        print(f"gate {applied}: terms={n} (sample {x.shape[0]}), mean weight={weight.mean():.1f}, "
              f"qubits with >1% non-identity={len(support)}, bits with 20-80% ones: z={usable_z} x={usable_x}")

        h = mix64(x, z)
        order_z = np.argsort(np.abs(pz - 0.5))
        sup = support if len(support) >= 8 else np.arange(NQ)
        regions = np.array_split(np.sort(sup), 8)       # contiguous index ranges inside the support
        line_hash, line_bit, line_reg, line_rand = [], [], [], []
        for g in (2, 4, 6, 8):
            R = 1 << g
            b, _ = balance((h % np.uint64(R)).astype(np.int64), R)
            line_hash.append(f"{b:.3f}")
            b, _ = balance(rows_to_owner([Z[:, q] for q in order_z[:g]]), R)
            line_bit.append(f"{b:.3f}")
            rows = [Z[:, reg].sum(1) & 1 for reg in regions[:g]]
            b, _ = balance(rows_to_owner(rows), R)
            line_reg.append(f"{b:.3f}")
            rows = []
            for _ in range(g):
                pick = sup[rng.random(len(sup)) < 0.5]
                rows.append(Z[:, pick].sum(1) & 1)
            b, _ = balance(rows_to_owner(rows), R)
            line_rand.append(f"{b:.3f}")
        print("   max/mean bucket for R=4,16,64,256")
        print("     hash of whole string        : " + "  ".join(line_hash))
        print("     single Z bits (most even)   : " + "  ".join(line_bit))
        print("     Z parity over lattice region: " + "  ".join(line_reg))
        print("     Z parity over random subset : " + "  ".join(line_rand))


if __name__ == "__main__":
    main()
