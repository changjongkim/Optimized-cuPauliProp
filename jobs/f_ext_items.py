#!/usr/bin/env python3
"""External baselines of campaign f (up to four nodes), one run each.

One line per run in queue_f/ext_todo.txt:  id|nodes|backend|case|cutoff|policy|ranks_per_node|threads|timeout_s
The runs go through the adapters and environments built in the worktree kcj/pauliprop-sectorq-impl
(bench/external.py); every implementation receives the same operation list as SectorQ. "matched" applies
SectorQ's truncation rule (same answer expected); MonoProp and OBP only have their own rule ("native").
CPU implementations get all 64 cores of a node; the MPI ones also run on two and four nodes.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = {"ki": ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6"],
          "tfim": ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "5e-6"],
          "noise": ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "5e-6"]}
#            backend    policy     ranks/node threads node counts  cases
BACKENDS = [("spd",      "matched", 1,  8,  (1,),       ("ki", "tfim")),
            ("orqa",     "matched", 64, 1,  (1, 2, 4),  ("ki", "tfim", "noise")),
            ("monoprop", "native",  8,  8,  (1, 2, 4),  ("ki", "tfim")),
            ("julia",    "matched", 1,  64, (1,),       ("ki", "tfim", "noise")),
            ("ppvm",     "matched", 1,  64, (1,),       ("ki", "tfim", "noise")),
            ("qiskit",   "matched", 1,  1,  (1,),       ("ki", "tfim")),
            ("obp",      "native",  1,  64, (1,),       ("ki", "tfim", "noise"))]
items = []
for case in ("ki", "tfim", "noise"):
    for backend, policy, rpn, threads, node_counts, cases in BACKENDS:
        if case not in cases:
            continue
        for nodes in node_counts:
            for coef in LADDER[case]:               # from the easiest cutoff down; the runner stops a ladder at its first failure
                items.append(f"{backend}.{case}.{coef}.n{nodes}|{nodes}|{backend}|{case}|{coef}|{policy}|{rpn}|{threads}|900")
open(os.path.join(ROOT, "queue_f", "ext_todo.txt"), "w").write("\n".join(items) + "\n")
print(len(items), "external runs")
