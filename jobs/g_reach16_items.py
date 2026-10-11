#!/usr/bin/env python3
"""Campaign g, added 2026-10-10 (4): the kicked Ising ladder on 16 GPUs below cutoff 1e-6, down to the memory limit.

  python3 jobs/g_reach16_items.py        writes queue_g/todo_i.jsonl

The sweep of the campaign ended at cutoff 1e-6 on 16 GPUs, which finished with 788 of 1,024 GiB of host memory: the
step where the four nodes run out of memory was not in the list. These runs continue the same ladder (same settings),
with a limit of two hours per run so that memory, not time, ends it. The same ladder runs on 32-512 GPUs in the scale
campaign (jobs/s_items.py).
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
items = [dict(id=f"eng.ki.{x}.g16", nodes=4, per=4, kind="eng", limit=7200, ladder="eng.ki.g16", coef=float(x), est=3600,
              args=f"--circuit ki_lab --weight 0 --coef {x} --target 6000000 --trace 1 --vram-frac 0.9")
         for x in ("7e-7", "5e-7")]
tmp = os.path.join(Q, ".todo_i.tmp")
with open(tmp, "w") as f:
    for i in items:
        f.write(json.dumps(i) + "\n")
os.replace(tmp, os.path.join(Q, "todo_i.jsonl"))
print(f"{len(items)} runs in queue_g/todo_i.jsonl")
