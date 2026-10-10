#!/usr/bin/env python3
"""Campaign g: SectorQ runs on more than one GPU that were made with the first campaign binary go back to the list.

The engine for these runs changed on 2026-10-10 01:20 (bin/pp_engine_h: the rows of the next window are chosen while
the window runs, one bucket per GPU for short windows). A run whose log does not name pp_engine_h is set aside
(queue_g/done_g/<id>, logs/g_first/) and is open again. Safe to repeat: runs that were in flight are caught later.
"""
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queue_g")
os.makedirs(os.path.join(Q, "done_g"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs", "g_first"), exist_ok=True)
moved = 0
for path in sorted(glob.glob(os.path.join(Q, "todo_a*.jsonl")) + glob.glob(os.path.join(Q, "todo_c*.jsonl"))):
    for line in open(path):
        if not line.strip():
            continue
        i = json.loads(line)
        done = os.path.join(Q, "done", i["id"])
        if i["kind"] != "eng" or i["nodes"] * i["per"] == 1 or not os.path.exists(done):
            continue
        log = os.path.join(ROOT, "logs", "g", i["id"] + ".out")
        record = json.load(open(done))
        if record.get("bin") == "pp_engine_h" or (os.path.exists(log) and "binary=pp_engine_h" in open(log, errors="replace").read(400)):
            continue
        if record["status"] in ("skipped_after_failure", "predicted_tl"):
            os.remove(done)                      # decided again by the ladder
            continue
        os.replace(done, os.path.join(Q, "done_g", i["id"]))
        if os.path.exists(log):
            shutil.copy2(log, os.path.join(ROOT, "logs", "g_first", i["id"] + ".out"))
        for f in glob.glob(os.path.join(ROOT, "logs", "g", f"smi.{i['id']}.*.csv")):
            os.replace(f, os.path.join(ROOT, "logs", "g_first", os.path.basename(f)))
        lock = os.path.join(Q, "lock", i["id"])
        if os.path.isdir(lock):
            for f in glob.glob(os.path.join(lock, "*")):
                os.remove(f)
            os.rmdir(lock)
        moved += 1
print(moved, "runs set aside and open again")
