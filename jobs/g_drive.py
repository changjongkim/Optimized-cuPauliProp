#!/usr/bin/env python3
"""Campaign g on one allocation:   python3 jobs/g_drive.py --job <allocation id> --order up|down

One runner per node works through the one-node runs, one per pair of nodes through the two-node runs and one on
all four nodes through the four-node runs ("up": in this order, "down": the four-node runs first). The passes are
repeated while they finish runs, so that runs appended to the lists in the meantime are picked up.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "jobs"))
import g_run                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True)
ap.add_argument("--order", default="up")
a = ap.parse_args()
nodes = g_run.squeue(a.job, "%N")
hosts = subprocess.run(["scontrol", "show", "hostnames", nodes], capture_output=True, text=True).stdout.split()
if not hosts:
    sys.exit(f"no hosts for job {a.job}")
groups = [[[h] for h in hosts]]
if len(hosts) >= 2:
    groups.append([hosts[i:i + 2] for i in range(0, len(hosts) - 1, 2)])
if len(hosts) >= 4:
    groups.append([hosts[:4]])
if a.order == "down":
    groups.reverse()


def note(text):
    with open(os.path.join(g_run.Q, "drive.log"), "a") as f:
        f.write(f"{time.strftime('%m-%dT%H:%M:%S')} job={a.job} {text}\n")


note(f"drive order={a.order} hosts={' '.join(hosts)}")
while True:
    before = len(os.listdir(os.path.join(g_run.Q, "done")))
    g_run.clean_locks()
    for sets in groups:
        procs = []
        for hs in sets:
            log = open(os.path.join(g_run.LOGS, f"run-{a.job}-{'+'.join(hs)}.log"), "a")
            procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "jobs", "g_run.py"), "--job", a.job, "--hosts", ",".join(hs)],
                                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
        for p in procs:
            p.wait()
    after = len(os.listdir(os.path.join(g_run.Q, "done")))
    note(f"pass finished: {after - before} runs")
    if after == before or g_run.left(a.job) < 300:
        break
note("drive finished")
