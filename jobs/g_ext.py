"""Campaign g, external baselines: called by jobs/g_run.py for the runs of kind "ext".

The run itself is done by jobs/g_ext_impl.py in a process of its own, so that this module stays the same for
runners that are already working.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(item, job, hosts):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "jobs", "g_ext_impl.py"), "--job", str(job), "--hosts", hosts,
                        "--item", json.dumps(item)], capture_output=True, text=True, cwd=ROOT)
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return dict(status="fail", message=(p.stderr or p.stdout)[-300:])
