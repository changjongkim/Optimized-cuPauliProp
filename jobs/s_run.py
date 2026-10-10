#!/usr/bin/env python3
"""Runner of the scale campaign (queue_s, logs/s): the runner of campaign g on another queue.

  python3 jobs/s_run.py --job <job id> --hosts <host[,host...]>      works through the runs that need this many nodes
  python3 jobs/s_run.py --count                                      number of runs still open

jobs/g_run.py is used as it is (ladders, time prediction, locks, records); only its queue and log directories are
replaced here, so the file of the running campaign g is not touched. The results of the CPU implementations go to
results/g_ext like those of campaign g (their names do not collide: n8, n16, g32, g64).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_run                                    # noqa: E402

g_run.Q = os.path.join(g_run.ROOT, "queue_s")
g_run.LOGS = os.path.join(g_run.ROOT, "logs", "s")
_run_engine = g_run.run_engine


def run_engine(item, job, hosts):
    record = _run_engine(item, job, hosts)
    record["log"] = f"logs/s/{item['id']}.out"
    return record


g_run.run_engine = run_engine
if __name__ == "__main__":
    g_run.main()
