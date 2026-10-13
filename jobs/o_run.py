#!/usr/bin/env python3
"""Runner of campaign o (queue_o, logs/o): the runner of campaign g on another queue.

  python3 jobs/o_run.py --job <allocation id> --hosts <host[,host...]>
  python3 jobs/o_run.py --count                                         number of runs still open

Campaign o runs cuPauliProp on the original circuits (jobs/o_items.py). jobs/g_run.py is used
as it is; only its queue and log directories are replaced here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_run                                    # noqa: E402

g_run.Q = os.path.join(g_run.ROOT, "queue_o")
g_run.LOGS = os.path.join(g_run.ROOT, "logs", "o")
_run_engine = g_run.run_engine


def run_engine(item, job, hosts):
    record = _run_engine(item, job, hosts)
    record["log"] = f"logs/o/{item['id']}.out"
    return record


g_run.run_engine = run_engine
if __name__ == "__main__":
    g_run.main()
