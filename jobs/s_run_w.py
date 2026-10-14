#!/usr/bin/env python3
"""The runner of the scale campaign (jobs/s_run.py) with a watch on stalled runs.

  python3 jobs/s_run_w.py --job <job id> --hosts <host[,host...]>

Job 59660400 (256 GPUs) lost two hours on one run: SectorQ at cutoff 5e-7 printed windows 0-10 at about 20 s each
and then nothing until its time limit, and the log does not tell where it stopped. Here a thread watches the log of
every engine run. When no line has come for max(STALL_MIN, STALL_FACTOR x the longest gap so far), it writes the
stacks of every engine process on every node (gstack) to logs/s/<id>.stacks.txt and stops the run; the run is then
started once more (a stall that does not come back is not a property of the problem). The log of the stalled
attempt is kept as logs/s/<id>.stall<k>.out, and the record says what happened (stalls, stall_logs, stacks).
"""
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_run                                    # noqa: E402  (sets the queue and log directories of the scale campaign)
import g_run                                    # noqa: E402

STALL_MIN = 900          # seconds without a new line before a run counts as stalled
STALL_FACTOR = 5         # ... or this many times the longest gap between two lines of the same run
TRIES = 2                # attempts of a run that stalls
POLL = 20                # seconds between two looks at the log

_run_engine = g_run.run_engine


def _children(name):
    """Processes `name` started by this runner (the `timeout` wrapper of srun)."""
    out = subprocess.run(["pgrep", "-P", str(os.getpid()), "-x", name], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def _stacks(item, job, hosts, path):
    nn = len(hosts.split(","))
    probe = ('echo "== $(hostname) $(date +%H:%M:%S)"; for p in $(pgrep -u $USER -f "bin/pp_engine"); do '
             'echo "-- pid $p"; timeout 120 gstack $p 2>&1 | head -400; done')
    cmd = g_run.srun_prefix(job, hosts, nn, 1, 1, "none") + ["timeout", "300", "bash", "-c", probe]   # as the GPU sampler of g_run
    with open(path, "w") as f:
        f.write(f"stacks of {item['id']} at {time.strftime('%m-%dT%H:%M:%S')}\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=420)


def _watch(item, job, hosts, state, stop):
    log = os.path.join(g_run.LOGS, item["id"] + ".out")
    last_size, last_change, gap = -1, time.time(), 0.0
    while not stop.wait(POLL):
        try:
            size = os.path.getsize(log)
        except OSError:
            continue
        now = time.time()
        if size != last_size:
            if last_size >= 0:
                gap = max(gap, now - last_change)
            last_size, last_change = size, now
            continue
        quiet = now - last_change
        if quiet < max(STALL_MIN, STALL_FACTOR * gap):
            continue
        state["stalled"] = round(quiet)
        stacks = os.path.join(g_run.LOGS, item["id"] + ".stacks.txt")
        try:
            _stacks(item, job, hosts, stacks)
            state["stacks"] = os.path.relpath(stacks, g_run.ROOT)
        except Exception as e:                  # the stop below still happens
            state["stacks_error"] = repr(e)[:200]
        state["note"] = f"stall: no output for {round(quiet)} s (longest earlier gap {round(gap)} s); stacks in {state.get('stacks')}; stopped"
        for pid in _children("timeout"):
            os.kill(pid, signal.SIGTERM)
        return


def run_engine(item, job, hosts):
    if item["kind"] != "eng":
        return _run_engine(item, job, hosts)
    stalls, stall_logs, stacks = [], [], []
    for attempt in range(1, TRIES + 1):
        state, stop = {}, threading.Event()
        watcher = threading.Thread(target=_watch, args=(item, job, hosts, state, stop), daemon=True)
        watcher.start()
        record = _run_engine(item, job, hosts)
        stop.set()
        watcher.join(timeout=600)
        if "stalled" not in state:
            break
        with open(os.path.join(g_run.LOGS, item["id"] + ".out"), "a") as f:    # after the runner has closed the log
            f.write(state["note"] + "\n")
        stalls.append(state["stalled"])
        if state.get("stacks"):
            moved = state["stacks"].replace(".stacks.txt", f".stacks{attempt}.txt")
            os.replace(os.path.join(g_run.ROOT, state["stacks"]), os.path.join(g_run.ROOT, moved))
            stacks.append(moved)
        kept = os.path.join(g_run.LOGS, f"{item['id']}.stall{attempt}.out")
        os.replace(os.path.join(g_run.LOGS, item["id"] + ".out"), kept)
        stall_logs.append(os.path.relpath(kept, g_run.ROOT))
        record = dict(record, status="stall", log=stall_logs[-1])
        if g_run.left(job) < item["limit"] + 300:           # no time for another attempt in this allocation
            break
    if stalls:
        record.update(stalls=stalls, stall_logs=stall_logs, stacks=stacks)
    return record


g_run.run_engine = run_engine
if __name__ == "__main__":
    g_run.main()
