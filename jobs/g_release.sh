#!/bin/bash
# Campaign g: puts the lists waiting in queue_g/hold into the queue as soon as the runners that are working now have
# ended (the driver writes a line to queue_g/drive.log at the end of every round and at the start of an allocation).
# Used for runs that need a runner newer than the one in memory (jobs/g_wide_items.py).
#   ( nohup setsid bash jobs/g_release.sh > logs/g/release.log 2>&1 < /dev/null & )      (on the login node of the loop)
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
Q=$PP_ROOT/queue_g
n0=$(wc -l < $Q/drive.log)
echo "$(date +%m-%dT%H:%M:%S) waiting: drive.log has $n0 lines; holding $(ls $Q/hold/*.jsonl 2>/dev/null | wc -l) lists"
while [ "$(wc -l < $Q/drive.log)" -le "$n0" ]; do sleep 15; done
ls $Q/hold/*.jsonl > /dev/null 2>&1 || { echo "nothing waiting"; exit 0; }
mv $Q/hold/*.jsonl $Q/
echo "$(date +%m-%dT%H:%M:%S) lists released: $(tail -1 $Q/drive.log); $($PP_ROOT/env/bin/python3 jobs/g_run.py --count) runs open"
