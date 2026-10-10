#!/bin/bash
# Campaign g, chain: when both loops (jobs/g_loop.sh a, b) have ended, the lists waiting in queue_g/later are
# put into the queue and the two loops are started again. This keeps those runs behind everything else.
#   nohup setsid bash jobs/g_chain.sh > logs/g/chain.log 2>&1 &
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
Q=$PP_ROOT/queue_g
echo $$ > $Q/chain.pid
# a loop is alive while its process exists and is still that script (ps can hang on the login nodes: read /proc)
alive() { local p; p=$(cat $Q/loop_$1.pid 2>/dev/null); [ -n "$p" ] && [ -r /proc/$p/cmdline ] && tr '\0' ' ' < /proc/$p/cmdline | grep -q "g_loop.sh $1"; }
while alive a || alive b; do sleep 120; done
echo "$(date +%m-%dT%H:%M:%S) both loops have ended; $($PP_ROOT/env/bin/python3 jobs/g_run.py --count) runs open"
[ -e $Q/stop ] && { echo "stop file found"; exit 0; }
ls $Q/later/*.jsonl > /dev/null 2>&1 || { echo "nothing waiting"; exit 0; }
mv $Q/later/*.jsonl $Q/
echo "$(date +%m-%dT%H:%M:%S) lists added; $($PP_ROOT/env/bin/python3 jobs/g_run.py --count) runs open"
( nohup setsid bash jobs/g_loop.sh a up > logs/g/loop_a2.log 2>&1 < /dev/null & )
sleep 30
( nohup setsid bash jobs/g_loop.sh b down > logs/g/loop_b2.log 2>&1 < /dev/null & )
echo "$(date +%m-%dT%H:%M:%S) loops started again"
