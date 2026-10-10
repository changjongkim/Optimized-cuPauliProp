#!/bin/bash
# Campaign g on one interactive allocation: when loop b (jobs/g_loop.sh b down) has ended, the lists waiting in
# queue_g/later are put into the queue and loop b is started again. It takes the place of g_chain.sh, which starts two
# loops, from 2026-10-10 on: the second interactive allocation of the account is left free.
#   nohup setsid bash jobs/g_chain_one.sh > logs/g/chain_one.log 2>&1 &      (on the login node of the loop)
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
Q=$PP_ROOT/queue_g
echo $$ > $Q/chain.pid
# the loop is alive while its process exists and is still that script (ps can hang on the login nodes: read /proc)
alive() { local p; p=$(cat $Q/loop_$1.pid 2>/dev/null); [ -n "$p" ] && [ -r /proc/$p/cmdline ] && tr '\0' ' ' < /proc/$p/cmdline | grep -q "g_loop.sh $1"; }
while alive b; do sleep 120; done
echo "$(date +%m-%dT%H:%M:%S) loop b has ended; $($PP_ROOT/env/bin/python3 jobs/g_run.py --count) runs open"
[ -e $Q/stop ] && { echo "stop file found"; exit 0; }
ls $Q/later/*.jsonl > /dev/null 2>&1 || { echo "nothing waiting"; exit 0; }
mv $Q/later/*.jsonl $Q/
echo "$(date +%m-%dT%H:%M:%S) lists added; $($PP_ROOT/env/bin/python3 jobs/g_run.py --count) runs open"
( nohup setsid bash jobs/g_loop.sh b down > logs/g/loop_b2.log 2>&1 < /dev/null & )
echo "$(date +%m-%dT%H:%M:%S) loop b started again"
