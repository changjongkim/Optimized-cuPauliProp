#!/bin/bash
# The paulistrings-rs baseline (jobs/p_drive.py) without supervision: keeps a four-node,
# four-hour interactive allocation and works through
# the list of jobs/p_drive.py until every run is recorded. One loop only: the account may hold two interactive allocations and
# the other one is left to the other users of the account.
#   nohup setsid bash jobs/p_loop.sh a > logs/p/loop_a.log 2>&1 &
# When an allocation ends, the next one is requested; while both interactive slots are taken the request is
# repeated every three minutes.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
NAME=$1; JOB=$2
Q=$PP_ROOT/queue_p
PY=$PP_ROOT/env/bin/python3
echo $$ > $Q/loop_$NAME.pid
for round in $(seq 1 ${PP_P_ROUNDS:-10}); do
  before=$($PY jobs/p_drive.py --count)
  echo "$(date +%m-%dT%H:%M:%S) round $round: $before runs open"
  [ "$before" -le 0 ] && break
  [ -e $Q/stop ] && { echo "stop file found"; break; }
  own=0
  if [ -z "$JOB" ] || [ -z "$(squeue -h -j $JOB -t RUNNING -o %i 2>/dev/null)" ]; then
    JOB=""; own=1
    for try in $(seq 1 400); do
      out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_p_$NAME 2>&1)
      JOB=$(echo "$out" | grep -o 'Granted job allocation [0-9]*' | awk '{print $4}')
      [ -n "$JOB" ] && break
      echo "$out" | grep -q -E 'QOSMaxSubmitJobPerUserLimit|QOSMaxJobsPerUserLimit' || { echo "$out"; sleep 600; continue; }
      [ $((try % 10)) -eq 1 ] && echo "$(date +%m-%dT%H:%M:%S) both interactive slots of the account are taken; waiting"
      sleep 180
    done
    [ -z "$JOB" ] && { echo "no allocation"; exit 1; }
    for w in $(seq 1 60); do [ -n "$(squeue -h -j $JOB -t RUNNING -o %i 2>/dev/null)" ] && break; sleep 10; done
  fi
  echo "$(date +%m-%dT%H:%M:%S) allocation $JOB"
  echo $JOB > $Q/job_$NAME
  # the one-node ladders side by side, one node each; then the ladders that take all four nodes, the GPUs first
  $PY jobs/p_drive.py --job $JOB --modes h1,h1u,g1 --circuits ki,tfim,noise,tfimsq --node 0 &
  $PY jobs/p_drive.py --job $JOB --modes h1c,g1 --circuits ki,xxz,qaoa,hea --node 1 &
  $PY jobs/p_drive.py --job $JOB --modes g4 --circuits ki,tfim,noise,tfimsq --node 2 &
  $PY jobs/p_drive.py --job $JOB --modes g4c,g4 --circuits ki,xxz,qaoa,hea --node 3 &
  wait
  $PY jobs/p_drive.py --job $JOB --modes mg16,mg16c
  $PY jobs/p_drive.py --job $JOB --modes m16,m16c
  after=$($PY jobs/p_drive.py --count)
  if [ "$after" -le 0 ]; then [ $own -eq 1 ] && scancel $JOB; break; fi
  # runs are left: they did not fit in what remained of this allocation; let it go and take a fresh one
  [ $own -eq 1 ] && [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ] && scancel $JOB
  if [ "$after" -ge "$before" ] && [ $own -eq 1 ]; then echo "no progress in a fresh allocation; stopping"; break; fi
  JOB=""
done
echo "$(date +%m-%dT%H:%M:%S) loop finished: $($PY jobs/p_drive.py --count) runs open"
