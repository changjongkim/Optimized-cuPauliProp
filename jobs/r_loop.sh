#!/bin/bash
# Campaign r (the SectorQ runs of campaign g again with bin/pp_engine_i) without supervision: keeps a four-node,
# four-hour interactive allocation and works through
# queue_r/todo*.jsonl until every run is done. One loop only: the account may hold two interactive allocations and
# the other one is left to the other users of the account.
#   nohup setsid bash jobs/r_loop.sh a down > logs/r/loop_a.log 2>&1 &
# When an allocation ends, the next one is requested; while both interactive slots are taken the request is
# repeated every three minutes.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
NAME=$1; ORDER=$2; JOB=$3
Q=$PP_ROOT/queue_r
PY=$PP_ROOT/env/bin/python3
echo $$ > $Q/loop_$NAME.pid
for round in $(seq 1 ${PP_R_ROUNDS:-10}); do
  before=$($PY jobs/r_run.py --count)
  echo "$(date +%m-%dT%H:%M:%S) round $round: $before runs open"
  [ "$before" -le 0 ] && break
  [ -e $Q/stop ] && { echo "stop file found"; break; }
  own=0
  if [ -z "$JOB" ] || [ -z "$(squeue -h -j $JOB -t RUNNING -o %i 2>/dev/null)" ]; then
    JOB=""; own=1
    for try in $(seq 1 400); do
      out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_r_$NAME 2>&1)
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
  $PY jobs/r_drive.py --job $JOB --order $ORDER
  after=$($PY jobs/r_run.py --count)
  if [ "$after" -le 0 ]; then [ $own -eq 1 ] && scancel $JOB; break; fi
  # runs are left: they did not fit in what remained of this allocation; let it go and take a fresh one
  [ $own -eq 1 ] && [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ] && scancel $JOB
  if [ "$after" -ge "$before" ] && [ $own -eq 1 ]; then echo "no progress in a fresh allocation; stopping"; break; fi
  JOB=""
done
echo "$(date +%m-%dT%H:%M:%S) loop finished: $($PY jobs/r_run.py --count) runs open"
