#!/bin/bash
# Campaign f in interactive allocations (four nodes, four hours), one after another until the list is done:
#   nohup setsid bash jobs/f_interactive.sh > logs/f/interactive.log 2>&1 &
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
Q=$PP_ROOT/queue_f
MAX=${PP_F_ALLOCS:-3}
for round in $(seq 1 $MAX); do
  total=$(grep -c . $Q/todo.txt); fin=$(ls $Q/done | wc -l)
  echo "$(date +%m-%dT%H:%M:%S) round $round: $fin of $total done"
  [ "$fin" -ge "$total" ] && break
  # The account may hold two interactive allocations at a time. While both are taken the request is refused,
  # so it is repeated every three minutes (for up to PP_F_WAIT_H hours).
  JOB=""
  for try in $(seq 1 $(( ${PP_F_WAIT_H:-10} * 20 ))); do
    out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_f_int 2>&1)
    JOB=$(echo "$out" | grep -o 'Granted job allocation [0-9]*' | awk '{print $4}')
    [ -n "$JOB" ] && break
    echo "$out" | grep -q -E 'QOSMaxSubmitJobPerUserLimit|QOSMaxJobsPerUserLimit' || { echo "$out"; break; }
    [ $((try % 10)) -eq 1 ] && echo "$(date +%m-%dT%H:%M:%S) both interactive slots of the account are taken; waiting"
    sleep 180
    [ "$(ls $Q/done | wc -l)" -ge "$(grep -c . $Q/todo.txt)" ] && [ ! -f jobs/f_after.sh ] && break
  done
  [ -z "$JOB" ] && { echo "no allocation"; exit 1; }
  echo "$(date +%m-%dT%H:%M:%S) allocation $JOB"
  bash jobs/f_drive.sh $JOB
  [ -f jobs/f_after.sh ] && bash jobs/f_after.sh $JOB        # further work in the same allocation (external baselines)
  [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ] && [ ! -e $Q/keep_allocation ] && scancel $JOB
  now=$(ls $Q/done | wc -l)
  [ "$now" -le "$fin" ] && { echo "no progress in this allocation; stopping"; break; }
done
echo "$(date +%m-%dT%H:%M:%S) interactive loop finished: $(ls $Q/done | wc -l) of $(grep -c . $Q/todo.txt) done"
