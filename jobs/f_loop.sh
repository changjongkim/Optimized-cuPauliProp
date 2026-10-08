#!/bin/bash
# Campaign f without supervision: keeps a four-node, four-hour interactive allocation and works through
# queue_f/todo.txt (SectorQ, cuPauliProp) and queue_f/ext_todo.txt (external baselines) until both are done.
#   nohup setsid bash jobs/f_loop.sh [allocation to use first] > logs/f/loop.log 2>&1 &
# When an allocation ends, the next one is requested; while both interactive slots of the account are
# taken the request is repeated every three minutes.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
Q=$PP_ROOT/queue_f
echo $$ > $Q/loop.pid
JOB=$1
remaining() { echo $(( $(grep -c . $Q/todo.txt) - $(ls $Q/done | wc -l) + $(grep -c . $Q/ext_todo.txt 2>/dev/null || echo 0) - $(ls $Q/ext_done 2>/dev/null | wc -l) )); }
for round in $(seq 1 ${PP_F_ROUNDS:-12}); do
  before=$(remaining)
  echo "$(date +%m-%dT%H:%M:%S) round $round: $before runs open"
  [ "$before" -le 0 ] && break
  own=0
  if [ -z "$JOB" ] || [ -z "$(squeue -h -j $JOB -t RUNNING -o %i 2>/dev/null)" ]; then
    JOB=""; own=1
    for try in $(seq 1 400); do
      out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_f_int 2>&1)
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
  bash jobs/f_drive.sh $JOB
  bash jobs/f_after.sh $JOB
  after=$(remaining)
  if [ "$after" -le 0 ]; then [ $own -eq 1 ] && scancel $JOB; break; fi
  # runs are left: they did not fit in what remained of this allocation; let it go and take a fresh one
  [ $own -eq 1 ] && [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ] && scancel $JOB
  if [ "$after" -ge "$before" ] && [ $own -eq 1 ]; then echo "no progress in a fresh allocation; stopping"; break; fi
  JOB=""
done
echo "$(date +%m-%dT%H:%M:%S) loop finished: $(remaining) runs open"
