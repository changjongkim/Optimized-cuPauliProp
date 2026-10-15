#!/bin/bash
# Continues the ablation of jobs/x_ablate.sh on a new interactive allocation once the running loop has ended.
#   nohup setsid bash jobs/x_continue.sh <pid of the running loop> <its job id> >> logs/x/loop.log 2>&1 &
# 1. waits until the running loop exits (its allocation ends at the time limit)
# 2. moves aside the runs that failed because the allocation ended (non-zero exit, written at or after the end of the
#    job) and their stop markers, so that the new loop runs them again; earlier failures (out of memory) stay
# 3. starts one new loop (x_ablate.sh asks for a new interactive allocation of 4 nodes); runs with a log that ends in
#    "exit=" are skipped there, so only the remaining ones run (Ansatz on 16 GPUs, then the 4-GPU ladders)
cd /pscratch/sd/s/sgkim/kcj/pauliprop
LOOP=$1 JOB=$2
while kill -0 "$LOOP" 2>/dev/null; do sleep 60; done
sleep 30
end=$(date -d "$(sacct -j "$JOB" -X -n -o End | head -1 | tr -d ' ')" +%s 2>/dev/null || date +%s)
mkdir -p logs/x/alloc_end
moved=0
for f in logs/x/*.out logs/x/stop.*; do
  [ -e "$f" ] || continue
  [ "$(stat -c %Y "$f")" -ge $((end - 120)) ] || continue
  case "$f" in
    *.out) grep -q "^exit=0$" "$f" && continue ;;
  esac
  mv "$f" logs/x/alloc_end/ && moved=$((moved + 1))
done
echo "$(date +%m-%dT%H:%M:%S) continue: allocation $JOB ended $(date -d @$end +%H:%M:%S); moved $moved logs and stop markers written after it to logs/x/alloc_end; starting one new loop"
exec bash jobs/x_ablate.sh
