#!/bin/bash
# Campaign f on one allocation: bash jobs/f_drive.sh <allocation id>
# First every node works through the one-node runs, then pairs of nodes through the two-node runs,
# then all four nodes through the four-node runs.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
JOB=$1
Q=$PP_ROOT/queue_f
H=($(scontrol show hostnames "$(squeue -h -j $JOB -o %N)"))
echo "$(date +%m-%dT%H:%M:%S) drive job=$JOB hosts=${H[*]}" >> $Q/runner.log
[ ${#H[@]} -ge 1 ] || { echo "no hosts for job $JOB"; exit 1; }
# claims of runs whose allocation is gone and that did not finish go back to the list
for d in $Q/lock/*/; do
  id=$(basename $d); [ -e $Q/done/$id ] && continue
  owner=$(cut -d' ' -f1 $d/owner 2>/dev/null)
  if [ -z "$owner" ] || [ -z "$(squeue -h -j $owner -o %i 2>/dev/null)" ]; then rm -rf $d; fi
done
for h in "${H[@]}"; do bash jobs/f_runner.sh $JOB $h & done; wait
if [ ${#H[@]} -ge 2 ]; then
  bash jobs/f_runner.sh $JOB ${H[0]},${H[1]} &
  [ ${#H[@]} -ge 4 ] && bash jobs/f_runner.sh $JOB ${H[2]},${H[3]} &
  wait
fi
[ ${#H[@]} -ge 4 ] && bash jobs/f_runner.sh $JOB ${H[0]},${H[1]},${H[2]},${H[3]}
echo "$(date +%m-%dT%H:%M:%S) drive job=$JOB finished; done=$(ls $Q/done | wc -l) of $(grep -c . $Q/todo.txt)" >> $Q/runner.log
