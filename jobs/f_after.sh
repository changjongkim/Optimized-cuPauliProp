#!/bin/bash
# Campaign f, external baselines on one interactive allocation: bash jobs/f_after.sh <allocation id>
# One-node runs on every node in parallel, then two-node runs on pairs, then four-node runs.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
JOB=$1
Q=$PP_ROOT/queue_f
[ -f $Q/ext_todo.txt ] || exit 0
H=($(scontrol show hostnames "$(squeue -h -j $JOB -o %N)"))
[ ${#H[@]} -ge 1 ] || exit 1
mkdir -p $Q/ext_lock $Q/ext_done logs/f
echo "$(date +%m-%dT%H:%M:%S) external job=$JOB hosts=${H[*]}" >> $Q/runner.log
for d in $Q/ext_lock/*/; do
  [ -d "$d" ] || continue
  id=$(basename $d); [ -e $Q/ext_done/$id ] && continue
  owner=$(cut -d' ' -f1 $d/owner 2>/dev/null)
  if [ -z "$owner" ] || [ -z "$(squeue -h -j $owner -o %i 2>/dev/null)" ]; then rm -rf $d; fi
done
PY=$PP_ROOT/env/bin/python3
for h in "${H[@]}"; do $PY jobs/f_ext.py --job $JOB --hosts $h >> logs/f/ext-$JOB-$h.log 2>&1 & done; wait
if [ ${#H[@]} -ge 2 ]; then
  $PY jobs/f_ext.py --job $JOB --hosts ${H[0]},${H[1]} >> logs/f/ext-$JOB-pair0.log 2>&1 &
  [ ${#H[@]} -ge 4 ] && $PY jobs/f_ext.py --job $JOB --hosts ${H[2]},${H[3]} >> logs/f/ext-$JOB-pair1.log 2>&1 &
  wait
fi
[ ${#H[@]} -ge 4 ] && $PY jobs/f_ext.py --job $JOB --hosts ${H[0]},${H[1]},${H[2]},${H[3]} >> logs/f/ext-$JOB-quad.log 2>&1
echo "$(date +%m-%dT%H:%M:%S) external job=$JOB finished; done=$(ls $Q/ext_done | wc -l) of $(grep -c . $Q/ext_todo.txt)" >> $Q/runner.log
