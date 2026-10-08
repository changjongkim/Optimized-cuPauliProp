#!/bin/bash
# Campaign f runner: executes the lines of queue_f/todo.txt that need as many nodes as it was given.
#   bash jobs/f_runner.sh <allocation id> <host[,host...]>
# Several runners may work on the same list (one per node, or per pair of nodes): a run is claimed with
# mkdir queue_f/lock/<id>, and a finished run leaves queue_f/done/<id> and logs/f/<id>.out.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
JOB=$1; HOSTS=$2
NN=$(echo "$HOSTS" | tr ',' '\n' | grep -c .)
Q=$PP_ROOT/queue_f; OUT=$PP_ROOT/logs/f
BIN=${PP_BIN:-bin/pp_engine_f}
# seconds left in the allocation
left() { local t; t=$(squeue -h -j $JOB -o %L 2>/dev/null); [ -z "$t" ] && { echo 0; return; }
  local d=0; if [[ $t == *-* ]]; then d=${t%%-*}; t=${t#*-}; fi
  IFS=: read -r a b c <<< "$t"; if [ -z "$c" ]; then c=$b; b=$a; a=0; fi
  echo $((10#$d * 86400 + 10#$a * 3600 + 10#$b * 60 + 10#$c)); }
while IFS='|' read -r -u 3 id nodes per kind tmo args; do
  [ "$nodes" = "$NN" ] || continue
  [ -e $Q/done/$id ] && continue
  l=$(left)
  ((l < tmo + 90)) && continue                       # does not fit in what is left of the allocation
  mkdir $Q/lock/$id 2>/dev/null || continue           # another runner has it
  echo "$JOB $HOSTS" > $Q/lock/$id/owner
  t0=$SECONDS
  {
    case $kind in
      eng)
        echo "===== engine nodes=$NN gpus=$((NN * per)) args: $args --gpus $per"
        MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun --jobid=$JOB --nodelist=$HOSTS -N $NN -n $NN --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
          stdbuf -oL ./$BIN --gpus $per $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)"
        echo "exit=${PIPESTATUS[0]}" ;;
      ref)
        echo "===== ref nodes=1 gpus=1 args: $args"
        MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun --jobid=$JOB --nodelist=$HOSTS -N 1 -n 1 --ntasks-per-node=1 -c 32 --gpus-per-node=4 --cpu-bind=none \
          stdbuf -oL ./$BIN --mode ref $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)" | awk '!seen[$0]++' | head -40
        echo "exit=${PIPESTATUS[0]}" ;;
      cupp)
        echo "===== cupp nodes=$NN gpus=$((NN * per)) args: $args"
        CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun --jobid=$JOB --nodelist=$HOSTS -N $NN -n $((NN * per)) --ntasks-per-node=$per -c $((128 / per)) --gpus-per-node=4 --cpu-bind=cores \
          stdbuf -oL ./$BIN --mode cupp $args < /dev/null 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk '!seen[$0]++' | head -40
        echo "exit=${PIPESTATUS[0]}" ;;
      *) echo "unknown kind $kind"; echo "exit=99" ;;
    esac
  } > $OUT/$id.out 2>&1
  touch $Q/done/$id
  echo "$(date +%m-%dT%H:%M:%S) job=$JOB hosts=$HOSTS id=$id seconds=$((SECONDS - t0)) $(grep -m1 '^exit=' $OUT/$id.out) $(grep -m1 -E '^wall=[0-9.]+' $OUT/$id.out | cut -d' ' -f1)" >> $Q/runner.log
done 3< $Q/todo.txt
