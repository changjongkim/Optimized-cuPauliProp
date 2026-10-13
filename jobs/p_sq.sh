#!/bin/bash
# SectorQ with the truncation rule of paulistrings-rs (a truncation after every operation, --trunc-every 1) on the
# same kicked Ising circuit, so that the two run the same computation: logs/p/sq.e1.g<GPUs>.<cutoff>.out.
#   nohup setsid bash jobs/p_sq.sh > logs/p/sq_loop.log 2>&1 &
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export MPICH_GPU_SUPPORT_ENABLED=0
JOB=""
for try in $(seq 1 200); do
  out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 02:30:00 --gpus-per-node=4 --no-shell -J pp_p_sq 2>&1)
  JOB=$(echo "$out" | grep -o 'Granted job allocation [0-9]*' | awk '{print $4}')
  [ -n "$JOB" ] && break
  echo "$(date +%m-%dT%H:%M:%S) no allocation yet: $(echo "$out" | tail -1)"; sleep 180
done
[ -z "$JOB" ] && { echo "no allocation"; exit 1; }
for w in $(seq 1 60); do [ -n "$(squeue -h -j $JOB -t RUNNING -o %i 2>/dev/null)" ] && break; sleep 10; done
HOSTS=$(scontrol show hostnames $(squeue -h -j $JOB -o %N) | paste -sd,)
FIRST=${HOSTS%%,*}
echo "$(date +%m-%dT%H:%M:%S) allocation $JOB on $HOSTS"
run() {   # GPUs, nodes, gpus per process, cutoff
  local g=$1 n=$2 per=$3 x=$4 log=logs/p/sq.e1.g$1.$4.out hosts=$HOSTS
  [ -s $log ] && grep -q "^exit=" $log && return
  [ $n -eq 1 ] && hosts=$FIRST
  echo "===== SectorQ, truncation after every operation, $g GPUs, cutoff $x" > $log
  timeout 2400 srun --jobid=$JOB --nodelist=$hosts --overlap -N $n -n $n --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_i --gpus $per --circuit ki_lab --weight 0 --coef $x --target 6000000 --trace 0 --vram-frac 0.9 --trunc-every 1 >> $log 2>&1
  echo "exit=$?" >> $log
  echo "$(date +%m-%dT%H:%M:%S) $g GPUs $x: $(grep -m1 '^expectation=' $log) $(grep -o '^wall=[0-9.]* s' $log | head -1) $(tail -1 $log)"
}
for x in 1e-3 1e-4 3e-5 1e-5 3e-6 1e-6; do run 16 4 4 $x; done
for x in 1e-4 3e-5 1e-5 3e-6; do run 4 1 4 $x; done
for x in 1e-4 3e-5 1e-5; do run 1 1 1 $x; done
scancel $JOB
echo "$(date +%m-%dT%H:%M:%S) finished"
