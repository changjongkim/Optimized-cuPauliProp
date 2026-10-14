#!/bin/bash
# SectorQ with the truncation rule of paulistrings-rs (a truncation after every operation, --trunc-every 1) on the
# same seven circuits, so that the two run the same computation: logs/p/sq.e1[.<circuit>].g<GPUs>.<cutoff>.out.
#   nohup setsid bash jobs/p_sq.sh > logs/p/sq_loop.log 2>&1 &
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export MPICH_GPU_SUPPORT_ENABLED=0
JOB=""
for try in $(seq 1 200); do
  out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_p_sq 2>&1)
  JOB=$(echo "$out" | grep -o 'Granted job allocation [0-9]*' | awk '{print $4}')
  [ -n "$JOB" ] && break
  echo "$(date +%m-%dT%H:%M:%S) no allocation yet: $(echo "$out" | tail -1)"; sleep 180
done
[ -z "$JOB" ] && { echo "no allocation"; exit 1; }
for w in $(seq 1 60); do [ -n "$(squeue -h -j $JOB -t RUNNING -o %i 2>/dev/null)" ] && break; sleep 10; done
HOSTS=$(scontrol show hostnames $(squeue -h -j $JOB -o %N) | paste -sd,)
FIRST=${HOSTS%%,*}
echo "$(date +%m-%dT%H:%M:%S) allocation $JOB on $HOSTS"
declare -A CIRC=(
  [ki]="--circuit ki_lab --weight 0 --target 6000000"
  [tfim]="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000"
  [noise]="--circuit ki_noisy --noise 0.005 --target 6000000"
  [tfimsq]="--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000"
  [xxz]="--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000"
  [qaoa]="--circuit qaoa --n 100 --layers 4 --target 4000000"
  [hea]="--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000"
)
run() {   # circuit, GPUs, nodes, gpus per process, cutoff; kicked Ising keeps the log names of the first runs
  local c=$1 g=$2 n=$3 per=$4 x=$5 hosts=$HOSTS
  local log=logs/p/sq.e1.$c.g$g.$x.out
  [ $c = ki ] && log=logs/p/sq.e1.g$g.$x.out
  [ -s $log ] && grep -q "^exit=" $log && return
  [ -e logs/p/sq.stop.$c.g$g ] && return          # a ladder stops at its first run that does not finish
  [ $n -eq 1 ] && hosts=$FIRST
  echo "===== SectorQ, truncation after every operation, $c, $g GPUs, cutoff $x" > $log
  timeout 2400 srun --jobid=$JOB --nodelist=$hosts --overlap -N $n -n $n --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_i --gpus $per ${CIRC[$c]} --coef $x --trace 0 --vram-frac 0.9 --trunc-every 1 >> $log 2>&1
  local code=$?
  echo "exit=$code" >> $log
  [ $code -ne 0 ] && touch logs/p/sq.stop.$c.g$g
  echo "$(date +%m-%dT%H:%M:%S) $c $g GPUs $x: $(grep -m1 '^expectation=' $log) $(grep -o '^wall=[0-9.]* s' $log | head -1) exit=$code"
}
declare -A LADDER=([ki]="1e-3 1e-4 3e-5 1e-5 3e-6 1e-6" [tfim]="1e-4 3e-5 1e-5 3e-6 1e-6" [noise]="1e-4 3e-5 1e-5 3e-6 1e-6" [tfimsq]="1e-4 3e-5 1e-5 3e-6"
                   [xxz]="1e-4 3e-5 1e-5 3e-6" [qaoa]="1e-4 3e-5 1e-5 3e-6 1e-6" [hea]="1e-4 3e-5 1e-5 3e-6 1e-6 3e-7")
for c in ki tfim noise tfimsq xxz qaoa hea; do for x in ${LADDER[$c]}; do run $c 16 4 4 $x; done; done
for c in ki tfim noise tfimsq xxz qaoa hea; do for x in ${LADDER[$c]}; do [ $x = 1e-6 ] || [ $x = 3e-7 ] || run $c 4 1 4 $x; done; done
for c in ki tfim noise tfimsq xxz qaoa hea; do for x in ${LADDER[$c]}; do case $x in 1e-3|1e-4|3e-5|1e-5) run $c 1 1 1 $x;; esac; done; done
scancel $JOB
echo "$(date +%m-%dT%H:%M:%S) finished"
