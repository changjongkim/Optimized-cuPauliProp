#!/bin/bash
# Ablation: rows fixed for the whole run (--fixed-rows 1) against rows recomputed for every window (the default), with the
# same engine (bin/pp_engine_x), on the circuits of the evaluation. Logs: logs/x/<mode>.<circuit>.g<GPUs>.<cutoff>[.t1].out
#   nohup setsid bash jobs/x_ablate.sh > logs/x/loop.log 2>&1 &
# 1. check: a truncation after every operation (--trunc-every 1) on one node, where both modes must give the same value
# 2. 16 GPUs (4 nodes) down the cutoff ladder of each circuit, both modes (a ladder stops at its first failure)
# 3. 4 GPUs (1 node), the same
# Noisy kicked Ising is left out: its two-qubit Clifford gates change the bits of the terms between branching rotations,
# which the fixed-row mode does not handle.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export MPICH_GPU_SUPPORT_ENABLED=0
mkdir -p logs/x
JOB="${PP_X_JOB:-}"                               # an allocation that is already running, or a new one
for try in $(seq 1 200); do
  [ -n "$JOB" ] && break
  out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_x 2>&1)
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
  [tfimsq]="--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000"
  [xxz]="--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000"
  [qaoa]="--circuit qaoa --n 100 --layers 4 --target 4000000"
  [hea]="--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000"
)
run() {   # mode (re|fx), circuit, GPUs, cutoff, extra tag, extra arguments
  local m=$1 c=$2 g=$3 x=$4 tag=$5; shift 5
  local n=$((g / 4)) hosts=$HOSTS
  [ $n -eq 1 ] && hosts=$FIRST
  local log=logs/x/$m.$c.g$g.$x$tag.out
  [ -s $log ] && grep -q "^exit=" $log && return
  [ -e logs/x/stop.$m.$c.g$g$tag ] && return
  local mode=""; [ $m = fx ] && mode="--fixed-rows 1"
  echo "===== $m $c $g GPUs cutoff $x $mode $*" > $log
  timeout 2400 srun --jobid=$JOB --nodelist=$hosts --overlap -N $n -n $n --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_x --gpus 4 ${CIRC[$c]} --coef $x --trace 1 --vram-frac 0.9 $mode "$@" >> $log 2>&1
  local code=$?
  echo "exit=$code" >> $log
  [ $code -ne 0 ] && touch logs/x/stop.$m.$c.g$g$tag
  echo "$(date +%m-%dT%H:%M:%S) $m $c $g GPUs $x$tag: $(grep -m1 '^expectation=' $log) $(grep -o '^wall=[0-9.]* s' $log | head -1) $(grep -m1 -o '^windows=[0-9]*' $log) exit=$code"
}
# kicked Ising and TFIM heavy-hex stop at 3e-6: with fixed rows kicked Ising takes 1,095 s there (9.3 times the default),
# so 1e-6 would exceed the 40 minutes of a run; the default mode at 1e-6 is in campaign r (1,518 s on 16 GPUs)
declare -A LADDER=([ki]="1e-4 3e-5 1e-5 3e-6" [tfim]="1e-4 3e-5 1e-5 3e-6" [tfimsq]="1e-4 3e-5 1e-5 3e-6"
                   [xxz]="1e-4 3e-5 1e-5 3e-6" [qaoa]="1e-4 3e-5 1e-5 3e-6 1e-6" [hea]="1e-4 3e-5 1e-5 3e-6 1e-6")
# 1. check of the answer (one node, a truncation after every operation)
for c in ki tfim tfimsq xxz qaoa hea; do for m in re fx; do run $m $c 4 1e-4 .t1 --trunc-every 1; done; done
# 2. 16 GPUs, 3. 4 GPUs
for g in 16 4; do for c in ki tfim tfimsq xxz qaoa hea; do for x in ${LADDER[$c]}; do for m in re fx; do run $m $c $g $x ""; done; done; done; done
scancel $JOB
echo "$(date +%m-%dT%H:%M:%S) finished"
