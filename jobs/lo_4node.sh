#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -J pp_lo
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Parity-owned prototype (lo_mpi) on 1, 4, 8 and 16 GPUs, same circuits as the cuPauliProp runs.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
OUT=$PP_ROOT/results/lo_$SLURM_JOB_ID
mkdir -p $OUT
cd $PP_ROOT

run() {  # run <binary> <tag> <ranks> <timeout_s> <args...>
  local bin=$1 tag=$2 ranks=$3 tmo=$4; shift 4
  local nodes=$(( (ranks + 3) / 4 ))
  echo "===== $bin $tag nodes=$nodes ranks=$ranks args: $*"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 \
    > $OUT/$tag.gpu.csv 2>/dev/null &
  local smi=$!
  timeout $tmo srun -N $nodes -n $ranks --ntasks-per-node=$(( ranks < 4 ? ranks : 4 )) --gpus-per-node=4 -c 32 --cpu-bind=cores \
    ./bin/$bin --tag $tag --outdir $OUT "$@" 2>&1 | grep -v -E '^$|gtl_init|Current device_id|IPC is disabled'
  echo "exit=${PIPESTATUS[0]}"
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
}

A="--coef 1e-4 --weight 0"
S="--coef 1e-4 --weight 8"
B="--coef 1e-5 --weight 0"

for r in 1 4 8 16; do run lo_mpi LO_A_r$r $r 200 $A; done
for r in 4 16; do run lo_mpi LO_S_r$r $r 120 $S; done
for r in 4 8 16; do run lo_mpi LO_B_r$r $r 420 $B; done
# Reference on the same nodes: cuPauliProp single GPU and its distributed mode on 4 GPUs.
export PP_EARLY_DEVICE=0
run ki_prof REF_A_r1 1 120 $A --vram-frac 0.9 --dist 0
run ki_prof REF_A_r4 4 200 $A --vram-frac 0.9
echo "done"
