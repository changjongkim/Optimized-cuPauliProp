#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -J pp_prof1
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# First measurement of cuPauliProp's distributed mode on one node (4 x A100).
# Each step logs per-gate, per-rank timings and MPI activity; nvidia-smi samples GPU utilization.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
OUT=$PP_ROOT/results/prof1_$SLURM_JOB_ID
mkdir -p $OUT
cd $PP_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {  # run <tag> <ranks> <timeout_s> <driver args...>
  local tag=$1 ranks=$2 tmo=$3; shift 3
  echo "===== $tag ranks=$ranks args: $*"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 \
    > $OUT/$tag.gpu.csv 2>/dev/null &
  local smi=$!
  timeout $tmo srun -n $ranks --gpus-per-node=4 -c 32 --cpu-bind=cores \
    ./bin/ki_prof --tag $tag --outdir $OUT "$@" 2>&1 | grep -v -E '^$'
  echo "exit=${PIPESTATUS[0]}"
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
}

# Config A: coefficient cutoff 1e-4, no weight truncation (22.5M terms on one GPU).
A="--coef 1e-4 --weight 0 --vram-frac 0.9"
run A_r1_single 1 300 $A --dist 0
run A_r1_dist   1 300 $A
run A_r2        2 300 $A
run A_r4        4 300 $A
run A_r4_kd10   4 300 $A --keepdup 1 --dedup-every 10
run A_r4_kd1    4 300 $A --keepdup 1 --dedup-every 1

# NVIDIA's example configuration (cutoff 1e-4, weight 8): a small expansion on 4 ranks.
S="--coef 1e-4 --weight 8 --vram-frac 0.9"
run S_r1_single 1 120 $S --dist 0
run S_r4        4 120 $S

# Config B: coefficient cutoff 1e-5, no weight truncation (exceeds one GPU at 50% memory).
B="--coef 1e-5 --weight 0 --vram-frac 0.9"
run B_r1_single 1 420 $B --dist 0
run B_r4        4 600 $B
echo "done"
