#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -J pp_prof4n
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Multi-node measurement: per-gate cost and capacity at 4, 8 and 16 GPUs.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
OUT=$PP_ROOT/results/prof4n_$SLURM_JOB_ID
mkdir -p $OUT
cd $PP_ROOT
export PP_EARLY_DEVICE=0

run() {  # run <tag> <nodes> <ranks> <timeout_s> <driver args...>
  local tag=$1 nodes=$2 ranks=$3 tmo=$4; shift 4
  echo "===== $tag nodes=$nodes ranks=$ranks args: $*"
  timeout $tmo srun -N $nodes -n $ranks --ntasks-per-node=4 --gpus-per-node=4 -c 32 --cpu-bind=cores \
    ./bin/ki_prof --tag $tag --outdir $OUT "$@" 2>&1 | grep -v -E '^$'
  echo "exit=${PIPESTATUS[0]}"
}

A="--coef 1e-4 --weight 0 --vram-frac 0.9"
S="--coef 1e-4 --weight 8 --vram-frac 0.9"
B="--coef 1e-5 --weight 0 --vram-frac 0.9"

# Small expansion (NVIDIA example settings), first 1500 gates: per-gate floor vs rank count.
for r in 4 8 16; do run S_r$r $((r/4)) $r 200 $S --max-gates 1500; done
# 22.5M-term problem, full circuit.
for r in 8 16; do run A_r$r $((r/4)) $r 300 $A; done
run A_r16_kd10 4 16 300 $A --keepdup 1 --dedup-every 10
# Large problem: capacity and throughput.
for r in 8 16; do run B_r$r $((r/4)) $r 420 $B; done
run B_r16_kd10 4 16 420 $B --keepdup 1 --dedup-every 10
echo "done"
