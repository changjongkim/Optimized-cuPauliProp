#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -J pp_lofr
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Parity-owned prototype with the Clifford frame (no ZZ gate touches data) and sampled row
# selection, against cuPauliProp given the same frame circuit.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
OUT=$PP_ROOT/results/lofr_$SLURM_JOB_ID
mkdir -p $OUT
cd $PP_ROOT

run() {  # run <binary> <tag> <ranks> <timeout_s> <args...>
  local bin=$1 tag=$2 ranks=$3 tmo=$4; shift 4
  local nodes=$(( (ranks + 3) / 4 ))
  echo "===== $bin $tag nodes=$nodes ranks=$ranks args: $*"
  timeout $tmo srun -N $nodes -n $ranks --ntasks-per-node=$(( ranks < 4 ? ranks : 4 )) --gpus-per-node=4 -c 32 --cpu-bind=cores \
    ./bin/$bin --tag $tag --outdir $OUT "$@" 2>&1 | grep -v -E '^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG'
  echo "exit=${PIPESTATUS[0]}"
}

A="--coef 1e-4 --weight 0"
B="--coef 1e-5 --weight 0"
C="--coef 3e-6 --weight 0"

for r in 1 4 8 16; do run lo_mpi FR_A_r$r $r 200 $A --frame 1; done
for r in 4 16; do run lo_mpi LAB_A_r$r $r 200 $A --frame 0; done
for r in 4 8 16; do run lo_mpi FR_B_r$r $r 300 $B --frame 1; done
run lo_mpi FR_C_r16 16 300 $C --frame 1
# cuPauliProp given the same frame circuit: one GPU, and its distributed mode on 4 and 16 GPUs.
export PP_EARLY_DEVICE=0
run ki_prof REF_FR_A_r1 1 120 $A --vram-frac 0.9 --dist 0 --frame 1
run ki_prof REF_FR_A_r4 4 200 $A --vram-frac 0.9 --frame 1
run ki_prof REF_FR_A_r16 16 200 $A --vram-frac 0.9 --frame 1
run ki_prof REF_FR_B_r16 16 300 $B --vram-frac 0.9 --frame 1
echo "done"
