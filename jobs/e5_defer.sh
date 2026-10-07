#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e5
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Deferred merging inside a bucket (--defer-dedup k): equal strings are merged every k operations
# and before each truncation instead of after every operation. One node, 80 GB GPUs limited to 36 GB.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_dd}
eng() { local tmo=$1; shift; echo "===== engine nodes=1 gpus=4 args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N 1 -n 1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
for d in 0 10 5 3 2; do eng 120 $KI --coef 1e-5 --target 6000000 $V --defer-dedup $d; done
for d in 0 10 3; do eng 150 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000 $V --defer-dedup $d; done
for d in 0 10; do eng 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V --defer-dedup $d; done
for d in 0 10 3; do eng 120 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000 $V --defer-dedup $d; done
for d in 10 0; do eng 250 $KI --coef 1e-5 --target 6000000 $V --defer-dedup $d --gpus 1; done
echo "done"
