#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e6
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Where the engine's wall time goes on 16 and 32 GPUs (per-node accounting), with and without
# deferred merging inside a bucket. 80 GB GPUs limited to 36 GB.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_dd2}
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
for d in 0 10 3; do eng 8 100 $KI --coef 1e-5 --target 6000000 $V --defer-dedup $d; done
for d in 0 10 3; do eng 4 100 $KI --coef 1e-5 --target 6000000 $V --defer-dedup $d; done
for d in 0 10; do eng 8 100 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000 $V --defer-dedup $d; done
for d in 0 10; do eng 8 250 $KI --coef 3e-6 --target 6000000 $V --defer-dedup $d; done
for d in 0 10; do eng 8 100 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000 $V --defer-dedup $d; done
echo "done"
