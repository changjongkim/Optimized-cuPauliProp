#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:12:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_x1_fused
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Side experiment, not part of the engine: cuPauliProp's experimental fused operator application on one GPU
# (src/proto/fused_test.cpp). Noisy kicked Ising; consecutive noise channels in one library call against one
# call per operation. Two repetitions of every setting.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
run() { echo "===== fused_test args: $*"; timeout 150 srun -N 1 -n 1 -c 16 --gpus-per-node=4 --cpu-bind=cores ./bin/fused_test --vram-frac 0.9 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
for rep in 1 2; do
  for coef in 1e-3 1e-4 3e-5; do
    for fuse in 0 2 5 10; do run --coef $coef --fuse $fuse; done
  done
done
echo "done after ${SECONDS}s"
