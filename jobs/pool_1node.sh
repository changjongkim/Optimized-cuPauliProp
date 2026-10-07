#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_pool
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Bucket-task prototype (threads, no MPI) on the 4 GPUs of one node.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
run() { echo "===== $*"; timeout $1 srun -n 1 -c 128 --gpus-per-node=4 stdbuf -oL ./bin/lo_pool "${@:2}" 2>&1 | grep -v -E '^$'; echo "exit=${PIPESTATUS[0]}"; }
A="--coef 1e-4 --weight 0"
B="--coef 1e-5 --weight 0"
C="--coef 3e-6 --weight 0"
for g in 1 2 4; do run 200 --gpus $g $A --target 500000; done
for g in 4 2; do run 400 --gpus $g $B --target 6000000; done
run 700 --gpus 4 $C --target 6000000
echo "done"
