#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_engine
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Unified engine on 1, 2 and 4 nodes (4 GPUs each): host-memory buckets, whole-bucket exchange
# between nodes. MPI carries host buffers only, so the GPU transport of Cray MPICH is switched off.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
export MPICH_GPU_SUPPORT_ENABLED=0
cd $PP_ROOT
run() {  # run <nodes> <timeout_s> <engine args...>
  local nodes=$1 tmo=$2; shift 2
  echo "===== nodes=$nodes gpus=$((nodes * 4)) args: $*"
  timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine --gpus 4 "$@" 2>&1 | grep -v -E '^$'
  echo "exit=${PIPESTATUS[0]}"
}
KI="--circuit ki --weight 0"
for n in 1 2 4; do run $n 200 $KI --coef 1e-4 --target 500000; done
for n in 1 2 4; do run $n 300 $KI --coef 1e-5 --target 6000000; done
run 4 500 $KI --coef 3e-6 --target 6000000
run 4 200 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 --target 4000000
run 4 200 --circuit xxz --lx 6 --ly 6 --jxy 0.1 --jz 0.15 --steps 6 --coef 1e-4 --target 4000000
echo "done"
