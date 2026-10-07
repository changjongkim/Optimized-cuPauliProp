#!/bin/bash
# Helpers for runs inside an interactive allocation: source this file, then call eng / cupp.
#   export PP_JOB=<allocation id>   PP_BIN=<engine binary, default bin/pp_engine>
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
PP_BIN=${PP_BIN:-bin/pp_engine}
# engine: one process per node, host buffers only
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun --jobid=$PP_JOB -N $nodes -n $nodes --ntasks-per-node=1 -c 64 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$PP_BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$|^srun: (Job|Step)"; echo "exit=${PIPESTATUS[0]}"; }
# cuPauliProp distributed mode: one rank per GPU, GPU-aware MPI, NVIDIA's plugin
cupp() { local nodes=$1 tmo=$2; shift 2; echo "===== cupp nodes=$nodes gpus=$((nodes * 4)) args: $*"
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun --jobid=$PP_JOB -N $nodes -n $((nodes * 4)) --ntasks-per-node=4 -c 16 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./$PP_BIN --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk '!seen[$0]++' | head -12; echo "exit=${PIPESTATUS[0]}"; }
