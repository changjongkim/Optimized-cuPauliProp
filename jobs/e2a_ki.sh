#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e2a
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
# engine: one process per node, host buffers only (GPU transport of Cray MPICH off)
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
# cuPauliProp distributed mode: one rank per GPU, GPU-aware MPI, original NVIDIA plugin
cupp() { local nodes=$1 tmo=$2; shift 2; echo "===== cupp nodes=$nodes gpus=$((nodes * 4)) args: $*"
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N $nodes -n $((nodes * 4)) --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./bin/pp_engine --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk "!seen[\$0]++" | head -12; echo "exit=${PIPESTATUS[0]}"; }

# Kicked Ising (original circuit; the frame is found by the engine): node scaling, host-memory
# ablation, lower cutoff, checkpoint and restart across nodes.
KI="--circuit ki_lab --weight 0"
for n in 1 2 4; do eng $n 200 $KI --coef 1e-5 --target 6000000; done
eng 4 200 $KI --coef 1e-5 --target 6000000 --pinned 0
eng 4 400 $KI --coef 3e-6 --target 6000000
CK=$PP_ROOT/results/proto/ki_b_4node.ckpt
eng 4 200 $KI --coef 1e-5 --target 6000000 --ckpt-window 12 --ckpt-file $CK --stop-at-ckpt 1
eng 4 200 $KI --coef 1e-5 --target 6000000 --resume $CK
rm -f $CK.r*
echo "done"
