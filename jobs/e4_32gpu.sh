#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e4
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# 32 GPUs (8 nodes): does the gap to cuPauliProp's distributed mode widen with the GPU count
# on a setting beyond one GPU? 80 GB GPUs limited to 36 GB, as in the interactive batches.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
cupp() { local nodes=$1 tmo=$2; shift 2; echo "===== cupp nodes=$nodes gpus=$((nodes * 4)) args: $*"
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N $nodes -n $((nodes * 4)) --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./bin/pp_engine --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk "!seen[\$0]++" | head -12; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
# beyond one GPU: kicked Ising cutoff 1e-5 on 32 and 16 GPUs of the same nodes
eng 8 100 $KI --coef 1e-5 --target 6000000 $V
cupp 8 240 $KI --coef 1e-5 $V
eng 4 100 $KI --coef 1e-5 --target 6000000 $V
cupp 4 150 $KI --coef 1e-5 $V
for k in 0 5 3; do cupp 8 150 $KI --coef 1e-5 --cupp-tuned 1 --cupp-dedup-every $k $V; done
# a lower cutoff
eng 8 250 $KI --coef 3e-6 --target 6000000 $V
cupp 8 200 $KI --coef 3e-6 $V
# fits one GPU
eng 8 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V
cupp 8 150 --circuit ki_lab --coef 1e-4 $V
cupp 8 100 --circuit ki_lab --coef 1e-4 --cupp-tuned 1 $V
# a second circuit beyond one GPU
eng 8 100 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000 $V
cupp 8 240 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 $V
echo "done"
