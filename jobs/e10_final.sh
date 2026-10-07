#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e10
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Engine with its final defaults (a bucket per node from the growth phase on) against both
# cuPauliProp baselines on the same nodes: 32 GPUs, then 16 GPUs. 80 GB GPUs limited to 36 GB.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_dd6}
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
cupp() { local nodes=$1 tmo=$2; shift 2; echo "===== cupp nodes=$nodes gpus=$((nodes * 4)) args: $*"
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N $nodes -n $((nodes * 4)) --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./$BIN --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk "!seen[\$0]++" | head -12; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"
NO="--circuit ki_noisy --noise 0.005"
# 32 GPUs
eng 8 100 $KI --coef 1e-5 --target 6000000 $V
eng 8 100 $TF --coef 1e-5 --target 4000000 $V
eng 8 100 $NO --coef 1e-5 --target 6000000 $V
cupp 8 120 $KI --coef 1e-5 $V
cupp 8 80 $KI --coef 1e-5 --cupp-tuned 1 $V
cupp 8 160 $TF --coef 1e-5 $V
cupp 8 80 $TF --coef 1e-5 --cupp-tuned 1 $V
cupp 8 140 $NO --coef 1e-5 $V
cupp 8 60 $NO --coef 1e-5 --cupp-tuned 1 $V
# 16 GPUs (hand-tuned with the fastest deduplication interval that fitted: 3, 2 and none)
eng 4 100 $KI --coef 1e-5 --target 6000000 $V
eng 4 100 $TF --coef 1e-5 --target 4000000 $V
eng 4 100 $NO --coef 1e-5 --target 6000000 $V
cupp 4 120 $KI --coef 1e-5 $V
cupp 4 120 $KI --coef 1e-5 --cupp-tuned 1 --cupp-dedup-every 3 $V
cupp 4 140 $TF --coef 1e-5 $V
cupp 4 120 $TF --coef 1e-5 --cupp-tuned 1 --cupp-dedup-every 2 $V
cupp 4 140 $NO --coef 1e-5 $V
cupp 4 80 $NO --coef 1e-5 --cupp-tuned 1 $V
# settings that fit one GPU, engine only (the baselines of these are in logs/inter_run5.out)
eng 4 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V
eng 4 60 $TF --coef 1e-4 --target 1000000 $V
eng 4 60 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-4 --target 1000000 $V
eng 4 60 --circuit xxz --lx 6 --ly 6 --jxy 0.1 --jz 0.15 --steps 4 --coef 1e-3 --target 1000000 $V
eng 4 60 --circuit qaoa --n 60 --layers 3 --coef 1e-4 --target 1000000 $V
eng 4 60 --circuit hea --n 60 --layers 8 --amp 0.3 --coef 1e-5 --target 1000000 $V
eng 4 60 $NO --coef 1e-4 --target 1000000 $V
eng 8 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V
# number of GPUs
eng 2 100 $KI --coef 1e-5 --target 6000000 $V
eng 1 100 $KI --coef 1e-5 --target 6000000 $V
eng 1 150 $KI --coef 1e-5 --target 6000000 $V --gpus 2
# a lower cutoff
eng 8 200 $KI --coef 3e-6 --target 6000000 $V
echo "done"
