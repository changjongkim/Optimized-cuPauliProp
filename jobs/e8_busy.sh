#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e8
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Spread the growth phase over all GPUs (--busy-window t): once a window kept a node busy for t
# seconds, buckets as small as 2000 terms are allowed. Engine with and without the rule, and both
# cuPauliProp baselines on the same nodes, 16 and 32 GPUs. 80 GB GPUs limited to 36 GB.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_dd4}
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
# engine, with and without the rule
for b in 0 0.1 0.2; do eng 8 100 $KI --coef 1e-5 --target 6000000 $V --busy-window $b; done
for b in 0 0.2; do eng 4 100 $KI --coef 1e-5 --target 6000000 $V --busy-window $b; done
for b in 0 0.1 0.2; do eng 8 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V --busy-window $b; done
for b in 0 0.2; do eng 4 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V --busy-window $b; done
for n in 8 4; do eng $n 100 $TF --coef 1e-5 --target 4000000 $V --busy-window 0.2; eng $n 100 $NO --coef 1e-5 --target 6000000 $V --busy-window 0.2; done
# baselines on the same nodes
cupp 8 150 $KI --coef 1e-5 $V
cupp 4 150 $KI --coef 1e-5 $V
cupp 8 100 $KI --coef 1e-5 --cupp-tuned 1 $V
cupp 8 200 $TF --coef 1e-5 $V
cupp 8 150 $TF --coef 1e-5 --cupp-tuned 1 $V
cupp 8 150 $NO --coef 1e-5 $V
cupp 8 100 $NO --coef 1e-5 --cupp-tuned 1 $V
# a lower cutoff
eng 8 200 $KI --coef 3e-6 --target 6000000 $V --busy-window 0.2
eng 4 250 $KI --coef 3e-6 --target 6000000 $V --busy-window 0.2
echo "done"
