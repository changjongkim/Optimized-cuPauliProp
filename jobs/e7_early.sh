#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e7
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Partition earlier: the windows with a single bucket keep one GPU of one node busy while the
# others wait (2.5 s of 10.7 s on 32 GPUs, job 59481492). Smaller minimum bucket and a higher
# growth limit let the engine use all GPUs from the growth phase on.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_dd3}
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
eng 8 100 $KI --coef 1e-5 --target 6000000 $V
for mb in 20000 2000; do for gc in 8 64; do eng 8 100 $KI --coef 1e-5 --target 6000000 $V --min-bucket $mb --growth-cap $gc; done; done
eng 8 100 $KI --coef 1e-5 --target 6000000 $V --growth-cap 64
eng 4 100 $KI --coef 1e-5 --target 6000000 $V
eng 4 100 $KI --coef 1e-5 --target 6000000 $V --min-bucket 2000 --growth-cap 64
eng 8 100 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000 $V --min-bucket 2000 --growth-cap 64
eng 8 100 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000 $V
eng 8 100 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000 $V --min-bucket 2000 --growth-cap 64
eng 8 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V
eng 8 60 --circuit ki_lab --coef 1e-4 --target 1000000 $V --min-bucket 2000 --growth-cap 64
eng 8 250 $KI --coef 3e-6 --target 6000000 $V --min-bucket 2000 --growth-cap 64
echo "done"
