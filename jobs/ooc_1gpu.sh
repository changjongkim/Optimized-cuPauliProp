#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -J pp_ooc
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Out-of-core prototype on ONE A100-40GB: problems whose term count exceeds what the GPU can hold.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
OUT=$PP_ROOT/results/ooc_$SLURM_JOB_ID
mkdir -p $OUT
cd $PP_ROOT
free -g | head -2
echo "===== B (coef 1e-5): cuPauliProp fails on 1/4/8/16 GPUs"
timeout 700 ./bin/lo_ooc --coef 1e-5 --weight 0 --vram-frac 0.9 --target 6000000 --tag oocB --outdir $OUT 2>&1 | grep -v -E '^$'
echo "exit=$?"
echo "===== C (coef 3e-6)"
timeout 1000 ./bin/lo_ooc --coef 3e-6 --weight 0 --vram-frac 0.9 --target 6000000 --tag oocC --outdir $OUT 2>&1 | grep -v -E '^$'
echo "exit=$?"
echo "done"
