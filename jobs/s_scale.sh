#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 8
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_s_g32
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Scale campaign: every experiment of one GPU count in one job (list: jobs/s_items.py -> queue_s/todo_s.jsonl).
#   sbatch -N 8  -J pp_s_g32  -t 03:00:00 jobs/s_scale.sh
#   sbatch -N 16 -J pp_s_g64  -t 03:30:00 jobs/s_scale.sh
#   sbatch -N 32 -J pp_s_g128 -t 03:30:00 jobs/s_scale.sh
#   sbatch -N 64 -J pp_s_g256 -t 03:30:00 jobs/s_scale.sh
#   sbatch -N 128 -J pp_s_g512 -t 01:30:00 jobs/s_scale.sh     (512 GPUs: the scale-out runs of SectorQ only)
# The runner takes the runs that need as many nodes as the job has, from the top of the list: SectorQ over the cutoffs,
# weak scaling, cuPauliProp's distributed mode, the CPU implementations on the same nodes. A run that does not fit into
# the time that is left stays open; the same script submitted again continues with what is open.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
env/bin/python3 jobs/s_items.py --take          # runs that the scaling jobs e15 made in the meantime are not repeated
# Job 59660398 (64 GPUs) hung at this count for its whole 2.5 hours and started no run. If the runner does not answer
# within five minutes the job ends at once instead of holding its nodes.
open=$(timeout 300 env/bin/python3 jobs/s_run.py --count) || { echo "the runner did not answer within 5 minutes on $(hostname); the job gives its nodes back"; exit 1; }
echo "scale job $SLURM_JOB_ID: nodes=$SLURM_NNODES gpus=$((SLURM_NNODES * 4)) open=$open"
env/bin/python3 jobs/s_run.py --job $SLURM_JOB_ID --hosts $HOSTS
echo "done after ${SECONDS}s; open=$(env/bin/python3 jobs/s_run.py --count)"
