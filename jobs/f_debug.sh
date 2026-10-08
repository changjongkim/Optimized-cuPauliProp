#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_f_debug
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/f/%x-%j.out
# Campaign f in a debug allocation (four nodes, 30 minutes): the runs that fit are taken from queue_f/todo.txt.
bash /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/f_drive.sh $SLURM_JOB_ID
echo "done after ${SECONDS}s"
