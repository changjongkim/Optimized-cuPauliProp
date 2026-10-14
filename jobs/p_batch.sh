#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 8
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_p_g32
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# paulistrings-rs against SectorQ under the same truncation rule (a truncation after every operation) on more GPUs than
# the four-node interactive runs (jobs/p_drive.py, jobs/p_sq.sh): 32 GPUs (8 nodes) or 64 GPUs (16 nodes).
#   sbatch -N 8  -J pp_p_g32 -t 03:00:00 jobs/p_batch.sh
#   sbatch -N 16 -J pp_p_g64 -t 03:00:00 jobs/p_batch.sh
# First SectorQ down the ladders of the seven circuits (logs/p/sq.e1[.<circuit>].g<GPUs>.<cutoff>.out, a ladder stops at
# its first run that does not finish), then paulistrings-rs with one MPI rank per GPU (mode mg32 or mg64 of p_drive.py).
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export MPICH_GPU_SUPPORT_ENABLED=0
JOB=$SLURM_JOB_ID
N=$SLURM_NNODES
G=$((N * 4))
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
echo "$(date +%m-%dT%H:%M:%S) job $JOB: $N nodes, $G GPUs"
declare -A CIRC=(
  [ki]="--circuit ki_lab --weight 0 --target 6000000"
  [tfim]="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --target 4000000"
  [noise]="--circuit ki_noisy --noise 0.005 --target 6000000"
  [tfimsq]="--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --target 4000000"
  [xxz]="--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000"
  [qaoa]="--circuit qaoa --n 100 --layers 4 --target 4000000"
  [hea]="--circuit hea --n 100 --layers 16 --amp 0.4 --target 4000000"
)
run() {   # circuit, cutoff (as jobs/p_sq.sh, on all nodes of this job)
  local c=$1 x=$2
  local log=logs/p/sq.e1.$c.g$G.$x.out
  [ $c = ki ] && log=logs/p/sq.e1.g$G.$x.out
  [ -s $log ] && grep -q "^exit=" $log && return
  [ -e logs/p/sq.stop.$c.g$G ] && return
  echo "===== SectorQ, truncation after every operation, $c, $G GPUs, cutoff $x" > $log
  timeout 2400 srun --jobid=$JOB --nodelist=$HOSTS --overlap -N $N -n $N --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_i --gpus 4 ${CIRC[$c]} --coef $x --trace 0 --vram-frac 0.9 --trunc-every 1 >> $log 2>&1
  local code=$?
  echo "exit=$code" >> $log
  [ $code -ne 0 ] && touch logs/p/sq.stop.$c.g$G
  echo "$(date +%m-%dT%H:%M:%S) SectorQ $c $G GPUs $x: $(grep -m1 '^expectation=' $log) $(grep -o '^wall=[0-9.]* s' $log | head -1) exit=$code"
}
# the ladders of jobs/p_sq.sh and one cutoff further (MORE in jobs/p_drive.py)
declare -A LADDER=([ki]="1e-3 1e-4 3e-5 1e-5 3e-6 1e-6 5e-7" [tfim]="1e-4 3e-5 1e-5 3e-6 1e-6 5e-7" [noise]="1e-4 3e-5 1e-5 3e-6 1e-6 5e-7"
                   [tfimsq]="1e-4 3e-5 1e-5 3e-6 1e-6" [xxz]="1e-4 3e-5 1e-5 3e-6 1e-6" [qaoa]="1e-4 3e-5 1e-5 3e-6 1e-6 5e-7"
                   [hea]="1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 1e-7")
for c in ki tfim noise tfimsq xxz qaoa hea; do for x in ${LADDER[$c]}; do run $c $x; done; done
echo "$(date +%m-%dT%H:%M:%S) paulistrings-rs, mode mg$G: $(env/bin/python3 jobs/p_drive.py --count) runs open in all"
env/bin/python3 jobs/p_drive.py --job $JOB --modes mg$G
echo "$(date +%m-%dT%H:%M:%S) done; $(env/bin/python3 jobs/p_drive.py --count) runs open in all"
