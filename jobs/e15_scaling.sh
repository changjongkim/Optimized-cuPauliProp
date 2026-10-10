#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 8
#SBATCH -t 00:20:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e15_g32
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Scaling on 32-256 GPUs with the current engine (bin/pp_engine_h), one job per GPU count. It repeats what the job
# sets e12 (cuPauliProp) and e13 (SectorQ) measured on 32, 64 and 128 GPUs with the earlier build, and adds 256 GPUs:
#   sbatch -N 8  -J pp_e15_g32  -t 00:20:00 --export=ALL,PP_BUDGET=1080 jobs/e15_scaling.sh
#   sbatch -N 16 -J pp_e15_g64  -t 00:22:00 --export=ALL,PP_BUDGET=1200 jobs/e15_scaling.sh
#   sbatch -N 32 -J pp_e15_g128 -t 00:25:00 --export=ALL,PP_BUDGET=1380 jobs/e15_scaling.sh
#   sbatch -N 64 -J pp_e15_g256 -t 00:25:00 --export=ALL,PP_BUDGET=1380 jobs/e15_scaling.sh
# 1-16 GPUs are measured by campaign g (queue_g). Every run of a job uses all its nodes (4 GPUs each, 36 GB per GPU).
#   strong: the same problem at every GPU count (KI 3e-6 and 1e-5, TFIM and noisy KI 5e-6 and 1e-5)
#   weak:   cutoff = 1e-5 * (GPUs/4)^-0.52, the series of the large load of campaign g (2e-5 on 1 GPU ... 4.8e-6 on 16)
#   cuPauliProp distributed on the same nodes: default mode and deduplication deferred to the truncations, cutoff 1e-5
# The runs are ordered by importance: a run is skipped when the time that is left is too short for it.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_h}
N=$SLURM_NNODES
G=$((N * 4))
BUDGET=${PP_BUDGET:-1080}     # seconds the runs of this job may take in total
case $G in
  32) WC=3.4e-6;; 64) WC=2.3e-6;; 128) WC=1.6e-6;; 256) WC=1.15e-6;;
  *) echo "this job is for 32, 64, 128 or 256 GPUs, not $G"; exit 1;;
esac
# A run gets the time that is left; it is skipped when less than 60% of its limit is left.
fit() { local t=$1 left=$((BUDGET - SECONDS)); if ((t > left)); then if ((left * 10 < t * 6)); then echo 0; else echo $left; fi; else echo $t; fi; }
# Limit of a strong-scaling run from its time on 16 GPUs: 1.5x the ideal time plus two minutes.
lim() { echo $(($1 * 24 / G + 120)); }
# cuPauliProp's distributed mode does not get faster with more GPUs: on 256 GPUs its limits are 1.5x those of e12.
cl() { if ((G >= 256)); then echo $(($1 * 3 / 2)); else echo $1; fi; }
eng() { local tmo=$(fit $1); shift; echo "===== engine nodes=$N gpus=$G args: $*"
  if ((tmo == 0)); then echo "skipped: no time left"; return; fi
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $N -n $N --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
cupp() { local tmo=$(fit $1); shift; echo "===== cupp nodes=$N gpus=$G args: $*"
  if ((tmo == 0)); then echo "skipped: no time left"; return; fi
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N $N -n $((N * 4)) --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./$BIN --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk "!seen[\$0]++" | head -12; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.9"
E="--trace 1 $V"
KI="--circuit ki_lab --weight 0"
TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"
NO="--circuit ki_noisy --noise 0.005"
echo "scaling job: nodes=$N gpus=$G weak-scaling cutoff=$WC budget=${BUDGET}s binary=$BIN"

# SectorQ, engine defaults (up to 256 sectors per window)
eng $(lim 133) $KI --coef 3e-6 --target 6000000 $E          # strong scaling, the large problem
eng 300 $KI --coef $WC --target 6000000 $E                  # weak scaling
eng 300 $TF --coef $WC --target 4000000 $E
eng 300 $NO --coef $WC --target 6000000 $E
eng $(lim 14) $KI --coef 1e-5 --target 6000000 $E           # strong scaling, the other problems
eng $(lim 18) $TF --coef 1e-5 --target 4000000 $E
eng $(lim 11) $NO --coef 1e-5 --target 6000000 $E
eng $(lim 100) $TF --coef 5e-6 --target 4000000 $E
eng $(lim 80) $NO --coef 5e-6 --target 6000000 $E

# cuPauliProp distributed on the same nodes, cutoff 1e-5
cupp $(cl 150) $KI --coef 1e-5 $V
cupp $(cl 100) $KI --coef 1e-5 --cupp-tuned 1 $V
cupp $(cl 200) $TF --coef 1e-5 $V
cupp $(cl 120) $TF --coef 1e-5 --cupp-tuned 1 $V
cupp $(cl 170) $NO --coef 1e-5 $V
cupp $(cl 80) $NO --coef 1e-5 --cupp-tuned 1 $V

# SectorQ with up to 1,024 sectors per window: with 256 the sectors of the largest problems do not fit the GPU buffer
# once there are two or fewer per GPU
if ((G >= 64)); then
  eng 300 $KI --coef $WC --target 6000000 --gmax 10 $E
  eng $(lim 133) $KI --coef 3e-6 --target 6000000 --gmax 10 $E
fi
if ((G >= 128)); then
  # a cutoff that cuPauliProp cannot hold on any of these GPU counts (11.9 billion terms at the window boundaries)
  eng 420 $KI --coef 1e-6 --target 6000000 --gmax 10 $E
fi
if ((G >= 64)); then
  # cuPauliProp at the cutoff of the large problem: out of memory on 64 and 128 GPUs in e12
  cupp 200 $KI --coef 3e-6 $V
fi
echo "done after ${SECONDS}s"
