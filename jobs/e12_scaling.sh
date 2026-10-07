#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e12_g16
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Strong and weak scaling, one job per GPU count: every run of a job uses all its nodes (4 GPUs each).
# The same script is submitted for 4, 8, 16, 32 GPUs (debug) and 64, 128 GPUs (regular):
#   sbatch -N 1  -J pp_e12_g4   jobs/e12_scaling.sh
#   sbatch -N 16 -J pp_e12_g64  -q regular -t 00:40:00 --export=ALL,PP_BUDGET=2280 jobs/e12_scaling.sh
#   sbatch -N 32 -J pp_e12_g128 -q regular -t 00:45:00 --export=ALL,PP_BUDGET=2580 jobs/e12_scaling.sh
# 40 GB GPUs at 90% = the 36 GB per GPU of the earlier jobs (80 GB GPUs at 45%).
#   strong: the same problem at every GPU count (KI 3e-6 and 1e-5, TFIM and noisy KI 5e-6 and 1e-5)
#   weak:   the cutoff is lowered with the GPU count so that the work per GPU stays the same. Work = term
#           checks of the integrity line, measured 10.0x from 1e-5 to 3e-6 => cutoff = 1e-5 * (GPUs/4)^-0.52
# Up to 1,024 sectors per window (--gmax 10) in all engine runs, so that 128 GPUs get 8 sectors each.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_scal}
N=$SLURM_NNODES
G=$((N * 4))
BUDGET=${PP_BUDGET:-1650}     # seconds the runs of this job may take in total
case $G in
  4) WC=1e-5;; 8) WC=7e-6;; 16) WC=4.8e-6;; 32) WC=3.4e-6;; 64) WC=2.3e-6;; 128) WC=1.6e-6;;
  *) echo "no weak-scaling cutoff for $G GPUs"; exit 1;;
esac
# A run gets the time that is left; it is skipped when less than 60% of its limit is left.
fit() { local t=$1 left=$((BUDGET - SECONDS)); if ((t > left)); then if ((left * 10 < t * 6)); then echo 0; else echo $left; fi; else echo $t; fi; }
# Limit of a strong-scaling run from its time on 16 GPUs: 1.5x the ideal time plus two minutes.
lim() { echo $(($1 * 24 / G + 120)); }
eng() { local tmo=$(fit $1); shift; echo "===== engine nodes=$N gpus=$G args: $*"
  if ((tmo == 0)); then echo "skipped: no time left"; return; fi
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $N -n $N --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./$BIN --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
cupp() { local tmo=$(fit $1); shift; echo "===== cupp nodes=$N gpus=$G args: $*"
  if ((tmo == 0)); then echo "skipped: no time left"; return; fi
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N $N -n $((N * 4)) --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./$BIN --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk "!seen[\$0]++" | head -12; echo "exit=${PIPESTATUS[0]}"; }
V="--vram-frac 0.9"
E="--gmax 10 --trace 1 $V"
KI="--circuit ki_lab --weight 0"
TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"
NO="--circuit ki_noisy --noise 0.005"
echo "scaling job: nodes=$N gpus=$G weak-scaling cutoff=$WC budget=${BUDGET}s"

# strong scaling, the large problem (133 s on 16 GPUs in job 59483161)
eng $(lim 133) $KI --coef 3e-6 --target 6000000 $E
# weak scaling (on 4 GPUs the weak-scaling point is the 1e-5 run below)
if ((G > 4)); then
  eng 300 $KI --coef $WC --target 6000000 $E
  eng 300 $TF --coef $WC --target 4000000 $E
  eng 300 $NO --coef $WC --target 6000000 $E
fi
# strong scaling, the other problems
eng $(lim 14) $KI --coef 1e-5 --target 6000000 $E
eng $(lim 18) $TF --coef 1e-5 --target 4000000 $E
eng $(lim 11) $NO --coef 1e-5 --target 6000000 $E
eng $(lim 100) $TF --coef 5e-6 --target 4000000 $E
eng $(lim 80) $NO --coef 5e-6 --target 6000000 $E

# cuPauliProp distributed on the same nodes: default mode and hand-tuned (on 16 GPUs hand-tuned needs
# a deduplication every 3 (KI) and 2 (TFIM) operations to fit)
if ((G >= 16)); then
  KD=""; TD=""
  if ((G == 16)); then KD="--cupp-dedup-every 3"; TD="--cupp-dedup-every 2"; fi
  cupp 150 $KI --coef 1e-5 $V
  cupp 100 $KI --coef 1e-5 --cupp-tuned 1 $KD $V
  cupp 200 $TF --coef 1e-5 $V
  cupp 120 $TF --coef 1e-5 --cupp-tuned 1 $TD $V
  cupp 170 $NO --coef 1e-5 $V
  cupp 80 $NO --coef 1e-5 --cupp-tuned 1 $V
fi

# sector count: the large problem with the default limit of 256 sectors per window
eng $(lim 133) $KI --coef 3e-6 --target 6000000 --trace 1 $V
if ((G >= 64)); then
  # up to 4,096 sectors per window with smaller host chunks
  eng $(lim 133) $KI --coef 3e-6 --target 6000000 --gmax 12 --chunk 65536 --trace 1 $V
  eng 300 $KI --coef $WC --target 6000000 --gmax 12 --chunk 65536 --trace 1 $V
fi
if ((G >= 128)); then
  # a cutoff that did not finish in 25 minutes on 16 GPUs
  eng 420 $KI --coef 1e-6 --target 6000000 $E
fi
if ((G >= 64)); then
  # does cuPauliProp hold the large problem with this many GPUs? (out of memory on 32 GPUs)
  cupp 300 $KI --coef 3e-6 --cupp-tuned 1 --cupp-dedup-every 3 $V
  cupp 360 $KI --coef 3e-6 $V
fi
echo "done after ${SECONDS}s"
