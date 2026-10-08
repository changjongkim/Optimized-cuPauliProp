#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e13_g16
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Strong and weak scaling of the engine again, one job per GPU count (see e12_scaling.sh for the layout):
#   sbatch -N 1  -J pp_e13_g4   jobs/e13_scaling.sh        (also -N 2, 4, 8 with -J pp_e13_g8, _g16, _g32)
#   sbatch -N 16 -J pp_e13_g64  -q regular jobs/e13_scaling.sh
#   sbatch -N 32 -J pp_e13_g128 -q regular jobs/e13_scaling.sh
# Why again: in job set e12 the sample for choosing the rows was 200,000 terms per node instead of 200,000
# in total, so choosing the rows on node 0 took 0.4 s on one node and 15-45 s on 32 nodes. This binary
# takes 200,000 terms in total. The cuPauliProp runs of e12 are not affected and are not repeated.
# Main runs use the engine defaults (up to 256 sectors per window), the fastest setting in e12.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
BIN=${PP_BIN:-bin/pp_engine_scal2}
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
E="--trace 1 --vram-frac 0.9"
KI="--circuit ki_lab --weight 0"
TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"
NO="--circuit ki_noisy --noise 0.005"
echo "scaling job: nodes=$N gpus=$G weak-scaling cutoff=$WC budget=${BUDGET}s binary=$BIN"

# strong scaling, the large problem
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

# sector count and host chunk size: up to 1,024 sectors per window, with the default chunks (262,144 terms)
# and with chunks of 65,536 terms
if ((G > 4)); then
  eng 300 $KI --coef $WC --target 6000000 --gmax 10 $E
  eng 300 $KI --coef $WC --target 6000000 --gmax 10 --chunk 65536 $E
  eng 300 $TF --coef $WC --target 4000000 --gmax 10 --chunk 65536 $E
fi
eng $(lim 133) $KI --coef 3e-6 --target 6000000 --gmax 10 --chunk 65536 $E
eng $(lim 133) $KI --coef 3e-6 --target 6000000 --chunk 65536 $E
if ((G >= 128)); then
  # a cutoff that cuPauliProp cannot hold and that did not finish in 25 minutes on 16 GPUs
  eng 420 $KI --coef 1e-6 --target 6000000 --gmax 10 $E
fi
echo "done after ${SECONDS}s"
