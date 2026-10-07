#!/bin/bash
# Interactive batch 5: three-way comparison on the same nodes.
#   engine | cuPauliProp distributed, default | cuPauliProp distributed, hand-tuned
# Hand-tuned = operations applied rank-locally (keepDuplicates=1), global deduplication only before
# each truncation. All three get the same frame-compiled operation list.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/inter.sh
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
set -- "ki_lab --coef 1e-4" "tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-4" \
       "tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-4" \
       "xxz --lx 6 --ly 6 --jxy 0.1 --jz 0.15 --steps 4 --coef 1e-3" "qaoa --n 60 --layers 3 --coef 1e-4" \
       "hea --n 60 --layers 8 --amp 0.3 --coef 1e-5" "ki_noisy --noise 0.005 --coef 1e-4"
# 1. hand-tuned baseline, 16 GPUs: medium settings, then the large ones
for c in "$@"; do cupp 4 110 --circuit $c --cupp-tuned 1 $V; done
cupp 4 150 $KI --coef 1e-5 --cupp-tuned 1 $V
cupp 4 150 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --cupp-tuned 1 $V
cupp 4 150 --circuit ki_noisy --noise 0.005 --coef 1e-5 --cupp-tuned 1 $V
cupp 4 150 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 --cupp-tuned 1 $V
cupp 4 200 $KI --coef 3e-6 --cupp-tuned 1 $V
for n in 1 2; do cupp $n 150 $KI --coef 1e-5 --cupp-tuned 1 $V; done
# 2. engine on one and two GPUs of one node (the 4/8/16-GPU points are in batch 4)
eng 1 600 $KI --coef 1e-5 --target 6000000 $V --gpus 1
eng 1 400 $KI --coef 1e-5 --target 6000000 $V --gpus 2
# 3. both baselines with the whole 80 GB GPUs at the cutoffs the engine reaches (the engine's
#    cutoff 1e-6 run in batch 4 used --vram-frac 0.9)
cupp 4 250 $KI --coef 3e-6 --vram-frac 0.9
cupp 4 250 $KI --coef 1e-6 --vram-frac 0.9
cupp 4 250 $KI --coef 3e-6 --cupp-tuned 1 --vram-frac 0.9
# 4. engine and default baseline on the same nodes, medium settings
for c in "$@"; do
  eng 4 100 --circuit $c --target 1000000 $V
  cupp 4 110 --circuit $c $V
done
eng 4 100 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 --target 4000000 $V
cupp 4 150 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 $V
echo "done"
