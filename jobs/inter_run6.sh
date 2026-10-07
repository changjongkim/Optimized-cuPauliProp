#!/bin/bash
# Interactive batch 6: the hand-tuned cuPauliProp baseline, strengthened (bin/pp_engine_v6).
#  - the 1e-13 cutoff that removes the vanishing branch of a Clifford-angle rotation is applied inside
#    the rank-local application (batch 5 ran out of memory on the noisy circuit without it)
#  - a shorter global-deduplication interval for the settings where "only before each truncation"
#    ran out of memory in batch 5
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/inter.sh
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
T="--cupp-tuned 1"
cupp 4 110 --circuit ki_noisy --noise 0.005 --coef 1e-4 $T $V
for k in 5 3 2; do cupp 4 150 $KI --coef 1e-5 $T --cupp-dedup-every $k $V; done
for k in 5 2; do cupp 4 150 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 $T --cupp-dedup-every $k $V; done
cupp 4 150 --circuit ki_noisy --noise 0.005 --coef 1e-5 $T $V
for k in 5 2; do cupp 4 150 --circuit ki_noisy --noise 0.005 --coef 1e-5 $T --cupp-dedup-every $k $V; done
for k in 5 2; do cupp 4 150 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 $T --cupp-dedup-every $k $V; done
echo "done"
