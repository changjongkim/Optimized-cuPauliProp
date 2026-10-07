#!/bin/bash
# Interactive batch 4, final engine build (snap rule, 2 buckets per GPU, exchange after each window).
# 80 GB nodes limited to 36 GB per GPU so the numbers compare with the 40 GB nodes; the last run uses
# the whole GPUs.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/inter.sh
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
for n in 1 2 4; do eng $n 200 $KI --coef 1e-5 --target 6000000 $V; done
for n in 1 2; do cupp $n 150 $KI --coef 1e-5 $V; done
cupp 4 150 $KI --coef 1e-5 $V
eng 4 300 $KI --coef 3e-6 --target 6000000 $V
cupp 4 200 $KI --coef 3e-6 $V
cupp 4 150 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 $V
eng 4 100 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000 $V
cupp 4 150 --circuit ki_noisy --noise 0.005 --coef 1e-5 $V
eng 4 100 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000 $V
eng 4 1500 $KI --coef 1e-6 --target 12000000 --vram-frac 0.9
echo "done"
