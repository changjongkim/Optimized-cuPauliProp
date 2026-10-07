#!/bin/bash
# Interactive batch 3: exchange overlapped with computation (pp_engine_v3), snap rule without pilot.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/inter.sh
V="--vram-frac 0.45 --per-gpu 2 --pilot 0"
KI="--circuit ki_lab --weight 0"
for n in 2 4; do eng $n 150 $KI --coef 1e-5 --target 6000000 $V; done
eng 4 300 $KI --coef 3e-6 --target 6000000 $V
eng 4 100 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 --target 4000000 $V
eng 4 400 --circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps 6 --coef 1e-6 --target 4000000 $V
echo "done"
