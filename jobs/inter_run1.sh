#!/bin/bash
# Kicked Ising on the interactive 80 GB nodes, restricted to 36 GB per GPU (--vram-frac 0.45) to
# match the 40 GB nodes used before.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/inter.sh
V="--vram-frac 0.45"
KI="--circuit ki_lab --weight 0"
for n in 1 2 4; do eng $n 200 $KI --coef 1e-5 --target 6000000 $V; done
eng 4 200 $KI --coef 1e-5 --target 6000000 $V --pilot 0
eng 4 200 $KI --coef 1e-5 --target 6000000 $V --per-gpu 1
eng 4 300 $KI --coef 3e-6 --target 6000000 $V
eng 4 60 $KI --coef 1e-4 --target 500000 $V
echo "done"
