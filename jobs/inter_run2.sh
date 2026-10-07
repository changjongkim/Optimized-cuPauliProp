#!/bin/bash
# Interactive batch 2 (80 GB nodes limited to 36 GB per GPU): bucket-count setting, the other circuits
# with and without the pilot, accuracy against exact 30-qubit values.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/inter.sh
V="--vram-frac 0.45"
for pg in 2 4; do eng 4 120 --circuit ki_lab --weight 0 --coef 1e-5 --target 6000000 $V --per-gpu $pg; done
for p in 1 0; do
  eng 4 150 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 --target 4000000 $V --per-gpu 2 --pilot $p
  eng 4 200 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000 $V --per-gpu 2 --pilot $p
  eng 4 150 --circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --coef 1e-4 --target 4000000 $V --per-gpu 2 --pilot $p
  eng 4 150 --circuit qaoa --n 100 --layers 4 --coef 1e-4 --target 4000000 $V --per-gpu 2 --pilot $p
  eng 4 100 --circuit hea --n 100 --layers 16 --amp 0.4 --coef 1e-5 --target 4000000 $V --per-gpu 2 --pilot $p
  eng 4 150 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000 $V --per-gpu 2 --pilot $p
done
# accuracy: 5x6 lattice, exact 6 steps 0.790133273060, 4 steps 0.802867136639
for c in 1e-5 3e-6 1e-6; do eng 4 200 --circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps 6 --coef $c --target 4000000 $V --per-gpu 2; done
for c in 1e-6 1e-7; do eng 4 150 --circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps 4 --coef $c --target 4000000 $V --per-gpu 2; done
echo "done"
