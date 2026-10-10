#!/bin/bash
# u3: the engine with the refined rule on one GPU and host chunks prepared ahead (bin/pp_engine_u6).
#   bash jobs/u3_drive.sh <allocation of four nodes>     (one GPU, then 16 GPUs, then the largest one-GPU settings)
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export SLURM_JOB_ID=$1
OUT=$PP_ROOT/logs/u6
B=pp_engine_u6
run1() {   # run1 ID TIMEOUT ARGS...   one GPU of one node
  local id=$1 tmo=$2; shift 2
  local args="$*"
  { echo "===== engine nodes=1 gpus=1 args: $args --gpus 1"
    MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N 1 -n 1 --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
      stdbuf -oL ./bin/$B --gpus 1 $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)"
    echo "exit=${PIPESTATUS[0]}"
  } > $OUT/$B.$id.out 2>&1
  echo "$(date +%H:%M:%S) $B $id $(grep -m1 '^exit=' $OUT/$B.$id.out) $(grep -m1 -E '^wall=[0-9.]+' $OUT/$B.$id.out | cut -d' ' -f1) $(grep -m1 '^expectation=' $OUT/$B.$id.out) $(grep -m1 '^one GPU' $OUT/$B.$id.out)"
}
KI="--circuit ki_lab --weight 0"; TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"; NO="--circuit ki_noisy --noise 0.005"
SQ="--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10"; XX="--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6"
QA="--circuit qaoa --n 100 --layers 4"; HE="--circuit hea --n 100 --layers 16 --amp 0.4"
E6="--target 6000000 --trace 1 --vram-frac 0.9"; E4="--target 4000000 --trace 1 --vram-frac 0.9"
PP_OUT=u6 PP_BIN=$B bash jobs/u2_exchange.sh > $OUT/driver-$1.log 2>&1
{
run1 ki.1e-4.g1 200 $KI --coef 1e-4 $E6;   run1 tfim.1e-4.g1 200 $TF --coef 1e-4 $E4;  run1 noise.1e-4.g1 200 $NO --coef 1e-4 $E6
run1 ki.3e-5.g1 200 $KI --coef 3e-5 $E6;   run1 tfim.3e-5.g1 200 $TF --coef 3e-5 $E4;  run1 noise.3e-5.g1 200 $NO --coef 3e-5 $E6
run1 tfimsq.1e-4.g1 200 $SQ --coef 1e-4 $E4; run1 xxz.1e-4.g1 200 $XX --coef 1e-4 $E4
run1 qaoa.1e-4.g1 200 $QA --coef 1e-4 $E4;  run1 hea.1e-5.g1 200 $HE --coef 1e-5 $E4
echo "one GPU done after ${SECONDS}s"
} >> $OUT/driver-$1.log 2>&1
{
run1 ki.1e-5.g1 300 $KI --coef 1e-5 $E6
echo "all done after ${SECONDS}s"
} >> $OUT/driver-$1.log 2>&1
scancel $1
