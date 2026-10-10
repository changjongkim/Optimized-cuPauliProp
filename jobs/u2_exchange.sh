#!/bin/bash
# u2: where the time of the exchange between nodes goes (bin/pp_engine_u2 prints its parts: counts, posting the
# receives with new host chunks, transfer). Four nodes, 16 GPUs, the largest settings of campaign f.
#   SLURM_JOB_ID=<allocation of four nodes> bash jobs/u2_exchange.sh        (or inside a batch job)
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
OUT=$PP_ROOT/logs/${PP_OUT:-u2}
BIN=${PP_BIN:-pp_engine_u2}
run() {   # run ID TIMEOUT ARGS...
  local id=$1 tmo=$2; shift 2
  local args="$*"
  { echo "===== engine nodes=4 gpus=16 args: $args --gpus 4"
    MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N 4 -n 4 --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
      stdbuf -oL ./bin/$BIN --gpus 4 $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)"
    echo "exit=${PIPESTATUS[0]}"
  } > $OUT/$BIN.$id.out 2>&1
  echo "$(date +%H:%M:%S) $BIN $id $(grep -m1 '^exit=' $OUT/$BIN.$id.out) $(grep -m1 -E '^wall=[0-9.]+' $OUT/$BIN.$id.out | cut -d' ' -f1) $(grep -m1 '^expectation=' $OUT/$BIN.$id.out)"
}
KI="--circuit ki_lab --weight 0"; TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"; NO="--circuit ki_noisy --noise 0.005"
QA="--circuit qaoa --n 100 --layers 4"; HE="--circuit hea --n 100 --layers 16 --amp 0.4"
E6="--target 6000000 --trace 1 --vram-frac 0.9"; E4="--target 4000000 --trace 1 --vram-frac 0.9"
run ki.1e-5.g16 200 $KI --coef 1e-5 $E6
run hea.1e-6.g16 200 $HE --coef 1e-6 $E4
run noise.5e-6.g16 200 $NO --coef 5e-6 $E6
run qaoa.1e-6.g16 200 $QA --coef 1e-6 $E4
run tfim.5e-6.g16 300 $TF --coef 5e-6 $E4
run ki.3e-6.g16 400 $KI --coef 3e-6 $E6
echo "done after ${SECONDS}s"
