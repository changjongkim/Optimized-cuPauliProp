#!/bin/bash
# u7: where the time goes on 16 GPUs (four nodes), for SectorQ and for cuPauliProp's distributed mode (default, and
# with the deduplication deferred to the truncations). cuPauliProp runs with the profiling plugin, which times every
# MPI call the library makes; the engine binary times the library calls by phase. nvidia-smi samples the GPU
# utilization of every node during each run.
#   bash jobs/u7_breakdown.sh <allocation of four nodes> [defer3]     -> logs/u7/<kind>.<circuit>.<cutoff>.out, smi.<...>.csv
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export SLURM_JOB_ID=$1
OUT=$PP_ROOT/logs/u7
BIN=${PP_BIN:-pp_engine_u7}
run() {   # run KIND ID TIMEOUT PLUGIN ARGS...     KIND: eng | cupp      PLUGIN: prof | orig (cupp only)
  local kind=$1 id=$2 tmo=$3 plugin=$4; shift 4
  local args="$*"
  srun --overlap -N 4 -n 4 --ntasks-per-node=1 -c 1 --gpus-per-node=4 --cpu-bind=none bash jobs/smi_sample.sh $OUT/smi.$kind.$id > /dev/null 2>&1 &
  local smi=$!
  sleep 2
  {
    case $kind in
      eng) echo "===== engine nodes=4 gpus=16 args: $args --gpus 4"
           MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun --overlap -N 4 -n 4 --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
             stdbuf -oL ./bin/$BIN --gpus 4 $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)"
           echo "exit=${PIPESTATUS[0]}" ;;
      cupp) echo "===== cupp nodes=4 gpus=16 args: $args"
           CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_$plugin.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun --overlap -N 4 -n 16 --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
             stdbuf -oL ./bin/$BIN --mode cupp $args < /dev/null 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk '!seen[$0]++' | head -60
           echo "exit=${PIPESTATUS[0]}" ;;
    esac
  } > $OUT/$kind.$id.out 2>&1
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  echo "$(date +%H:%M:%S) $kind $id $(grep -m1 '^exit=' $OUT/$kind.$id.out) $(grep -m1 -E '^wall=[0-9.]+' $OUT/$kind.$id.out | cut -d' ' -f1) $(grep -m1 '^expectation=' $OUT/$kind.$id.out)"
}
KI="--circuit ki_lab --weight 0"; TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"; NO="--circuit ki_noisy --noise 0.005"
E6="--target 6000000 --trace 1 --vram-frac 0.9"; E4="--target 4000000 --trace 1 --vram-frac 0.9"; R="--vram-frac 0.9"
if [ "${2:-}" = defer3 ]; then
  # With the deduplication only at the truncations, kicked Ising and TFIM at 1e-5 run out of memory on 16 GPUs;
  # the deferred baseline of these settings also deduplicates every 3 operations (as in campaign f).
  run cupp ki.1e-5.defer 300 prof $KI --coef 1e-5 $R --cupp-tuned 1 --cupp-dedup-every 3
  run cupp tfim.1e-5.defer 300 prof $TF --coef 1e-5 $R --cupp-tuned 1 --cupp-dedup-every 3
  echo "done after ${SECONDS}s"
  exit 0
fi
for spec in "ki.1e-4|$KI --coef 1e-4|$E6" "ki.1e-5|$KI --coef 1e-5|$E6" "tfim.1e-5|$TF --coef 1e-5|$E4" "noise.1e-5|$NO --coef 1e-5|$E6"; do
  IFS='|' read -r id circ eng <<< "$spec"
  run eng $id 200 - $circ $eng
  run cupp $id 300 prof $circ $R
  run cupp $id.defer 300 prof $circ $R --cupp-tuned 1
done
# the same default run with the unmodified plugin: what the timers of the profiling plugin cost
run cupp ki.1e-4.orig 300 orig $KI --coef 1e-4 $R
run cupp ki.1e-5.orig 300 orig $KI --coef 1e-5 $R
echo "done after ${SECONDS}s"
