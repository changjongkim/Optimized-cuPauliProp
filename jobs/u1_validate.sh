#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_u1
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/u1/%x-%j.out
# u1: the engine with the automatic choice on one GPU (the operator stays on the GPU while it fits), grouped noise
# channels and the cut on leading Clifford gates (bin/pp_engine_u), next to the binary of campaign f (bin/pp_engine_f),
# on one node. One run per line; logs/u1/<binary>.<id>.out in the format of logs/f.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
OUT=$PP_ROOT/logs/u1
run() {   # run BINARY KIND ID GPUS TIMEOUT ARGS...
  local bin=$1 kind=$2 id=$3 per=$4 tmo=$5; shift 5
  local args="$*"
  {
    case $kind in
      eng) echo "===== engine nodes=1 gpus=$per args: $args --gpus $per"
           MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N 1 -n 1 --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
             stdbuf -oL ./bin/$bin --gpus $per $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)"
           echo "exit=${PIPESTATUS[0]}" ;;
      ref) echo "===== ref nodes=1 gpus=1 args: $args"
           MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N 1 -n 1 --ntasks-per-node=1 -c 32 --gpus-per-node=4 --cpu-bind=none \
             stdbuf -oL ./bin/$bin --mode ref $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)" | awk '!seen[$0]++' | head -40
           echo "exit=${PIPESTATUS[0]}" ;;
      cupp) echo "===== cupp nodes=1 gpus=$per args: $args"
           CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N 1 -n $per --ntasks-per-node=$per -c $((128 / per)) --gpus-per-node=4 --cpu-bind=cores \
             stdbuf -oL ./bin/$bin --mode cupp $args < /dev/null 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk '!seen[$0]++' | head -40
           echo "exit=${PIPESTATUS[0]}" ;;
    esac
  } > $OUT/$bin.$id.out 2>&1
  echo "$(date +%H:%M:%S) $bin $id $(grep -m1 '^exit=' $OUT/$bin.$id.out) $(grep -m1 -E '^wall=[0-9.]+' $OUT/$bin.$id.out | cut -d' ' -f1) $(grep -m1 '^expectation=' $OUT/$bin.$id.out)"
}
KI="--circuit ki_lab --weight 0"; TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"; NO="--circuit ki_noisy --noise 0.005"
SQ="--circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10"; XX="--circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6"
QA="--circuit qaoa --n 100 --layers 4"; HE="--circuit hea --n 100 --layers 16 --amp 0.4"
E6="--target 6000000 --trace 1 --vram-frac 0.9"; E4="--target 4000000 --trace 1 --vram-frac 0.9"; R="--vram-frac 0.9"
# 1. settings that fit one GPU: new and old engine, new and old single-GPU reference
for b in pp_engine_u pp_engine_f; do
  run $b eng ki.1e-4.g1 1 200 $KI --coef 1e-4 $E6;     run $b ref ki.1e-4.ref 1 200 $KI --coef 1e-4 $R
  run $b eng tfim.1e-4.g1 1 200 $TF --coef 1e-4 $E4;   run $b ref tfim.1e-4.ref 1 200 $TF --coef 1e-4 $R
  run $b eng noise.1e-4.g1 1 200 $NO --coef 1e-4 $E6;  run $b ref noise.1e-4.ref 1 200 $NO --coef 1e-4 $R
done
for b in pp_engine_u pp_engine_f; do
  run $b eng tfimsq.1e-4.g1 1 200 $SQ --coef 1e-4 $E4; run $b eng xxz.1e-4.g1 1 200 $XX --coef 1e-4 $E4
  run $b eng qaoa.1e-4.g1 1 200 $QA --coef 1e-4 $E4;   run $b eng hea.1e-5.g1 1 200 $HE --coef 1e-5 $E4
done
# 2. one GPU, settings that do not fit at once: the new engine must choose the host path as before
run pp_engine_u eng ki.3e-5.g1 1 200 $KI --coef 3e-5 $E6
run pp_engine_f eng ki.3e-5.g1 1 200 $KI --coef 3e-5 $E6
run pp_engine_u eng tfim.3e-5.g1 1 200 $TF --coef 3e-5 $E4
run pp_engine_u eng noise.3e-5.g1 1 200 $NO --coef 3e-5 $E6
# 3. four GPUs: unchanged path (kicked Ising), grouped noise channels in the engine and in the cuPauliProp baseline
run pp_engine_u eng ki.1e-5.g4 4 200 $KI --coef 1e-5 $E6
run pp_engine_u eng noise.1e-5.g4 4 200 $NO --coef 1e-5 $E6
run pp_engine_f eng noise.1e-5.g4 4 200 $NO --coef 1e-5 $E6
run pp_engine_u cupp noise.1e-4.cupp4 4 300 $NO --coef 1e-4 $R
run pp_engine_f cupp noise.1e-4.cupp4 4 300 $NO --coef 1e-4 $R
run pp_engine_u cupp noise.1e-4.defer4 4 300 $NO --coef 1e-4 $R --cupp-tuned 1
# 4. one GPU, the largest settings of the campaign (the host path; about 8 minutes)
run pp_engine_u eng noise.1e-5.g1 1 400 $NO --coef 1e-5 $E6
run pp_engine_u eng ki.1e-5.g1 1 400 $KI --coef 1e-5 $E6
run pp_engine_u eng tfim.1e-5.g1 1 400 $TF --coef 1e-5 $E4
echo "done after ${SECONDS}s"
