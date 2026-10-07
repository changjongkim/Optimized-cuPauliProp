#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -J pp_prof2
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out

# Second one-node measurement: one GPU bound per rank (keeps Cray GTL IPC enabled),
# manual deduplication schedules, profiler runs, and capacity limits.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
OUT=$PP_ROOT/results/prof2_$SLURM_JOB_ID
mkdir -p $OUT
cd $PP_ROOT
export PP_EARLY_DEVICE=0
srun -n 4 --gpus-per-node=4 bash -c 'echo "rank $SLURM_PROCID localid $SLURM_LOCALID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"' | sort

run() {  # run <tag> <ranks> <timeout_s> <driver args...>
  local tag=$1 ranks=$2 tmo=$3; shift 3
  echo "===== $tag ranks=$ranks args: $*"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 \
    > $OUT/$tag.gpu.csv 2>/dev/null &
  local smi=$!
  timeout $tmo srun -n $ranks --gpus-per-node=4 -c 32 --cpu-bind=cores $WRAP \
    ./bin/ki_prof --tag $tag --outdir $OUT "$@" 2>&1 | grep -v -E '^$'
  echo "exit=${PIPESTATUS[0]}"
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
}

A="--coef 1e-4 --weight 0 --vram-frac 0.9"
S="--coef 1e-4 --weight 8 --vram-frac 0.9"
B="--coef 1e-5 --weight 0 --vram-frac 0.9"

WRAP=""
run A_r1_single 1 300 $A --dist 0
run A_r4        4 300 $A
PP_EARLY_DEVICE=1 run A_r4_earlydev 4 120 $A
run S_r4        4 120 $S
CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so run A_r4_origplugin 4 300 $A

# Manual schedules: local application, global deduplication + truncation every k gates.
run A_r4_kd10   4 300 $A --keepdup 1 --dedup-every 10
run A_r4_kd20   4 300 $A --keepdup 1 --dedup-every 20 --trunc-every 20
run A_r1_kd10   1 300 $A --keepdup 1 --dedup-every 10 --dist 0

# Profilers on rank 0, first 1500 gates.
WRAP="$PP_ROOT/jobs/rank0_wrap.sh nsys $OUT/nsys_A_r4"
run P_nsys_r4 4 600 $A --max-gates 1500
WRAP="$PP_ROOT/jobs/rank0_wrap.sh nsys $OUT/nsys_A_r1"
run P_nsys_r1 1 300 $A --max-gates 1500 --dist 0
WRAP="$PP_ROOT/jobs/rank0_wrap.sh perf $OUT/perf_A_r4"
run P_perf_r4 4 300 $A --max-gates 1500
WRAP=""
for f in $OUT/nsys_A_r4.nsys-rep $OUT/nsys_A_r1.nsys-rep; do
  [ -f $f ] && nsys stats -r cuda_api_sum -r osrt_sum --format csv -o ${f%.nsys-rep} $f > /dev/null 2>&1
done
[ -f $OUT/perf_A_r4.perf.data ] && perf report -i $OUT/perf_A_r4.perf.data --stdio --no-children --sort dso,symbol 2>/dev/null | grep -v '^$' | head -70 > $OUT/perf_A_r4.txt

# Capacity: how many terms fit on 4 x 40 GB before the library refuses.
run B_r4            4 420 $B
run B_r4_share10    4 420 $B --exp-share 0.10
run B_r4_kd10       4 420 $B --keepdup 1 --dedup-every 10
echo "done"
