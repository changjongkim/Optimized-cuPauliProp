#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_h1
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/h1/%x-%j.out
# h1: small and medium operators on 4 and 16 GPUs, the campaign binary (bin/pp_engine_g) next to the engine that
# chooses the rows of the next window while the window runs and gives a GPU one bucket per short window
# (bin/pp_engine_h), on the same nodes. Variants of the new engine: the old rule of two buckets per GPU
# (--long-window 0), always one (--long-window 1e9), and no early bucket per node for tiny operators
# (--min-bucket-early 30000). One run per line; logs/h1/<variant>.<id>.g<GPUs>.out in the format of logs/g.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
OUT=$PP_ROOT/logs/h1
run() {   # run VARIANT BINARY ID NODES ARGS...
  local var=$1 bin=$2 id=$3 nodes=$4; shift 4
  local args="$*" f=$OUT/$var.$id.g$((nodes * 4)).out
  {
    echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $args --gpus 4"
    MPICH_GPU_SUPPORT_ENABLED=0 timeout 300 srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
      stdbuf -oL ./bin/$bin --gpus 4 $args < /dev/null 2>&1 | grep -v -E "^$|^srun: (Job|Step)"
    echo "exit=${PIPESTATUS[0]}"
  } > $f 2>&1
  echo "$var $id g$((nodes * 4)) $(grep -m1 -E '^wall=[0-9.]+' $f | cut -d' ' -f1) $(grep -m1 '^expectation=' $f) $(grep -m1 '^exit=' $f)"
}
KI="--circuit ki_lab --weight 0"; TF="--circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20"; NO="--circuit ki_noisy --noise 0.005"
E6="--target 6000000 --trace 1 --vram-frac 0.9"; E4="--target 4000000 --trace 1 --vram-frac 0.9"
for spec in "ki.1e-3|$KI --coef 1e-3 $E6" "ki.1e-4|$KI --coef 1e-4 $E6" "ki.5e-5|$KI --coef 5e-5 $E6" "ki.3e-5|$KI --coef 3e-5 $E6" \
            "tfim.3e-5|$TF --coef 3e-5 $E4" "noise.3e-5|$NO --coef 3e-5 $E6" "ki.1e-5|$KI --coef 1e-5 $E6"; do
  IFS='|' read -r id args <<< "$spec"
  for nodes in 4 1; do
    run g pp_engine_g $id $nodes $args
    run h pp_engine_h $id $nodes $args
    run h_two pp_engine_h $id $nodes $args --long-window 0
    run h_one pp_engine_h $id $nodes $args --long-window 1e9
    run h_early pp_engine_h $id $nodes $args --min-bucket-early 30000
  done
done
echo "done after ${SECONDS}s"
