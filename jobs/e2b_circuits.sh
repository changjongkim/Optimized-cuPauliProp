#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_e2b
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
# engine: one process per node, host buffers only (GPU transport of Cray MPICH off)
eng() { local nodes=$1 tmo=$2; shift 2; echo "===== engine nodes=$nodes gpus=$((nodes * 4)) args: $*"
  MPICH_GPU_SUPPORT_ENABLED=0 timeout $tmo srun -N $nodes -n $nodes --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine --gpus 4 "$@" 2>&1 | grep -v -E "^$"; echo "exit=${PIPESTATUS[0]}"; }
# cuPauliProp distributed mode: one rank per GPU, GPU-aware MPI, original NVIDIA plugin
cupp() { local nodes=$1 tmo=$2; shift 2; echo "===== cupp nodes=$nodes gpus=$((nodes * 4)) args: $*"
  CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_orig.so MPICH_GPU_SUPPORT_ENABLED=1 timeout $tmo srun -N $nodes -n $((nodes * 4)) --ntasks-per-node=4 -c 32 --gpus-per-node=4 --cpu-bind=cores \
    stdbuf -oL ./bin/pp_engine --mode cupp "$@" 2>&1 | grep -v -E "^$|gtl_init|Current device_id|IPC is disabled|GTL DEBUG|MPICH ERROR|aborting job|application called|^srun" | awk "!seen[\$0]++" | head -12; echo "exit=${PIPESTATUS[0]}"; }

# The other circuits at larger settings on 16 GPUs, and accuracy against the exact 30-qubit values
# (5x6 lattice: 4 steps 0.802867136639, 6 steps 0.790133273060).
eng 4 200 --circuit tfim_hh --theta 0.2 --phi -0.3 --steps 20 --coef 1e-5 --target 4000000
eng 4 150 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 1e-5 --target 4000000
eng 4 200 --circuit tfim_sq --lx 11 --ly 11 --theta 0.2 --phi -0.3 --steps 10 --coef 3e-6 --target 4000000
eng 4 200 --circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --coef 1e-4 --target 4000000
eng 4 200 --circuit qaoa --n 100 --layers 4 --coef 1e-4 --target 4000000
eng 4 150 --circuit hea --n 100 --layers 16 --amp 0.4 --coef 1e-5 --target 4000000
eng 4 200 --circuit ki_noisy --noise 0.005 --coef 1e-5 --target 6000000
for c in 1e-5 3e-6 1e-6; do eng 4 200 --circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps 6 --coef $c --target 4000000; done
for c in 1e-6 1e-7; do eng 4 150 --circuit tfim_sq --lx 5 --ly 6 --theta 0.2 --phi -0.3 --steps 4 --coef $c --target 4000000; done
echo "done"
