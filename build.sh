#!/bin/bash
# Builds the profiling plugin, the unmodified plugin, and the driver with the Cray wrappers.
# Needs PrgEnv-gnu, cray-mpich, craype-accel-nvidia80 and cudatoolkit loaded (login default).
set -euo pipefail
ROOT=/pscratch/sd/s/sgkim/kcj/pauliprop
SP=$ROOT/env/lib/python3.11/site-packages
CUQ=$SP/cuquantum
MATH=$(dirname "$CUDA_HOME")/../math_libs/$(basename "$CUDA_HOME")/lib64
MATH=$(readlink -f "$MATH")
mkdir -p "$ROOT/bin"
FAB=$(ls -d /opt/cray/libfabric/*/lib64 | tail -1)
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${FAB}:${LD_LIBRARY_PATH:-}"

INC="-I$CUQ/include -I$CUDA_HOME/include"
cc -shared -fPIC -std=gnu99 -O2 $INC "$ROOT/src/plugin/prof_mpi.c" -o "$ROOT/bin/libcupp_mpi_prof.so"
cc -shared -fPIC -std=gnu99 -O2 $INC "$ROOT/src/plugin/orig_mpi.c" -o "$ROOT/bin/libcupp_mpi_orig.so"
# times the MPI calls of an unmodified MPI program (LD_PRELOAD), for the distributed CPU baselines
cc -shared -fPIC -std=gnu99 -O2 "$ROOT/src/plugin/mpi_shim.c" -o "$ROOT/bin/libmpi_shim.so"
CC -O2 -std=c++17 $INC "$ROOT/src/driver/ki_prof.cpp" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/ki_prof"
echo "built: $(ls "$ROOT/bin")"
echo "math libs: $MATH"
CC -O2 -std=c++17 $INC "$ROOT/src/driver/ki_split.cpp" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/ki_split"
echo "built ki_split"
# Prototype: parity-owned propagation emulated with virtual ranks on one GPU.
"$CUDA_HOME/bin/nvcc" -O2 -std=c++17 -gencode arch=compute_80,code=sm_80 -c "$ROOT/src/proto/shuffle.cu" -o "$ROOT/bin/shuffle.o"
CC -O2 -std=c++17 $INC "$ROOT/src/proto/lo_virtual.cpp" "$ROOT/bin/shuffle.o" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/lo_virtual"
echo "built lo_virtual"
# Prototype: parity-owned propagation with one MPI rank per GPU.
CC -O2 -std=c++17 $INC "$ROOT/src/proto/lo_mpi.cpp" "$ROOT/bin/shuffle.o" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/lo_mpi"
echo "built lo_mpi"
# Prototype: out-of-core, bucket-at-a-time propagation on one GPU.
CC -O2 -std=c++17 $INC "$ROOT/src/proto/lo_ooc.cpp" "$ROOT/bin/shuffle.o" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/lo_ooc"
echo "built lo_ooc"
# Prototype: bucket-task propagation on the GPUs of one node (threads, no MPI).
CC -O2 -std=c++17 -pthread $INC "$ROOT/src/proto/lo_pool.cpp" "$ROOT/bin/shuffle.o" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/lo_pool"
echo "built lo_pool"
# Unified engine: general circuits, run-time windows, bucket pool, Clifford frame, checkpoint.
CC -O2 -std=c++17 -pthread $INC "$ROOT/src/engine/pp_engine.cpp" "$ROOT/bin/shuffle.o" \
   -L"$CUQ/lib" -l:libcupauliprop.so.0 -L"$CUDA_HOME/lib64" -lcudart -ldl \
   -Wl,-rpath,"$CUQ/lib" -Wl,-rpath,"$SP/cutensor/lib" -Wl,-rpath,"$MATH" -Wl,-rpath,"$CUDA_HOME/lib64" -Wl,-rpath,"$FAB" \
   -o "$ROOT/bin/pp_engine"
echo "built pp_engine"
# Measurement: how fast new pinned host memory is (cudaHostAlloc against touched and registered memory).
CC -O2 -std=c++17 -pthread -I"$CUDA_HOME/include" "$ROOT/src/proto/pin_test.cpp" -L"$CUDA_HOME/lib64" -lcudart -Wl,-rpath,"$CUDA_HOME/lib64" -o "$ROOT/bin/pin_test"
echo "built pin_test"
