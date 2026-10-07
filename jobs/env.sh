# Runtime environment shared by all job scripts (source this file).
export PP_ROOT=/pscratch/sd/s/sgkim/kcj/pauliprop
FAB=$(ls -d /opt/cray/libfabric/*/lib64 | tail -1)
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${FAB}:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export CUPAULIPROP_COMM_LIB=$PP_ROOT/bin/libcupp_mpi_prof.so
