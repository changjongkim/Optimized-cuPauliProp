#!/bin/bash
# Builds paulistrings-rs (github.com/lkdvos/paulistrings-rs, commit 99c3436 in third_party/paulistrings-rs) with its
# MPI and CUDA features into its own venv. Run on a login node (needs the network for the crates):
#   bash jobs/p_build.sh > logs/p/build.log 2>&1
set -eo pipefail
ROOT=/pscratch/sd/s/sgkim/kcj/pauliprop
T=$ROOT/third_party/paulistrings-rs
module load rust/stable            # not the llvm module: loading it switches cray-mpich off
export RUSTUP_TOOLCHAIN=stable        # the repository pins 1.94.0; the module's stable toolchain is newer
export CARGO_HOME=$ROOT/third_party/cargo-home CARGO_TARGET_DIR=$T/target
export LIBCLANG_PATH=/usr/lib64       # libclang for the bindgen of the mpi crate
export PIP_CACHE_DIR=$ROOT/third_party/pip-cache PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple   # the account's pip config names an index that does not resolve here
cd $T
[ -d .venv ] || $ROOT/env/bin/python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip maturin numpy
MPICC="cc -shared" .venv/bin/pip install -q --no-binary=mpi4py mpi4py
rustc --version; cargo --version; mpicc -show | cut -c1-200
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV VIRTUAL_ENV=$T/.venv PATH=$T/.venv/bin:$PATH .venv/bin/maturin develop --release --features mpi,cuda -j 16
.venv/bin/python -c "import paulistrings as p; print('built: mpi', p.mpi_available(), 'cuda', p.cuda_available())"
