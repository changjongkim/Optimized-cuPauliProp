#!/usr/bin/env python3
"""Exact state-vector value for the square-lattice Ising circuit (engine circuit `tfim_sq`).

Usage: exact_sv.py <lx> <ly> <steps> <theta/pi> <phi/pi>

Applies the same gates as the engine (per step: exp(-i theta/2 X) on every qubit, then
exp(-i phi/2 Z Z) on every edge) to |0...0> with cuStateVec and prints <Z> on the centre qubit.
30 qubits need 16 GiB of GPU memory.
"""
import sys

import cupy as cp
import numpy as np
from cuquantum import cudaDataType
from cuquantum.bindings import custatevec as cusv

C64 = int(cudaDataType.CUDA_C_64F)

lx, ly, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
theta, phi = float(sys.argv[4]) * np.pi, float(sys.argv[5]) * np.pi
n = lx * ly
edges = [(y * lx + x, y * lx + x + 1) for y in range(ly) for x in range(lx - 1)]
edges += [(y * lx + x, (y + 1) * lx + x) for y in range(ly - 1) for x in range(lx)]
obs = (ly // 2) * lx + lx // 2

sv = cp.zeros(1 << n, dtype=cp.complex128)
sv[0] = 1.0
h = cusv.create()
X, Z = int(cusv.Pauli.X), int(cusv.Pauli.Z)
for _ in range(steps):
    for q in range(n):
        # apply_pauli_rotation applies exp(i t P); the engine's gate is exp(-i angle/2 P)
        cusv.apply_pauli_rotation(h, sv.data.ptr, C64, n, -theta / 2, [X], [q], 1, [], [], 0)
    for a, b in edges:
        cusv.apply_pauli_rotation(h, sv.data.ptr, C64, n, -phi / 2, [Z, Z], [a, b], 2, [], [], 0)
out = np.zeros(1, dtype=np.float64)
cusv.compute_expectations_on_pauli_basis(h, sv.data.ptr, C64, n, out.ctypes.data, [[Z]], 1, [[obs]], [1])
cusv.destroy(h)
print(f"exact <Z_{obs}> on {lx}x{ly} ({n} qubits), {steps} steps: {out[0]:.12f}")
