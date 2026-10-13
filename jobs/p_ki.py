"""paulistrings-rs (third_party/paulistrings-rs) on the kicked Ising circuit of the evaluation: 127 qubits, heavy-hex,
20 steps, X angle pi/4, observable Z62, coefficient cutoff. The circuit comes from the library's own builder, so the
library applies every gate its own way (baseline as it operates).

  one process, host:      .venv/bin/python jobs/p_ki.py --cutoff 1e-4 [--partitions auto]
  one process, devices:   .venv/bin/python jobs/p_ki.py --cutoff 1e-4 --device 0,1,2,3
  one partition per rank: srun -n 16 ... .venv/bin/python jobs/p_ki.py --cutoff 1e-4 --mpi 1 [--device auto]

--rows seed (the library's default: rows drawn from a seed) or cut (its recommendation where the lattice is known:
one block of qubits per partition, here equal ranges of the qubit numbers). Rank 0 prints one JSON line."""
import argparse
import json
import math
import os
import sys
import time

T = "/pscratch/sd/s/sgkim/kcj/pauliprop/third_party/paulistrings-rs"
ap = argparse.ArgumentParser()
ap.add_argument("--cutoff", type=float, required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--theta", type=float, default=math.pi / 4)
ap.add_argument("--obs", type=int, default=62)
ap.add_argument("--partitions", default="")
ap.add_argument("--device", default="")
ap.add_argument("--mpi", type=int, default=0)
ap.add_argument("--rows", default="seed", choices=("seed", "cut"))
ap.add_argument("--edges", action="store_true", help="only compare the lattice with the one of results/f_ext/ir/kis20.ir.json")
a = ap.parse_args()

comm, rank, size = None, 0, 1
if a.mpi:
    import mpi4py
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "serialized"
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

sys.path.insert(0, os.path.join(T, "examples"))
import paulistrings as ps
from common import circuits, observables

if a.edges:
    mine = {tuple(sorted(e)) for e in circuits.heavy_hex_127_edges()}
    ir = json.load(open(os.path.join(os.path.dirname(T), "..", "results", "f_ext", "ir", "kis20.ir.json")))
    ours = {tuple(sorted(o["qubits"])) for o in ir["constructed_state_order"] if len(o["qubits"]) == 2}
    print(json.dumps(dict(edges_library=len(mine), edges_sectorq=len(ours), same=mine == ours)))
    sys.exit(0)

circuit = circuits.heavy_hex_kicked_ising(127, a.steps, a.theta)
obs = observables.single_z(a.obs, 127)
kw = dict(direction="heisenberg")
parts = 1
if a.device:
    kw["device"] = "auto" if a.device == "auto" else ([int(x) for x in a.device.split(",")] if "," in a.device else int(a.device))
    parts = len(kw["device"]) if isinstance(kw["device"], list) else 1
elif a.partitions:
    kw["partitions"] = "auto" if a.partitions == "auto" else int(a.partitions)
    parts = len(ps.numa_nodes()) if a.partitions == "auto" else int(a.partitions)
if comm is not None:
    kw.update(comm=comm, result="local")
    parts = size
if a.rows == "cut" and parts > 1:
    bounds = [round(127 * i / parts) for i in range(parts + 1)]
    kw["partition_row_blocks"] = [list(range(bounds[i], bounds[i + 1])) for i in range(parts)]

if comm is not None:
    comm.Barrier()
t0 = time.perf_counter()
out, stats = obs.propagate_with_stats(circuit, ps.truncation.coeff(a.cutoff), **kw)
if comm is not None:
    comm.Barrier()
seconds = time.perf_counter() - t0
value, terms = complex(out.expectation("z+")).real, len(out)
rss = next((float(l.split()[1]) / 1048576.0 for l in open("/proc/self/status") if l.startswith("VmHWM")), 0.0)   # peak resident memory, GiB
if comm is not None:
    value, terms, rss = comm.allreduce(value), comm.allreduce(terms), comm.allreduce(rss)
if rank == 0:
    rec = dict(expectation=value, wall=round(seconds, 3), terms=terms, cutoff=a.cutoff, steps=a.steps, partitions=parts, ranks=size,
               device=a.device, rows=a.rows, host_rss_gb=round(rss, 2), gates=len(circuit))
    for k in ("peak_terms", "layers"):
        try:
            v = getattr(stats, k)
            rec[k] = v() if callable(v) else v
        except Exception:
            pass
    print("RESULT " + json.dumps(rec), flush=True)
