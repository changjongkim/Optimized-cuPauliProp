"""cuPauliProp's default distributed mode on the original circuits, 32-512 GPUs: queue_s/todo_a0o_orig.jsonl.

  env/bin/python3 jobs/os_items.py            # print what would be written
  env/bin/python3 jobs/os_items.py --write    # write the list

Campaign o (jobs/o_items.py) gave cuPauliProp the circuit as it is on 1-16 GPUs. The scale campaign still holds the
runs that got the list of operations left by SectorQ's Clifford frame. This list repeats the default-mode runs of the
two circuits with Clifford gates that were run above 16 GPUs (kicked Ising, noisy kicked Ising) with --frame 0, under
the id "o.<id of the run with the reduced list>" and the ladder "o.<ladder>"; on TFIM heavy-hex the two inputs are the
same. The runs with deferred dedup are not repeated: that mode is put together by this work, with its input.
The list sorts after the ladder of SectorQ (todo_a0_limit.jsonl) and before its sweeps, so a scale job that is waiting
takes these runs second."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "queue_s", "todo_a0o_orig.jsonl")
CLIFFORD = ("--circuit ki_lab", "--circuit ki_noisy")


def wall(sid):
    """Seconds the run with the reduced list took (None: it did not finish)."""
    try:
        r = json.load(open(os.path.join(ROOT, "queue_s", "done", sid)))
    except (OSError, ValueError):
        return None
    return r.get("wall") if r.get("status") == "ok" else None


def items():
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(ROOT, "queue_s", "todo*.jsonl"))):
        if path == OUT:
            continue
        for line in open(path):
            if not line.strip():
                continue
            i = json.loads(line)
            a = i.get("args", "")
            if i["kind"] != "cupp" or i["id"] in seen or "--cupp-tuned" in a or "--frame" in a:
                continue
            if not any(c in a for c in CLIFFORD):
                continue
            seen.add(i["id"])
            w = wall(i["id"])
            j = dict(i, id="o." + i["id"], args=a + " --frame 0", est=int(2 * w + 60) if w else 300)
            if i.get("ladder"):
                j["ladder"] = "o." + i["ladder"]
            out.append(j)
    out.sort(key=lambda j: (j["nodes"], "ki_noisy" in j["args"], not j.get("ladder"), -j["coef"]))
    return out


if __name__ == "__main__":
    its = items()
    by = {}
    for i in its:
        g = i["nodes"] * i["per"]
        by.setdefault(g, [0, 0])
        by[g][0] += 1
        by[g][1] += i["est"]
    if "--write" not in sys.argv:
        for i in its:
            print(json.dumps(i))
        print(f"{len(its)} runs; by GPU count (runs, minutes expected if all finish): "
              f"{ {g: (n, round(s / 60)) for g, (n, s) in sorted(by.items())} }", file=sys.stderr)
        sys.exit(0)
    with open(OUT + ".tmp", "w") as f:
        for i in its:
            f.write(json.dumps(i) + "\n")
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {len(its)} runs to {OUT}; by GPU count: { {g: n for g, (n, s) in sorted(by.items())} }")
