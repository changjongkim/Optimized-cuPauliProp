"""The ladder of kicked Ising down to the memory limit first: queue_s/todo_a0_limit.jsonl.

  env/bin/python3 jobs/a0_items.py            # print what would be written
  env/bin/python3 jobs/a0_items.py --write

The list holds runs that the other lists already hold (same ids, so each runs once); it sorts before them, so a
scale job starts with the largest runs and gives the rest of its time to the short ones. In the order of the
other lists the last step toward the memory limit came after about a hundred minutes of shorter runs and would
not have started if those took longer than expected.
  32 and 64 GPUs: n10.ki 1e-6, 7e-7, 5e-7      (3e-7 does not fit: 4,376 GiB of kept terms)
  256 GPUs:       n12.ki 1e-6 ... 3e-7, then i12.ki.2e-7
  512 GPUs:       n12.ki 3e-6 ... 2e-7, then i12.ki.1.5e-7"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST = {32: ("n10.ki.1e-6", "n10.ki.7e-7", "n10.ki.5e-7"),
         64: ("n10.ki.1e-6", "n10.ki.7e-7", "n10.ki.5e-7"),
         256: ("n12.ki.1e-6", "n12.ki.7e-7", "n12.ki.5e-7", "n12.ki.3e-7", "i12.ki.2e-7"),
         512: ("n12.ki.3e-6", "n12.ki.2e-6", "n12.ki.1e-6", "n12.ki.7e-7", "n12.ki.5e-7", "n12.ki.3e-7", "n12.ki.2e-7", "i12.ki.1.5e-7")}


def items():
    known = {}
    for name in ("todo_a1_n.jsonl", "todo_a2_i.jsonl"):
        for line in open(os.path.join(ROOT, "queue_s", name)):
            if line.strip():
                i = json.loads(line)
                known[i["id"]] = i
    out = []
    for g, names in FIRST.items():
        for n in names:
            i = known[f"{n}.g{g}"]
            if not os.path.exists(os.path.join(ROOT, "queue_s", "done", i["id"])):
                out.append(i)
    return out


if __name__ == "__main__":
    its = items()
    if "--write" not in sys.argv:
        for i in its:
            print(json.dumps(i))
        print(len(its), "runs", file=sys.stderr)
        sys.exit(0)
    path = os.path.join(ROOT, "queue_s", "todo_a0_limit.jsonl")
    with open(path + ".tmp", "w") as f:
        for i in its:
            f.write(json.dumps(i) + "\n")
    os.replace(path + ".tmp", path)
    print(f"wrote {len(its)} runs to {path}")
