#!/usr/bin/env python3
"""Tables of the README that come from campaign g (1-16 GPUs), as Markdown.

  python3 analysis/readme_g.py            print every block
  python3 analysis/readme_g.py --write    rewrite the blocks of README.md in place

A block of the README is the text between <!-- g:NAME --> and <!-- /g:NAME -->. Cells: seconds, or 메모리 부족
(out of memory), 시간 초과 (over the 40 minute limit, measured or predicted from the easier steps), – (not run: an
easier step of the same ladder already failed), 측정 중 (not finished yet).
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_g as cg
sys.path.insert(0, os.path.join(cg.ROOT, "figures"))
import make_eval_figures as mf                      # parse(): the logs of the scaling jobs, one log per GPU count

ROOT = cg.ROOT
GPUS = cg.GPUS
KO = {"ki": "Kicked Ising", "tfim": "TFIM heavy-hex", "noise": "잡음 있는 kicked Ising", "tfimsq": "2D TFIM 11×11",
      "xxz": "XXZ 8×8", "qaoa": "QAOA 100 qubit", "hea": "Ansatz 100 qubit"}
MAIN = ("ki", "tfim", "noise")
ALL = ("ki", "tfim", "tfimsq", "xxz", "qaoa", "hea", "noise")
FAIL = {"oom": "메모리 부족", "tl": "시간 초과", "predicted_tl": "시간 초과", "skipped_after_failure": "–", "fail": "실패"}

items, state = cg.load()
ids = {i["id"] for i in items}
_text = {}
_ladders = {}
for _i in items:
    if _i.get("ladder"):
        _ladders.setdefault(_i["ladder"], []).append(_i["id"])
_ladder_of = {i["id"]: i["ladder"] for i in items if i.get("ladder")}


def waiting(item_id):
    """What to print for a run without a record: – when an earlier step of its ladder failed (it will not be run)."""
    if item_id not in ids:
        return ""
    steps = _ladders.get(_ladder_of.get(item_id), [])
    for e in steps[:steps.index(item_id)] if item_id in steps else []:
        if e in state and state[e]["status"] != "ok":
            return "–"
    return "측정 중"


def log_text(s):
    if not s or not s.get("log"):
        return ""
    if s["log"] not in _text:
        try:
            _text[s["log"]] = open(os.path.join(ROOT, s["log"]), errors="replace").read()
        except OSError:
            _text[s["log"]] = ""
    return _text[s["log"]]


def ok(s):
    return bool(s) and s["status"] == "ok" and s.get("wall") is not None


def sec(w):
    return ("%.2f초" % w) if w < 10 else ("%.1f초" % w) if w < 100 else ("{:,.0f}초".format(w))


def mult(x):
    return ("%.1f배" % x) if x < 9.95 else "{:,.0f}배".format(x)


def cell(item_id, against=None):
    """Seconds of a run, with the multiple of `against` (a SectorQ run) when both finished."""
    s = state.get(item_id) if isinstance(item_id, str) else item_id
    if s is None:
        return waiting(item_id) if isinstance(item_id, str) else ""
    if not ok(s):
        return FAIL.get(s["status"], s["status"])
    return sec(s["wall"]) + (" (%s)" % mult(s["wall"] / against["wall"]) if ok(against) else "")


def deferred(c, x, g):
    return cg.best([state.get(f"{t}.{c}.{x}.g{g}") for t in ("defer", "defer3", "defer2")])


def deferred_cell(c, x, g, against=None):
    s = deferred(c, x, g)
    if s is None:
        return waiting(f"defer.{c}.{x}.g{g}")
    return cell(s, against)


def ladder(prefix, suffix, done_only=False):
    pat = re.compile(re.escape(prefix) + r"\.([0-9.]+e-[0-9]+)\." + re.escape(suffix) + "$")
    pool = state if done_only else ids
    return sorted((m.group(1) for m in map(pat.match, pool) if m), key=float, reverse=True)


def facts(c, x):
    """(boundary terms, largest count inside a window) of a problem, from any SectorQ run of it."""
    for g in reversed(GPUS):
        f = cg.engine_facts("", state.get(f"eng.{c}.{x}.g{g}"))
        if f and f[0]:
            return f
    return (None, None, None)


def ko_terms(n):
    if n is None:
        return ""
    if n >= 1e8:
        v = n / 1e8
        return ("%.1f억" % v) if v < 10 else "{:,.0f}억".format(v)
    return "{:,.0f}만".format(n / 1e4) if n >= 1e5 else "%.1f만" % (n / 1e4)


def value(s):
    """An expectation value with 12 significant digits."""
    if not s or s.get("expectation") is None:
        return ""
    v = s["expectation"]
    digits = 11 - (math.floor(math.log10(abs(v))) if v else 0)
    return ("%.*f" % (max(digits, 0), v)).replace("-", "−")


def table(head, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


# ----------------------------------------------------------------------------------------------------------
def b_progress():
    mine = {k: s for k, s in state.items() if not k.startswith("test.")}
    opened = sum(1 for i in items if i["id"] not in state)
    n = lambda st: sum(1 for s in mine.values() if s["status"] == st and not s.get("reused"))
    reused = sum(1 for s in mine.values() if s.get("reused"))
    return (f"실행 {len(items):,}건 가운데 {len(items) - opened:,}건이 끝났다(완료 {n('ok'):,}, 메모리 부족 {n('oom')}, 시간 초과 {n('tl') + n('predicted_tl')}, "
            f"더 쉬운 칸에서 실패하여 돌리지 않음 {n('skipped_after_failure')}, 캠페인 f에서 가져온 기준선 실행 {reused}). 남은 {opened:,}건은 측정 중이다.")


def b_agree():
    values = {}
    for k, s in state.items():
        p = k.split(".")
        if s["status"] == "ok" and s.get("expectation") is not None and p[0] in ("eng", "cupp", "defer", "defer2", "defer3", "spd", "orqa", "julia", "ppvm", "qiskit"):
            m = re.match(r"[a-z0-9]+\.([a-z]+)\.([0-9.]+e-[0-9]+)\.[gn]\d+$", k)
            if m:
                values.setdefault(m.group(1, 2), {}).setdefault(p[0], []).append(s["expectation"])
    multi = {k: v for k, v in values.items() if sum(len(x) for x in v.values()) > 1}
    spread = max((max(sum(v.values(), [])) - min(sum(v.values(), [])) for v in multi.values()), default=0.0)
    impl = sorted({t for v in multi.values() for t in v})
    rows = []
    for c in ALL:
        mine = {k: v for k, v in multi.items() if k[0] == c}
        if not mine:
            continue
        runs = sum(len(x) for v in mine.values() for x in v.values())
        worst = max(max(sum(v.values(), [])) - min(sum(v.values(), [])) for v in mine.values())
        others = sorted({t for v in mine.values() for t in v} - {"eng"})
        rows.append((KO[c], len(mine), runs, "%.1e" % worst, ", ".join(others)))
    return (f"같은 truncation 규칙을 쓰는 구현의 완료한 실행을 설정(회로, cutoff)별로 모으면, 실행이 둘 이상인 설정 {len(multi)}개 모두에서 "
            f"기대값의 최대 차이가 {spread:.1e}이다.\n\n" + table(["회로", "설정 수", "실행 수", "기대값의 최대 차이", "SectorQ와 대조한 구현"], rows, ["---", "---:", "---:", "---:", "---"])
            + "\n\n표의 구현 이름은 실행 식별자이다: `cupp` cuPauliProp(GPU 1장과 분산 기본), `defer*` dedup을 미룬 cuPauliProp, `spd` SPD, `orqa` ORQA 재현, `julia` PauliPropagation.jl, `ppvm` ppvm, `qiskit` Qiskit pauli-prop.")


def largest_both(c, g, base_g):
    both = [x for x in ladder(f"eng.{c}", f"g{g}") if ok(state.get(f"eng.{c}.{x}.g{g}")) and ok(state.get(f"cupp.{c}.{x}.g{base_g}"))]
    return both[-1] if both else None


def b_speed(g=16):
    out = []
    for base_g, title in ((1, f"cuPauliProp이 GPU 1장으로 끝내는 가장 큰 설정을 {g} GPU에서 잰 것"), (g, f"cuPauliProp 분산 모드가 {g} GPU로 끝내는 가장 큰 설정")):
        rows = []
        for c in ALL:
            x = largest_both(c, g, base_g)
            if not x:
                continue
            e = state.get(f"eng.{c}.{x}.g{g}")
            rows.append((KO[c], x, ko_terms(facts(c, x)[0]), value(e), cell(e), deferred_cell(c, x, g, e), cell(f"cupp.{c}.{x}.g{g}", e)))
        out.append(f"**{title}.**\n\n" + table(["회로", "cutoff", "구간 경계의 항 수", "기대값", "SectorQ", "cuPauliProp dedup 미룸", "cuPauliProp 기본"], rows, ["---", "---", "---:", "---", "---:", "---:", "---:"]))
    return "\n\n".join(out)


BACKENDS = (("spd", "SPD", "GPU 1장", "n1"), ("orqa", "ORQA (재현)", "4 노드, 256 rank", "n4"), ("monoprop", "MonoProp", "4 노드, 자체 truncation 규칙", "n4"),
            ("julia", "PauliPropagation.jl", "1 노드, 64 thread", "n1"), ("ppvm", "ppvm", "1 노드, 64 thread", "n1"), ("qiskit", "Qiskit pauli-prop", "thread 1개", "n1"))
SOTA_CUTS = ("1e-3", "1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6")


def b_single_node():
    """One node: every implementation on the resources of one node (4 GPUs, 64 cores), time by cutoff."""
    cuts = ("1e-3", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6")
    out = []
    for c in MAIN:
        rows = [["구간 경계의 항 수", ""] + [ko_terms(facts(c, x)[0]) for x in cuts],
                ["**SectorQ**", "4 GPU"] + [cell(f"eng.{c}.{x}.g4") for x in cuts],
                ["**SectorQ**", "GPU 1장"] + [cell(f"eng.{c}.{x}.g1") for x in cuts],
                ["cuPauliProp 분산 (기본)", "4 GPU"] + [cell(f"cupp.{c}.{x}.g4") for x in cuts],
                ["cuPauliProp 분산 (dedup 미룸)", "4 GPU"] + [deferred_cell(c, x, 4) for x in cuts],
                ["cuPauliProp", "GPU 1장"] + [cell(f"cupp.{c}.{x}.g1") for x in cuts]]
        for b, name, res in (("spd", "SPD", "GPU 1장"), ("orqa", "ORQA (재현)", "64 rank"), ("monoprop", "MonoProp", "8 rank × 8 thread"),
                             ("julia", "PauliPropagation.jl", "64 thread"), ("ppvm", "ppvm", "64 thread"), ("qiskit", "Qiskit pauli-prop", "thread 1개")):
            if any(f"{b}.{c}.{x}.n1" in ids for x in ladder(f"{b}.{c}", "n1")):
                rows.append([name, res] + [cell(f"{b}.{c}.{x}.n1") for x in cuts])
        out.append(f"**{KO[c]}** (한 노드, 실행 시간)\n\n" + table(["구현", "자원"] + list(cuts), rows, ["---", "---"] + ["---:"] * len(cuts)))
    return "\n\n".join(out)


def mult_table(columns, ours_of):
    """Their seconds / SectorQ seconds at the largest setting both finished; columns: (title, prefix format, suffix, SectorQ suffix)."""
    rows = []
    for c in ALL:
        row = [KO[c]]
        for _, prefix, suffix, mine in columns:
            pick = ""
            xs = sorted({x for t in (("defer", "defer3", "defer2") if prefix == "defer" else (prefix,)) for x in ladder(f"{t}.{c}", suffix, True)}, key=float)
            for x in xs:
                a = state.get(f"eng.{c}.{x}.{mine}")
                b = deferred(c, x, int(suffix[1:])) if prefix == "defer" else state.get(f"{prefix}.{c}.{x}.{suffix}")
                if ok(a) and ok(b):
                    pick = f"{mult(b['wall'] / a['wall'])} ({x})"
                    break
            row.append(pick)
        rows.append(row)
    return table(["회로"] + [t for t, _, _, _ in columns], rows)


def b_node_mult():
    return mult_table((("cuPauliProp 분산 기본 (4 GPU)", "cupp", "g4", "g4"), ("cuPauliProp dedup 미룸 (4 GPU)", "defer", "g4", "g4"),
                       ("ORQA 재현 (64 rank)", "orqa", "n1", "g4"), ("MonoProp (64 코어)", "monoprop", "n1", "g4"),
                       ("PauliPropagation.jl (64 thread)", "julia", "n1", "g4"), ("ppvm (64 thread)", "ppvm", "n1", "g4"),
                       ("Qiskit pauli-prop (thread 1개)", "qiskit", "n1", "g4"), ("SPD (GPU 1장, SectorQ도 GPU 1장)", "spd", "n1", "g1")), None)


def b_multi_mult():
    return mult_table((("cuPauliProp 분산 기본 (8 GPU)", "cupp", "g8", "g8"), ("ORQA 재현 (2 노드)", "orqa", "n2", "g8"), ("MonoProp (2 노드)", "monoprop", "n2", "g8"),
                       ("cuPauliProp 분산 기본 (16 GPU)", "cupp", "g16", "g16"), ("cuPauliProp dedup 미룸 (16 GPU)", "defer", "g16", "g16"),
                       ("ORQA 재현 (4 노드)", "orqa", "n4", "g16"), ("MonoProp (4 노드)", "monoprop", "n4", "g16")), None)


def b_strong_sota(c="ki"):
    """The representative circuit on 4, 8 and 16 GPUs: every implementation that runs on several nodes, time by cutoff."""
    cuts = ("1e-4", "3e-5", "1e-5", "5e-6", "3e-6", "2e-6", "1e-6")
    out = []
    for g, n in ((4, 1), (8, 2), (16, 4)):
        e = lambda x: state.get(f"eng.{c}.{x}.g{g}")
        rows = [["**SectorQ**"] + [cell(f"eng.{c}.{x}.g{g}") for x in cuts],
                ["cuPauliProp 분산 (기본)"] + [cell(f"cupp.{c}.{x}.g{g}", e(x)) for x in cuts],
                ["cuPauliProp 분산 (dedup 미룸)"] + [deferred_cell(c, x, g, e(x)) for x in cuts],
                ["ORQA (재현)"] + [cell(f"orqa.{c}.{x}.n{n}", e(x)) for x in cuts],
                ["MonoProp"] + [cell(f"monoprop.{c}.{x}.n{n}", e(x)) for x in cuts]]
        out.append(f"**{KO[c]}, {g} GPU ({n} 노드).** 괄호는 SectorQ 대비이다.\n\n" + table(["구현"] + list(cuts), rows))
    return "\n\n".join(out)


def b_same_error(g=16, n=4):
    rows = []
    for c in ALL:
        pts = sorted(((float(x), s["expectation"], s["wall"], x) for x in ladder(f"eng.{c}", f"g{g}", True) for s in [state.get(f"eng.{c}.{x}.g{g}")]
                      if ok(s) and s.get("expectation") is not None), reverse=True)
        if len(pts) < 3:
            continue
        ref = pts[-1][1]
        curve = [(abs(v - ref), t, x) for _, v, t, x in pts]
        tail = [curve[-1]]
        for e in reversed(curve[:-1]):
            if e[0] <= tail[-1][0]:
                break
            tail.append(e)
        tail.reverse()
        last = None
        for x in sorted(ladder(f"monoprop.{c}", f"n{n}", True), key=float, reverse=True):
            m = state.get(f"monoprop.{c}.{x}.n{n}")
            if not ok(m) or m.get("expectation") is None or float(x) > float(tail[0][2]):
                continue
            err = abs(m["expectation"] - ref)
            for a, b in zip(tail, tail[1:]):
                if a[0] >= err >= b[0]:
                    f = (math.log(a[0]) - math.log(err)) / (math.log(a[0]) - math.log(b[0])) if b[0] > 0 else (a[0] - err) / a[0]
                    ours = a[1] * (b[1] / a[1]) ** f if b[0] > 0 else a[1] + (b[1] - a[1]) * f
                    same = state.get(f"eng.{c}.{x}.g{g}")
                    last = (KO[c], x, "%.4f" % err, sec(m["wall"]), sec(ours), mult(m["wall"] / ours), mult(m["wall"] / same["wall"]) if ok(same) else "")
                    break
        if last:
            rows.append(last)
    return table(["회로", "MonoProp이 끝낸 가장 작은 cutoff", "그 값의 오차", "MonoProp (4 노드)", "같은 오차에서의 SectorQ (16 GPU)", "같은 오차에서 SectorQ 대비", "같은 cutoff에서 SectorQ 대비"], rows,
                 ["---", "---", "---:", "---:", "---:", "---:", "---:"])


def b_given():
    rows = []
    groups = {}
    for i in sorted(ids):
        m = re.match(r"giv\.([a-z]+)\.([a-z]+)\.(.+)\.([ng])(\d+)$", i)
        if m:
            groups.setdefault(m.group(1, 2, 3), {})[int(m.group(5))] = i
    names = {b: n for b, n, _, _ in BACKENDS}
    for (b, c, x), d in sorted(groups.items(), key=lambda kv: (list(names).index(kv[0][0]), kv[0][1], -float(kv[0][2]))):
        unit = "GPU" if b == "spd" else "노드"
        own = cell(f"{b}.{c}.{x}.n1")
        rows.append((names[b], KO[c], x, f"{own}", ", ".join(f"{n} {unit}: {cell(d[n])}" for n in sorted(d))))
    return table(["구현", "회로", "cutoff", "자기 자원 (SPD는 GPU 1장, 나머지는 1 노드)", "자원을 더 주었을 때"], rows, ["---", "---", "---", "---:", "---"])


def b_threads():
    rows = []
    groups = {}
    for i in sorted(ids):
        m = re.match(r"thr\.([a-z]+)\.([a-z]+)\.(.+)\.c(\d+)$", i)
        if m:
            groups.setdefault(m.group(1, 2, 3), {})[int(m.group(4))] = i
    names = {b: n for b, n, _, _ in BACKENDS}
    for (b, c, x), d in sorted(groups.items(), key=lambda kv: (list(names).index(kv[0][0]), kv[0][1], -float(kv[0][2]))):
        base = cell(f"{b}.{c}.{x}.n1")
        get = lambda n: cell(d[n]) if n in d else ""
        rows.append((names[b], KO[c], x, base if b == "qiskit" else "", get(16), get(32), base if b != "qiskit" else get(64)))
    return table(["구현", "회로", "cutoff", "1 코어", "16 코어", "32 코어", "64 코어"], rows, ["---", "---", "---", "---:", "---:", "---:", "---:"])


def b_gpus():
    out = []
    for c in MAIN:
        rows = []
        for x in ("3e-5", "1e-5", "3e-6"):
            f = facts(c, x)
            for name, fn in (("SectorQ", lambda g: cell(f"eng.{c}.{x}.g{g}")), ("cuPauliProp dedup 미룸", lambda g: "" if g == 1 else deferred_cell(c, x, g, state.get(f"eng.{c}.{x}.g{g}"))),
                             ("cuPauliProp 기본", lambda g: cell(f"cupp.{c}.{x}.g{g}", state.get(f"eng.{c}.{x}.g{g}")))):
                rows.append([f"{x} ({ko_terms(f[0])})" if name == "SectorQ" else "", name] + [fn(g) for g in GPUS])
        out.append(f"**{KO[c]}**\n\n" + table(["cutoff (구간 경계의 항 수)", "방식"] + [f"{g} GPU" for g in GPUS], rows, ["---", "---"] + ["---:"] * len(GPUS)))
    return "\n\n".join(out)


def b_reach():
    rows = []
    for c in ALL:
        for name, tags in (("SectorQ", ("eng",)), ("cuPauliProp 기본", ("cupp",)), ("cuPauliProp dedup 미룸", ("defer", "defer3", "defer2"))):
            row = [KO[c] if name == "SectorQ" else "", name]
            for g in GPUS:
                if name.endswith("미룸") and g == 1:
                    row.append("")
                    continue
                done = [x for t in tags for x in ladder(f"{t}.{c}", f"g{g}", True) if ok(state.get(f"{t}.{c}.{x}.g{g}"))]
                still = any(f"{tags[0]}.{c}.{x}.g{g}" not in state for x in ladder(f"{tags[0]}.{c}", f"g{g}"))
                if not done:
                    row.append("측정 중" if still else "")
                    continue
                x = min(done, key=float)
                row.append(f"{x} ({ko_terms(facts(c, x)[0])})" + (" *" if still else ""))
            rows.append(row)
    return table(["회로", "방식"] + [f"{g} GPU" for g in GPUS], rows, ["---", "---"] + ["---:"] * len(GPUS)) + "\n\n\\* 더 작은 cutoff의 실행이 아직 측정 중이다."


def b_largest(g=16):
    rows = []
    for c in ALL:
        done = [x for x in ladder(f"eng.{c}", f"g{g}", True) if ok(state.get(f"eng.{c}.{x}.g{g}"))]
        if not done:
            continue
        x = min(done, key=float)
        s = state[f"eng.{c}.{x}.g{g}"]
        f = cg.engine_facts("", s) or (None, None, None)
        busiest = re.search(r"([\d.]+) GiB on the busiest node", log_text(s))
        base = [y for y in ladder(f"cupp.{c}", f"g{g}", True) if ok(state.get(f"cupp.{c}.{y}.g{g}"))]
        rows.append((KO[c], x, ko_terms(f[0]), ko_terms(f[1]), sec(s["wall"]), ("%.0f GiB" % f[2]) if f[2] else "", ("%.0f GiB" % float(busiest.group(1))) if busiest else "",
                     value(s), (min(base, key=float) + " (" + ko_terms(facts(c, min(base, key=float))[0]) + ")") if base else ""))
    return table(["회로", "cutoff", "구간 경계의 항 수", "구간 안 최대 항 수", "시간", "host memory 합", "가장 많이 쓴 노드", "기대값", "cuPauliProp 분산이 끝낸 가장 작은 cutoff"], rows,
                 ["---", "---", "---:", "---:", "---:", "---:", "---:", "---", "---:"])


def b_strong():
    rows = []
    for x in ("3e-6", "1e-5"):
        for c in MAIN:
            rs = [state.get(f"eng.{c}.{x}.g{g}") for g in GPUS]
            t1 = rs[0]["wall"] if ok(rs[0]) else None
            cells = [cell(f"eng.{c}.{x}.g{g}") + (f" ({mult(t1 / s['wall'])})" if t1 and ok(s) and g > 1 else "") for g, s in zip(GPUS, rs)]
            rows.append([f"{KO[c]}, cutoff {x} ({ko_terms(facts(c, x)[0])} 항)"] + cells)
    return table(["문제"] + [f"{g} GPU" for g in GPUS], rows)


def weak_run(load, c, g, tag=""):
    x = cg.WEAK[load][g]
    return state.get(f"weak{load}{tag}.{c}.{x}.g{g}") or state.get(f"{ {'': 'eng', 'c': 'cupp', 'd': 'defer'}[tag]}.{c}.{x}.g{g}"), x


def b_weak():
    out = []
    for load, label in (("S", "작은 부하 (4 GPU에서 cutoff 3e-5)"), ("L", "큰 부하 (4 GPU에서 cutoff 1e-5)")):
        rows = [["cutoff"] + [cg.WEAK[load][g] for g in GPUS]]
        for c in MAIN:
            rate = []
            for g in GPUS:
                s, x = weak_run(load, c, g)
                m = re.search(r"^integrity: (\d+) term checks", log_text(s), re.M) if ok(s) else None
                rate.append((s, int(m.group(1)) / s["wall"] / g) if m else (s, None))
            base = rate[0][1]
            cells = []
            for g, (s, r) in zip(GPUS, rate):
                x = cg.WEAK[load][g]
                item = f"weak{load}.{c}.{x}.g{g}" if f"weak{load}.{c}.{x}.g{g}" in ids else f"eng.{c}.{x}.g{g}"
                cells.append(f"{sec(s['wall'])} ({100 * r / base:.0f}%)" if (r and base) else (cell(s) if s else waiting(item)))
            rows.append([f"**SectorQ**, {KO[c]}"] + cells)
            for tag, name in (("c", "cuPauliProp 기본"), ("d", "cuPauliProp dedup 미룸")):
                cells = []
                for g in GPUS:
                    s, x = weak_run(load, c, g, tag)
                    if tag == "d" and g == 1:
                        cells.append("")
                        continue
                    if tag == "d" and s is None:
                        s = deferred(c, x, g)
                    first = f"weak{load}{tag}.{c}.{x}.g{g}"
                    cells.append(cell(s) if s else waiting(first if first in ids else f"{ {'c': 'cupp', 'd': 'defer'}[tag]}.{c}.{x}.g{g}"))
                rows.append([f"{name}, {KO[c]}"] + cells)
            for b, name in (("orqa", "ORQA (재현)"), ("monoprop", "MonoProp")):
                cells = [cell(f"{b}.{c}.{cg.WEAK[load][g]}.n1" if g == 4 else f"weak{load}.{b}.{c}.{cg.WEAK[load][g]}.g{g}") for g in GPUS]
                if any(cells):
                    rows.append([f"{name}, {KO[c]}"] + cells)
        out.append(f"**{label}.** 값은 실행 시간이다. SectorQ의 괄호는 GPU 한 장이 1초에 처리한 항 수를 1 GPU와 비교한 것이다. CPU 구현은 1 GPU와 2 GPU 자리에서 16 코어와 32 코어를, 4 / 8 / 16 GPU 자리에서 1 / 2 / 4 노드를 쓴다.\n\n"
                   + table(["실행"] + [f"{g} GPU" for g in GPUS], rows))
    return "\n\n".join(out)


def num(pat, text):
    m = re.search(pat, text, re.M)
    return float(m.group(1)) if m else None


def b_parts(g=16):
    rows = []
    for c in MAIN:
        for x in ("1e-4", "3e-5", "1e-5", "3e-6", "1e-6"):
            for gg in (4, g):
                s = state.get(f"eng.{c}.{x}.g{gg}")
                if not ok(s):
                    continue
                t = log_text(s)
                w = s["wall"]
                ops, part = num(r"operations=([\d.]+)", t), num(r"partition=([\d.]+)", t)
                cp = (num(r"host->gpu=([\d.]+)", t) or 0) + (num(r"gpu->host=([\d.]+)", t) or 0)
                idle, exch, wait = num(r"idle=([\d.]+)", t), num(r"GB sent, ([\d.]+) s per node", t), num(r"waiting for the slowest node=([\d.]+)", t)
                if ops is None:
                    continue
                p = lambda v, per=1: "" if v is None else "%.0f%%" % (100 * v / per / w)
                shares = [ops / gg, exch or 0, cp / gg, (part or 0) / gg, (idle or 0) / gg, wait or 0]
                rows.append((KO[c], x, ko_terms(facts(c, x)[0]), gg, sec(w), p(ops, gg), p(exch) if exch else "없음", p(cp, gg), p(part, gg), p(idle, gg), p(wait) if gg > 4 else "없음",
                             "%.0f%%" % max(0.0, 100 * (1 - sum(shares) / w))))
    return table(["회로", "cutoff", "구간 경계의 항 수", "GPU", "전체", "gate 적용", "노드 사이 교환", "GPU↔host 복사", "분할", "GPU 대기", "다른 노드 대기", "그 밖"], rows,
                 ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])


def b_base_parts(g=16):
    rows = []
    for c in MAIN:
        for x in ("1e-4", "3e-5"):
            for tag, name in (("bd", "SectorQ"), ("bdc", "cuPauliProp 기본"), ("bdd", "cuPauliProp dedup 미룸")):
                item = f"{tag}.{c}.{x}.g{g}"
                s = state.get(item)
                if not ok(s):
                    rows.append((KO[c], x, name, cell(item), "", "", "", "", ""))
                    continue
                t = log_text(s)
                w = s["wall"]
                busy = cg.gpu_busy(item, t)
                b = "" if busy is None else "%.0f%%" % busy
                if tag == "bd":
                    ops, exch = num(r"operations=([\d.]+)", t), num(r"GB sent, ([\d.]+) s per node", t)
                    sent = num(r"between nodes: ([\d.]+) GB sent", t)
                    rows.append((KO[c], x, name, sec(w), b, "%.0f%%" % (100 * ops / g / w), "%.0f%%" % (100 * (exch or 0) / w), "없음", "" if sent is None else "%.1f GB (전체)" % sent))
                else:
                    mpi, coll = num(r"cupp mpi, seconds per rank: .*total=([\d.]+)", t), num(r"collective=([\d.]+)", t)
                    per = num(r"cupp mpi allgather: calls/rank=\d+ \(([\d.]+) per operation\)", t)
                    gb = num(r"cupp mpi isend: .*GB/rank=([\d.]+)", t)
                    rows.append((KO[c], x, name, sec(w), b, "", "" if mpi is None else "%.0f%% (collective %.0f%%)" % (100 * mpi / w, 100 * (coll or 0) / w), "" if per is None else "%.1f회" % per,
                                 "" if gb is None else "%.1f GB (rank당)" % gb))
    return table(["회로", "cutoff", "방식", "전체", "GPU 사용률 (nvidia-smi 표본)", "gate 적용", "통신 (MPI 호출 안의 시간, SectorQ는 노드 사이 교환)", "연산당 allgather (rank당)", "보낸 양"], rows,
                 ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:"])


def b_params():
    rows = []
    for c in MAIN:
        for g in (4, 16):
            base = f"eng.{c}.1e-5.g{g}"
            tg = sorted((int(i.rsplit(".t", 1)[1]), i) for i in ids if i.startswith(f"tgt.{c}.1e-5.g{g}.t"))
            pg = sorted((int(i.rsplit(".p", 1)[1]), i) for i in ids if i.startswith(f"pg.{c}.1e-5.g{g}.p"))
            rows.append((KO[c], g, cell(base), " / ".join(f"{t / 1e6:g}M: {cell(i)}" for t, i in tg), " / ".join(f"{k}: {cell(i)}" for k, i in pg)))
    return table(["회로 (cutoff 1e-5)", "GPU", "기본 설정", "sector당 목표 항 수 (기본의 1/4, 1/2, 2배)", "GPU당 sector 수 (기본 2)"], rows, ["---", "---:", "---:", "---", "---"])


def b_vram():
    rows = []
    for c in MAIN:
        for g in (1, 4):
            rows.append([KO[c], g, cell(f"eng.{c}.1e-5.g{g}")] + [cell(f"vram.{c}.1e-5.g{g}.f{f}") for f in ("0.45", "0.2", "0.1")])
    return table(["회로 (cutoff 1e-5)", "GPU", "36 GB (기본)", "18 GB", "8 GB", "4 GB"], rows, ["---", "---:", "---:", "---:", "---:", "---:"])


def b_exact():
    rows = []
    for steps in (4, 6):
        for x in ladder(f"exact.tfim30.s{steps}", "g4"):
            s = state.get(f"exact.tfim30.s{steps}.{x}.g4")
            f = cg.engine_facts("", s) or (None, None, None)
            err = "" if not ok(s) or s.get("expectation") is None else "%.5f" % abs(s["expectation"] - cg.EXACT[steps])
            rows.append((f"{steps} step", "%.6f" % cg.EXACT[steps], x, value(s) if ok(s) else cell(f"exact.tfim30.s{steps}.{x}.g4"), err, ko_terms(f[0]), ko_terms(f[1]), cell(s) if ok(s) else ""))
    return table(["깊이", "exact", "cutoff", "SectorQ의 값 (4 GPU)", "오차", "구간 경계의 항 수", "구간 안 최대 항 수", "시간"], rows, ["---", "---", "---", "---", "---:", "---:", "---:", "---:"])


def b_ckpt():
    rows = []
    for g in (4, 16):
        whole, r = state.get(f"eng.ki.1e-5.g{g}"), state.get(f"ckptr.ki.1e-5.g{g}")
        m = re.search(r"^resumed at operation (\d+) with (\d+) terms", log_text(r), re.M)
        same = ok(whole) and ok(r) and abs(whole["expectation"] - r["expectation"]) < 1e-10
        rows.append((g, cell(whole), ("연산 %s, %s 항" % (m.group(1), ko_terms(int(m.group(2))))) if m else "", cell(r), value(whole), value(r), "같음" if same else ("다름" if ok(r) else "")))
    return table(["GPU", "끊지 않은 실행", "checkpoint 지점", "재개한 뒤의 시간", "끊지 않은 실행의 기대값", "재개한 실행의 기대값", ""], rows, ["---:", "---:", "---", "---:", "---", "---", "---"])


def b_conv():
    conv = sorted({i.split(".")[2] for i in ids if i.startswith("conv.eng.")}, key=lambda t: float(t[1:]))
    if not conv or not any(i.startswith("conv.") and i in state for i in ids):
        return "수렴 실험(224건)은 큐의 맨 뒤에 있으며 아직 시작하지 않았다."
    rungs = sorted({i.split(".")[3] for i in ids if i.startswith("conv.eng.")}, key=float, reverse=True)
    rows = []
    for t in conv:
        row = [t[1:] + "π"]
        for x in rungs:
            s = state.get(f"conv.eng.{t}.{x}.g16") or state.get(f"conv.eng.{t}.{x}.g1")
            row.append(("%+.5f" % s["expectation"]).replace("-", "−") if ok(s) and s.get("expectation") is not None else (cell(s) if s else "측정 중"))
        rows.append(row)
    return table(["rotation 각도"] + rungs, rows)


def b_single():
    """GPU 1장: the largest setting cuPauliProp finishes on one GPU, and how far SectorQ goes on one GPU."""
    rows = []
    for c in ALL:
        x = largest_both(c, 1, 1)
        if not x:
            continue
        a, b = state.get(f"eng.{c}.{x}.g1"), state.get(f"cupp.{c}.{x}.g1")
        done = [y for y in ladder(f"eng.{c}", "g1", True) if ok(state.get(f"eng.{c}.{y}.g1"))]
        far = min(done, key=float)
        rows.append((KO[c], x, ko_terms(facts(c, x)[0]), cell(b), sec(a["wall"]) + " (%s)" % mult(a["wall"] / b["wall"]), cell(f"eng.{c}.{x}.g16"),
                     f"{far} ({ko_terms(facts(c, far)[0])} 항, {sec(state[f'eng.{c}.{far}.g1']['wall'])})"))
    return table(["회로", "cuPauliProp이 GPU 1장으로 끝내는 가장 작은 cutoff", "구간 경계의 항 수", "cuPauliProp 단일 GPU", "SectorQ, GPU 1장 (cuPauliProp 대비)", "SectorQ, 16 GPU",
                  "SectorQ가 GPU 1장으로 끝내는 가장 작은 cutoff"], rows, ["---", "---", "---:", "---:", "---:", "---:", "---"])


def b_gain():
    """How much 16 GPUs gain over one GPU, by problem size."""
    rows = []
    for c in MAIN:
        for x in ("1e-3", "1e-4", "3e-5", "1e-5", "3e-6"):
            a, b = state.get(f"eng.{c}.{x}.g1"), state.get(f"eng.{c}.{x}.g16")
            f = facts(c, x)
            rows.append((KO[c], x, ko_terms(f[0]), ko_terms(f[1]), cell(f"eng.{c}.{x}.g1"), cell(f"eng.{c}.{x}.g16"), mult(a["wall"] / b["wall"]) if ok(a) and ok(b) else ""))
    return table(["회로", "cutoff", "구간 경계의 항 수", "구간 안 최대 항 수", "GPU 1장", "16 GPU", "1장 대비"], rows, ["---", "---", "---:", "---:", "---:", "---:", "---:"])


# 32-256 GPUs: the scaling jobs (jobs/e15_scaling.sh), one log per GPU count
BIG = (32, 64, 128, 256)
BIG_WEAK = {32: "3.4e-6", 64: "2.3e-6", 128: "1.6e-6", 256: "1.15e-6"}
CIRC = {"ki": "ki_lab", "tfim": "tfim_hh", "noise": "ki_noisy"}
_big = {}


def big_runs():
    if not _big:
        for path in sorted(glob.glob(os.path.join(ROOT, "logs", "pp_e15_g*-*.out"))):
            for r in mf.parse(os.path.basename(path)):
                _big.setdefault(r["gpus"], []).append(r)
    return _big


def big_run(kind, c, coef, g, tuned=False, gmax=None):
    for r in big_runs().get(g, []):
        if (r["kind"] == kind and r["circuit"] == CIRC[c] and abs(r["coef"] / float(coef) - 1) < 1e-9 and r["tuned"] == tuned
                and r["opt"].get("--gmax") == gmax and "--chunk" not in r["opt"]):
            return r
    return None


def big_cell(r, g, against=None):
    if g not in big_runs():
        return "대기 중"
    if r is None:
        return ""
    if not mf.ok(r):
        return "메모리 부족" if r["fatal"] else "시간 초과"
    return sec(r["wall"]) + (f" ({mult(against / r['wall'] if against > r['wall'] else r['wall'] / against)})" if against else "")


def b_large():
    """32-256 GPUs: strong and weak scaling of SectorQ and the comparison with cuPauliProp's distributed mode."""
    small = (1, 4, 16)
    out = []
    # strong scaling
    rows = []
    for c, x in (("ki", "3e-6"), ("tfim", "5e-6"), ("noise", "5e-6"), ("ki", "1e-5"), ("tfim", "1e-5"), ("noise", "1e-5")):
        one = state.get(f"eng.{c}.{x}.g1")
        t1 = one["wall"] if ok(one) else None
        row = [f"{KO[c]}, cutoff {x} ({ko_terms(facts(c, x)[0])} 항)"]
        for g in small:
            s = state.get(f"eng.{c}.{x}.g{g}")
            row.append(cell(f"eng.{c}.{x}.g{g}") + (f" ({mult(t1 / s['wall'])})" if t1 and ok(s) and g > 1 else ""))
        for g in BIG:
            r = big_run("engine", c, x, g)
            row.append(big_cell(r, g, t1 if (r and mf.ok(r)) else None))
        rows.append(row)
    out.append("**Strong scaling.** 같은 문제를 GPU 수만 바꾸어 실행한 SectorQ의 시간이다. 괄호는 1 GPU 대비이다.\n\n"
               + table(["문제"] + [f"{g} GPU" for g in small + BIG], rows))
    # weak scaling, the large load
    rows = [["cutoff"] + [cg.WEAK["L"][g] for g in small] + [BIG_WEAK[g] for g in BIG]]
    for c in MAIN:
        base, row = None, [KO[c]]
        for g in small:
            s, x = weak_run("L", c, g)
            m = re.search(r"^integrity: (\d+) term checks", log_text(s), re.M) if ok(s) else None
            rate = int(m.group(1)) / s["wall"] / g if m else None
            base = rate if g == 1 else base
            row.append(f"{sec(s['wall'])} ({100 * rate / base:.0f}%)" if rate and base else cell(s))
        for g in BIG:
            r = big_run("engine", c, BIG_WEAK[g], g)
            if g in big_runs() and r and mf.ok(r) and r.get("checks") and base:
                row.append(f"{sec(r['wall'])} ({100 * r['checks'] / r['wall'] / g / base:.0f}%)")
            else:
                row.append(big_cell(r, g))
        rows.append(row)
    out.append("**Weak scaling (큰 부하).** GPU 수에 맞추어 cutoff를 낮춘 SectorQ의 실행 시간이다. 괄호는 GPU 한 장이 1초에 처리한 항 수를 1 GPU와 비교한 것이다.\n\n"
               + table(["회로"] + [f"{g} GPU" for g in small + BIG], rows))
    # cuPauliProp on the same nodes, cutoff 1e-5
    rows = []
    for c in MAIN:
        for g in (16,) + BIG:
            if g == 16:
                e = state.get(f"eng.{c}.1e-5.g16")
                rows.append([KO[c], g, cell(e), deferred_cell(c, "1e-5", 16, e), cell(f"cupp.{c}.1e-5.g16", e)])
            else:
                e = big_run("engine", c, "1e-5", g)
                t = e["wall"] if (e and mf.ok(e)) else None
                rows.append(["", g, big_cell(e, g), big_cell(big_run("cupp", c, "1e-5", g, tuned=True), g, t), big_cell(big_run("cupp", c, "1e-5", g), g, t)])
    out.append("**cuPauliProp 분산 모드와의 비교 (cutoff 1e-5, 같은 노드).** 괄호는 SectorQ 대비이다.\n\n"
               + table(["회로", "GPU 수", "SectorQ", "cuPauliProp dedup 미룸", "cuPauliProp 기본"], rows, ["---", "---:", "---:", "---:", "---:"]))
    return "\n\n".join(out)


BLOCKS = {"large": b_large, "single": b_single, "gain": b_gain, "progress": b_progress, "agree": b_agree, "speed": b_speed, "single_node": b_single_node, "node_mult": b_node_mult,
          "multi_mult": b_multi_mult, "strong_sota": b_strong_sota, "same_error": b_same_error, "given": b_given,
          "threads": b_threads, "gpus": b_gpus, "reach": b_reach, "largest": b_largest, "strong": b_strong, "weak": b_weak, "parts": b_parts,
          "base_parts": b_base_parts, "params": b_params, "vram": b_vram, "exact": b_exact, "ckpt": b_ckpt, "conv": b_conv}


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("--")] or list(BLOCKS)
    made = {name: BLOCKS[name]() for name in want}
    if "--write" not in sys.argv:
        for name, text in made.items():
            print(f"<!-- g:{name} -->\n{text}\n<!-- /g:{name} -->\n")
        return
    path = os.path.join(ROOT, "README.md")
    doc = open(path).read()
    for name, text in made.items():
        pat = re.compile(r"(<!-- g:%s -->\n)(?:.*?\n)?(<!-- /g:%s -->)" % (name, name), re.S)
        if not pat.search(doc):
            print("no block for", name)
            continue
        doc = pat.sub(lambda m: m.group(1) + text + "\n" + m.group(2), doc)
    open(path, "w").write(doc)
    print("README.md: blocks rewritten:", ", ".join(n for n in made))


if __name__ == "__main__":
    main()
