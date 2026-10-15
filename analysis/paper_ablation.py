#!/usr/bin/env python3
"""Read only completed ablation logs at logs/x's top level (exclude archived runs)."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_log(path):
    text = path.read_text(errors='replace')
    def num(pattern, cast=float):
        match = re.search(pattern, text, re.M)
        return cast(match.group(1)) if match else None
    result = dict(log=str(path.relative_to(ROOT)))
    for key, pattern, cast in [
        ('exit', r'^exit=(\d+)\s*$', int), ('wall', r'^wall=([\d.]+)', float),
        ('expectation', r'^expectation=([^\s]+)', float), ('windows', r'^windows=(\d+)', int),
        ('terms', r'^max_terms_at_window_boundary=(\d+)', int),
        ('workers', r'summed over (\d+) GPU workers:', int),
        ('operations', r'operations=([\d.]+)', float), ('partition', r'partition=([\d.]+)', float),
        ('h2d', r'host->gpu=([\d.]+)', float), ('d2h', r'gpu->host=([\d.]+)', float),
        ('idle', r'idle=([\d.]+)', float), ('planning', r'planning=([\d.]+)', float),
        ('row_wait', r'workers waited ([\d.]+) s', float),
        ('sent_gb', r'between nodes: ([\d.]+) GB', float),
        ('exchange', r'between nodes: [\d.]+ GB sent, ([\d.]+) s', float),
        ('sync', r'waiting for the slowest node=([\d.]+)', float),
        ('counts', r'exchange per node on average: counts ([\d.]+)', float),
        ('posting', r'posting ([\d.]+) s', float), ('transfer', r'transfer ([\d.]+) s', float),
        ('new_chunks', r'of it new host chunks ([\d.]+) s', float),
        ('gpu_splits', r'splits_on_gpu=(\d+)', int), ('host_splits', r'splits_on_host=(\d+)', int),
        ('host_gib', r'GiB on the busiest node, ([\d.]+) GiB', float),
    ]:
        result[key] = num(pattern, cast)
    traces = []
    for line in text.splitlines():
        if line.startswith('trace window='):
            trace = {k: float(v) for k, v in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)(?=\s|$)', line)}
            if trace.get('terms_in', 0):
                trace['imbalance_nonempty'] = trace['largest_sector_in'] * trace['sectors'] / trace['terms_in']
                trace['imbalance_all'] = trace['largest_sector_in'] * (2 ** trace['rows']) / trace['terms_in']
            traces.append(trace)
    result['trace'] = traces
    return result

def main():
    rows = []
    for path in sorted((ROOT / 'logs/x').glob('*.out')):
        match = re.fullmatch(r'(re|fx)\.(\w+)\.g(4|16)\.([\de.-]+)\.out', path.name)
        if not match:
            continue
        mode, circuit, gpus, cutoff = match.groups()
        row = parse_log(path)
        # Preserve the source path and summary counters without duplicating all
        # per-window logs in the checked-in table. parse_log retains traces for
        # callers that analyze occupancy or candidate selection.
        row['trace_windows'] = len(row.pop('trace'))
        row.update(mode=mode, circuit=circuit, gpus=int(gpus), cutoff=cutoff)
        rows.append(row)
    out = ROOT / 'analysis/paper_fixed_rows.json'
    out.write_text(json.dumps(rows, indent=2) + '\n')
    fields = [k for k in rows[0] if k != 'trace']
    with (ROOT / 'analysis/paper_fixed_rows.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for g in (16, 4):
        for c in ('ki', 'tfim', 'tfimsq', 'xxz', 'qaoa', 'hea'):
            runs = {(r['mode'], r['cutoff']): r for r in rows if r['gpus'] == g and r['circuit'] == c}
            pairs = [(r, runs.get(('fx', x))) for (m, x), r in runs.items() if m == 're']
            pairs = [(a, b) for a, b in pairs if b and a['exit'] == b['exit'] == 0]
            ratios = [b['wall'] / a['wall'] for a, b in pairs]
            errors = [abs(a['expectation'] - b['expectation']) for a, b in pairs]
            print(g, c, 'pairs', len(pairs), 'speedup', round(min(ratios), 2), round(max(ratios), 2), 'max answer diff', max(errors))

if __name__ == '__main__':
    main()
