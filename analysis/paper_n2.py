#!/usr/bin/env python3
"""Audit same-allocation candidate-selection and overlap experiments.

Default: read current logs only. --write saves a separate JSON for an allocation.
Incomplete/failed attempts remain visible and never enter performance ratios.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import re
import shlex

from paper_ablation import ROOT, parse_log

NAME = re.compile(r'(re|rb|ov[01])\.(\w+)\.g(4|16)\.([\de.-]+)(?:\.r([123]))?\.out')


def read_runs(allocation=None):
    rows = []
    for path in sorted((ROOT / 'logs/b').glob('*.out')):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        text = path.read_text(errors='replace')
        header = dict(re.findall(r'(\w+)=([^\s]+)', text.splitlines()[0]))
        if allocation is not None and header.get('allocation') != allocation:
            continue
        mode, circuit, gpus, cutoff, repetition = match.groups()
        row = parse_log(path)
        row.update(mode=mode, circuit=circuit, gpus=int(gpus), cutoff=cutoff,
                   repetition=int(repetition or 0), allocation=header.get('allocation'))
        row['status'] = 'incomplete' if row['exit'] is None else 'failed'
        if row['exit'] == 0:
            row['status'] = 'complete' if row['wall'] and row['expectation'] is not None else 'invalid'
        sha = re.search(r'^binary_sha256=(\S+)$', text, re.M)
        args = re.search(r'^args:(.*)$', text, re.M)
        row['binary_sha256'] = sha.group(1) if sha else None
        argv = shlex.split(args.group(1)) if args else []
        row['options'] = dict(zip(argv[::2], argv[1::2]))
        row['idle_per_worker'] = row['idle'] / row['gpus'] if row['idle'] is not None else None
        row['row_wait_per_worker'] = row['row_wait'] / row['gpus'] if row['row_wait'] is not None else None
        for basis in ('nonempty', 'all'):
            values = [t['imbalance_' + basis] for t in row['trace'] if 'imbalance_' + basis in t]
            row['max_imbalance_' + basis] = max(values) if values else None
        if row['status'] == 'complete':
            expected = {'--gpus': '4', '--coef': cutoff, '--seed': '7', '--trace': '1',
                        '--vram-frac': '0.9', '--cands': '1' if mode == 'rb' else '8',
                        '--overlap-plan': '0' if mode == 'ov0' else '1'}
            if (not sha or not row['allocation'] or row['workers'] != row['gpus']
                    or any(row['options'].get(k) != v for k, v in expected.items())
                    or not math.isfinite(row['expectation']) or not row['trace']):
                row['status'] = 'invalid'
        rows.append(row)
    return rows


def compare(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row['circuit'], row['gpus'], row['cutoff'], row['repetition'])
        groups[key][row['mode']] = row
    pairs = []
    for key, modes in sorted(groups.items()):
        circuit, gpus, cutoff, repetition = key
        control, variant, changed = ('ov1', 'ov0', '--overlap-plan') if repetition else ('re', 'rb', '--cands')
        a, b = modes.get(control), modes.get(variant)
        pair = dict(circuit=circuit, gpus=gpus, cutoff=cutoff, repetition=repetition,
                    control=control, variant=variant, status='incomplete', ratio=None)
        if a and b:
            if a['status'] == b['status'] == 'complete':
                opts_a = {k: v for k, v in a['options'].items() if k != changed}
                opts_b = {k: v for k, v in b['options'].items() if k != changed}
                difference = abs(a['expectation'] - b['expectation'])
                valid = (a['allocation'] == b['allocation']
                         and a['binary_sha256'] == b['binary_sha256']
                         and opts_a == opts_b and difference <= 1e-9)
                pair.update(status='complete' if valid else 'invalid',
                            expectation_difference=difference,
                            allocation=a['allocation'],
                            ratio=b['wall'] / a['wall'] if valid else None)
            elif 'invalid' in (a['status'], b['status']):
                pair['status'] = 'invalid'
            elif 'failed' in (a['status'], b['status']):
                pair['status'] = 'failed'
        pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.allocation is not None and not args.allocation.isdigit():
        parser.error('--allocation must be a numeric job ID')
    if args.write and args.allocation is None:
        parser.error('--write requires --allocation')
    rows = read_runs(args.allocation)
    pairs = compare(rows)
    counts = Counter(r['status'] for r in rows)
    print('Runs:', dict(counts), 'pairs:', dict(Counter(p['status'] for p in pairs)))
    for control in ('re', 'ov1'):
        ratios = [p['ratio'] for p in pairs if p['control'] == control and p['status'] == 'complete']
        print(control, 'completed pairs', len(ratios),
              'variant/control time range', (min(ratios), max(ratios)) if ratios else None)
    if args.write:
        path = ROOT / 'analysis' / ('paper_n2.' + args.allocation + '.json')
        path.write_text(json.dumps(dict(rows=rows, pairs=pairs), indent=2) + '\n')
        print('Saved', path.relative_to(ROOT))
    if counts['invalid'] or any(p['status'] == 'invalid' for p in pairs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
