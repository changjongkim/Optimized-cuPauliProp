#!/usr/bin/env python3
"""Completed default-configuration cutoff sweeps for the paper's time breakdown."""
import csv
import json
import re
import shlex

from paper_ablation import ROOT, parse_log

GPUS = (16, 64, 128, 256, 512)
CIRCUITS = ('ki', 'tfim', 'noise')
PARTS = ('operations', 'exchange', 'copies', 'partition', 'idle', 'waiting', 'other')


def read_sweeps():
    rows, seen = [], set()
    for g in GPUS:
        for circuit in CIRCUITS:
            prefixes = ('eng',) if g == 16 else (('i.ref', 'n.eng') if circuit == 'ki' else ('n.eng',))
            for prefix in prefixes:
                if g == 16:
                    # Preserve the existing 16-GPU breakdown's measured runs.
                    paths = [ROOT / ('logs/{}/eng.{}.{}.g16.out'.format(folder, circuit, cutoff))
                             for folder, cuts in (('g', ('1e-4', '3e-5', '1e-5')), ('r', ('3e-6', '1e-6')))
                             for cutoff in cuts]
                else:
                    pattern = '{}.{}.*.g{}.out'.format(prefix, circuit, g)
                    paths = sorted((ROOT / 'logs/s').glob(pattern))
                for path in paths:
                    match = re.fullmatch(re.escape(prefix + '.' + circuit) + r'\.([\de.-]+)\.g' + str(g) + r'\.out', path.name)
                    if not match:
                        continue
                    r = parse_log(path)
                    if r['exit'] != 0 or not r['wall']:
                        continue
                    text = path.read_text(errors='replace')
                    args = shlex.split(text.splitlines()[0].split(' args: ', 1)[1])
                    options = dict(zip(args[::2], args[1::2]))
                    binary = re.search(r'^binary=(\S+)', text, re.M).group(1)
                    assert binary in (('pp_engine_h', 'pp_engine_i') if g == 16 else ('pp_engine_i',)), path
                    assert 'gmax=8 window=ann host=pinned chunks' in text, path
                    assert 'truncEvery=10 ' in text, path
                    assert r['workers'] == g, path
                    # The extended KI capacity runs change the cap/chunk configuration.
                    # They remain in paper_large_time.json, outside this controlled sweep.
                    for flag in ('--gmax', '--chunk', '--window', '--fixed-rows', '--cands', '--trunc-every'):
                        assert flag not in options, (path, flag)
                    cutoff = match.group(1)
                    assert float(options['--coef']) == float(cutoff), path
                    key = (g, circuit, float(cutoff))
                    assert key not in seen, key
                    seen.add(key)
                    required = ('operations', 'partition', 'h2d', 'd2h', 'idle', 'exchange', 'sync')
                    assert all(r[k] is not None and r[k] >= 0 for k in required), path
                    parts = dict(operations=r['operations'] / g,
                                 exchange=r['exchange'],
                                 copies=(r['h2d'] + r['d2h']) / g,
                                 partition=r['partition'] / g,
                                 idle=r['idle'] / g,
                                 waiting=r['sync'])
                    parts['other'] = r['wall'] - sum(parts.values())
                    assert parts['other'] >= 0, (path, parts['other'])
                    row = dict(circuit=circuit, gpus=g, cutoff=cutoff, id=path.stem,
                               log=r['log'], binary=binary, wall=r['wall'], terms=r['terms'], expectation=r['expectation'],
                               sector_cap=256, target_terms=int(options['--target']), trunc_every=10)
                    row.update(parts)
                    rows.append(row)
    rows.sort(key=lambda r: (r['gpus'], CIRCUITS.index(r['circuit']), -float(r['cutoff'])))
    for circuit in CIRCUITS:
        cuts = [{r['cutoff'] for r in rows if r['circuit'] == circuit and r['gpus'] == g} for g in GPUS if g != 16]
        assert cuts[0] and all(c == cuts[0] for c in cuts), (circuit, cuts)
        assert len({r['target_terms'] for r in rows if r['circuit'] == circuit}) == 1, circuit
    return rows


def main():
    rows = read_sweeps()
    (ROOT / 'analysis/paper_time_sweeps.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (ROOT / 'analysis/paper_time_sweeps.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for g in GPUS:
        for c in CIRCUITS:
            group = [r for r in rows if r['gpus'] == g and r['circuit'] == c]
            print(g, c, len(group), 'cutoffs', ','.join(r['cutoff'] for r in group),
                  'idle endpoints %', round(100 * group[0]['idle'] / group[0]['wall'], 1),
                  round(100 * group[-1]['idle'] / group[-1]['wall'], 1))
    print('completed runs:', len(rows))


if __name__ == '__main__':
    main()
