#!/usr/bin/env python3
"""Figure 17 profiles at larger workloads on larger GPU allocations.

The cutoff plan is fixed before the missing measurements finish. Selection uses
problem size, never the measured idle share. Missing or failed logs have no time
or term-count values; completed profiles retain their exact source/configuration.
The separate paper_time_sweeps dataset remains the controlled default-cap sweep.
"""
import csv
import json
import re
import shlex

from paper_ablation import ROOT, parse_log
from paper_time_sweeps import PARTS, read_sweeps

CIRCUITS = ('ki', 'tfim', 'noise')
CUTOFFS = {16: ('1e-4', '1e-5', '1e-6'),
           64: ('1e-4', '5e-6', '5e-7'),
           256: ('1e-4', '2e-6', '3e-7')}
CAPACITY_LOGS = {
    (64, 'ki'): 'n10.ki.5e-7.g64',
    (64, 'tfim'): 'i.runtime10.tfim.5e-7.g64',
    (64, 'noise'): 'i.runtime10.noise.5e-7.g64',
    (256, 'ki'): 'r12.ki.3e-7.g256',
    (256, 'tfim'): 'i.runtime12.tfim.3e-7.g256',
    (256, 'noise'): 'i.runtime12.noise.3e-7.g256',
}


def capacity_profile(g, circuit, cutoff):
    cap, chunk = (1024, 262144) if g == 64 else (4096, 65536)
    item_id = CAPACITY_LOGS[g, circuit]
    path = ROOT / 'logs/s' / (item_id + '.out')
    row = dict(circuit=circuit, gpus=g, cutoff=cutoff, id=item_id,
               log=str(path.relative_to(ROOT)), binary='pp_engine_i',
               sector_cap=cap, chunk_terms=chunk, target_terms=4000000 if circuit == 'tfim' else 6000000,
               trunc_every=10, status='pending', wall=None, terms=None, expectation=None,
               **{k: None for k in PARTS})
    if not path.exists():
        return row
    r = parse_log(path)
    if r['exit'] != 0 or not r['wall']:
        # A failed measurement must not appear as either a completed bar or pending.
        row['status'] = 'incomplete' if r['exit'] is None else 'failed'
        return row
    text = path.read_text(errors='replace')
    args = shlex.split(text.splitlines()[0].split(' args: ', 1)[1])
    options = dict(zip(args[::2], args[1::2]))
    assert re.search(r'^binary=pp_engine_i$', text, re.M), path
    assert r['workers'] == g, path
    assert float(options['--coef']) == float(cutoff), path
    assert options['--circuit'] == {'ki': 'ki_lab', 'tfim': 'tfim_hh', 'noise': 'ki_noisy'}[circuit], path
    assert 2 ** int(options['--gmax']) == cap, path
    assert int(options.get('--chunk', 262144)) == chunk, path
    assert int(options['--target']) == row['target_terms'], path
    assert 'window=ann host=pinned chunks' in text and 'truncEvery=10 ' in text, path
    for flag in ('--window', '--fixed-rows', '--cands', '--trunc-every'):
        assert flag not in options, (path, flag)
    for key in ('operations', 'partition', 'h2d', 'd2h', 'idle', 'exchange', 'sync'):
        assert r[key] is not None and r[key] >= 0, (path, key)
    row.update(status='complete', wall=r['wall'], terms=r['terms'], expectation=r['expectation'],
               operations=r['operations'] / g, exchange=r['exchange'],
               copies=(r['h2d'] + r['d2h']) / g, partition=r['partition'] / g,
               idle=r['idle'] / g, waiting=r['sync'])
    row['other'] = row['wall'] - sum(row[k] for k in PARTS if k != 'other')
    assert row['other'] >= 0 and row['terms'] > 0, path
    return row


def main():
    sweeps = {(r['gpus'], r['circuit'], r['cutoff']): r for r in read_sweeps()}
    rows = []
    for g, cutoffs in CUTOFFS.items():
        for circuit in CIRCUITS:
            for index, cutoff in enumerate(cutoffs):
                if g > 16 and index == 2:
                    row = capacity_profile(g, circuit, cutoff)
                else:
                    row = dict(sweeps[g, circuit, cutoff], chunk_terms=262144, status='complete')
                row['point'] = index
                row['terms_per_gpu'] = row['terms'] / g if row['terms'] is not None else None
                rows.append(row)
    assert len(rows) == 27
    (ROOT / 'analysis/paper_runtime_profiles.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (ROOT / 'analysis/paper_runtime_profiles.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['gpus'], row['circuit'], row['cutoff'], row['status'],
              'terms', row['terms'], 'cap', row['sector_cap'], 'chunk', row['chunk_terms'])
    print('complete:', sum(r['status'] == 'complete' for r in rows), '/', len(rows))


if __name__ == '__main__':
    main()
