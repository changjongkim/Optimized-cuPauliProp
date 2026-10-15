#!/usr/bin/env python3
"""Break down completed pp_engine_i scale logs; no new runs and no recursive scan."""
import csv
import json
from paper_ablation import ROOT, parse_log

def main():
    rows=[]
    for p in sorted((ROOT/'logs/s').glob('*.out')):
        if not any('.g'+str(g)+'.out' in p.name for g in (128,256,512)):
            continue
        t=p.read_text(errors='replace')
        if 'binary=pp_engine_i\n' not in t or 'mode=engine ' not in t:
            continue
        r=parse_log(p)
        if r['exit'] != 0 or not r['wall']:
            continue
        r['id']=p.stem
        # Worker counters are accumulated worker time, not GPU utilization samples.
        g=r['workers']
        for key in ('operations','partition','h2d','d2h','idle','row_wait'):
            r[key+'_avg']=r[key]/g if r[key] is not None else None
        r['residual']=r['wall']-sum(r[k+'_avg'] for k in ('operations','partition','h2d','d2h','idle'))-r['exchange']-r['sync']
        rows.append(r)
    (ROOT/'analysis/paper_large_time.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=[k for k in rows[0] if k!='trace']
    with (ROOT/'analysis/paper_large_time.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    for r in rows:
        if (r['id'].startswith(('i.ref.','n.eng.')) and '.3e-6.' in r['id']) or r['id']=='i12.ki.1.5e-7.g512':
            w=r['wall']
            print(r['id'], 'wall', w, 'operations%',round(100*r['operations_avg']/w,1), 'idle%',round(100*r['idle_avg']/w,1),'exchange%',round(100*r['exchange']/w,1), 'sync%',round(100*r['sync']/w,1), 'sectors',max(t['sectors'] for t in r['trace']), 'row wait sum',r['row_wait'])

if __name__=='__main__':
    main()
