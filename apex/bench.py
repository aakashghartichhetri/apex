import argparse,csv,json,platform
from pathlib import Path
from .core import Calibration,workload,evaluate,POLICIES

def main():
    p=argparse.ArgumentParser(); p.add_argument('--n',type=int,default=1000); p.add_argument('--seeds',type=int,default=5); p.add_argument('--out',default='results'); a=p.parse_args()
    if a.n<1 or a.seeds<1: p.error('n and seeds must be positive')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cal=Calibration().fit(workload(5000,90001)); rows=[]
    for regime in ('steady','burst','shift'):
        for seed in range(a.seeds):
            data=workload(a.n,seed,regime)
            for policy in POLICIES:
                rows.append(dict(regime=regime,seed=seed,policy=policy,**evaluate(data,cal,policy,seed)))
    with (out/'simulation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'manifest.json').write_text(json.dumps(dict(kind='synthetic simulation',n=a.n,seeds=list(range(a.seeds)),training_seed=90001,python=platform.python_version(),deadline=.8),indent=2)+'\n')
    for regime in ('steady','burst','shift'):
        print(regime)
        for policy in POLICIES:
            r=[x for x in rows if x['regime']==regime and x['policy']==policy]
            print(policy,*(f'{sum(x[k] for x in r)/len(r):.3f}' for k in ('accuracy','compute_seconds','p95_latency','deadline_miss')))
if __name__=='__main__':main()
