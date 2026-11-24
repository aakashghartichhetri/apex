"""Offline trace replay: measured serial model durations, simulated queueing."""
import argparse,json,random,csv
from pathlib import Path
from .core import Calibration,Request,choose,POLICIES
import math

def feature(row):
    return min(1.,len(row['question'].split())/100)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True);p.add_argument('--out',default='results/replay.csv');p.add_argument('--rate',type=float,default=.1);p.add_argument('--deadline',type=float,default=30);a=p.parse_args()
    if a.rate<=0 or a.deadline<=0:p.error('rate and deadline must be positive')
    tr=json.loads(Path(a.train).read_text());te=json.loads(Path(a.test).read_text())
    if tr['split']!='train' or te['split']!='test':p.error('separate train and test splits required')
    train=tr['records'];test=te['records'];
    if not train or not test:p.error('empty trace')
    for label in ('small','large'):
        if len({r[label]['model'] for r in train+test})!=1:p.error('model mismatch')
    if tr['device']!=te['device']:p.error('device mismatch')
    cal=Calibration().fit(Request(0,feature(r),r['small']['correct'],r['large']['correct'],r['small']['seconds'],r['large']['seconds']) for r in train)
    durations=[sum(r[k]['seconds'] for r in train)/len(train) for k in ('small','large')]
    rng=random.Random(42); arrivals=[];t=0
    for r in test:t+=rng.expovariate(a.rate);arrivals.append(t)
    rows=[]
    for policy in POLICIES:
        free=[0.,0.];lat=[];cost=[];correct=[];routes=[];rng=random.Random(42)
        for r,t in zip(test,arrivals):
            waits=[max(0,f-t) for f in free];route=choose(policy,cal.predict(feature(r)),waits,durations,rng,deadline=a.deadline)
            actual=r[('small','large')[route]];elapsed=waits[route]+actual['seconds'];free[route]=t+elapsed
            lat.append(elapsed);cost.append(actual['seconds']);correct.append(actual['correct']);routes.append(route)
        n=len(test);rows.append(dict(policy=policy,n=n,accuracy=sum(correct)/n,compute_seconds=sum(cost)/n,p95_latency=sorted(lat)[math.ceil(.95*n)-1],deadline_miss=sum(x>a.deadline for x in lat)/n,large_fraction=sum(routes)/n))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
