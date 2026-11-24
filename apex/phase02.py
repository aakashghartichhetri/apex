"""Phase 02: deterministic heterogeneous-device scheduling laboratory."""
import argparse
import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Device:
    name: str
    memory: float
    speed: float

@dataclass(frozen=True)
class Model:
    name: str
    memory: float
    seconds: float
    quality: float

DEVICES = (Device("cpu", 16, 1), Device("accelerator", 8, 4))
MODELS = (Model("small", 2, .4, .72), Model("large", 6, 1.4, .9))
POLICIES = ("fastest", "quality", "queue", "apex")
def candidates(devices=DEVICES, models=MODELS):
    return [(d,m) for d in devices for m in models if m.memory <= d.memory]

def choose(policy, options, free, arrival, difficulty, deadline):
    def score(pair):
        d,m = pair
        service = m.seconds * (1+difficulty) / d.speed
        latency = max(0,free[d.name]-arrival)+service
        quality = max(0,m.quality-.2*difficulty)
        if policy == "fastest": return -service
        if policy == "quality": return quality
        if policy == "queue": return -latency
        if policy == "apex": return quality-.08*service-.5*max(0,latency-deadline)
        raise ValueError(policy)
    return max(options,key=score)

def run(policy, seed=1, n=1000, regime="burst", deadline=1.0):
    if n < 1 or deadline <= 0: raise ValueError("positive n and deadline required")
    rng = random.Random(seed)
    free = {d.name:0.0 for d in DEVICES}
    busy = dict(free)
    arrival=0.0
    latencies=[]
    correct=misses=0
    options=candidates()
    for i in range(n):
        rate=8 if regime=="burst" and (i//100)%2 else 2
        arrival += rng.expovariate(rate)
        difficulty, outcome = rng.random(),rng.random()
        d,m=choose(policy,options,free,arrival,difficulty,deadline)
        service=m.seconds*(1+difficulty)/d.speed
        free[d.name]=max(arrival,free[d.name])+service
        busy[d.name]+=service
        latency=free[d.name]-arrival
        latencies.append(latency)
        correct += outcome < max(0,m.quality-.2*difficulty)
        misses += latency > deadline
    horizon=max(free.values())
    return dict(policy=policy,seed=seed,regime=regime,n=n,accuracy=correct/n,
                p95=sorted(latencies)[(95*n+99)//100-1],miss_rate=misses/n,
                cpu_util=busy["cpu"]/horizon,accelerator_util=busy["accelerator"]/horizon,
                throughput=n/horizon)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=1000)
    p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--out",default="results/phase02.csv")
    a=p.parse_args()
    if a.seeds<1: p.error("seeds must be positive")
    rows=[run(policy,s,a.n,r) for r in ("steady","burst") for s in range(a.seeds) for policy in POLICIES]
    path=Path(a.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in ("steady","burst"):
        for policy in POLICIES:
            group=[x for x in rows if x["regime"]==r and x["policy"]==policy]
            print(r,policy,*(f"{k}={statistics.mean(x[k] for x in group):.4f}" for k in ("accuracy","p95","miss_rate","throughput")))
if __name__=="__main__": main()
