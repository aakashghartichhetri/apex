from dataclasses import dataclass
import math
import random

@dataclass(frozen=True)
class Request:
    arrival: float
    difficulty: float
    small_ok: bool
    large_ok: bool
    small_time: float
    large_time: float

class Calibration:
    """Laplace-smoothed empirical success estimates, fitted only on training data."""
    def __init__(self, bins=10):
        self.bins=bins
        self.counts=[[0,0,0] for _ in range(bins)]
    def index(self, d):
        return min(self.bins-1,max(0,int(d*self.bins)))
    def fit(self, requests):
        self.counts=[[0,0,0] for _ in range(self.bins)]
        for r in requests:
            c=self.counts[self.index(r.difficulty)]
            c[0]+=1; c[1]+=r.small_ok; c[2]+=r.large_ok
        return self
    def predict(self,d):
        n,s,l=self.counts[self.index(d)]
        return (s+1)/(n+2),(l+1)/(n+2)

def workload(n, seed, regime='steady'):
    rng=random.Random(seed); arrival=0.; result=[]
    for i in range(n):
        rate=3 if regime=='steady' else (10 if (i//100)%2 else 2)
        arrival+=rng.expovariate(rate)
        d=rng.betavariate(4,2) if regime=='shift' and i>=n//2 else rng.random()
        u=rng.random()
        result.append(Request(arrival,d,u<.98-.65*d,u<.995-.20*d,
                              .08+.06*d,.25+.35*d))
    return result

POLICIES=('small','large','random','quality','load','apex')
def choose(policy, estimates, waits, times, rng, threshold=.18, deadline=.8):
    s,l=estimates; ws,wl=waits; ts,tl=times
    if policy=='small': return 0
    if policy=='large': return 1
    if policy=='random': return rng.randrange(2)
    if policy=='quality': return int(l-s>threshold)
    if policy=='load': return int(wl+tl<ws+ts)
    if policy=='apex':
        # Expected quality benefit minus compute and deadline penalties.
        def utility(q,w,t):
            return q-.35*t-.8*max(0,w+t-deadline)
        return int(utility(l,wl,tl)>utility(s,ws,ts))
    raise ValueError(policy)

def evaluate(requests, calibration, policy, seed=0, deadline=.8, threshold=.18):
    rng=random.Random(seed); free=[0.,0.]; rows=[]
    for r in requests:
        waits=[max(0,t-r.arrival) for t in free]
        times=[r.small_time,r.large_time]
        estimates=calibration.predict(r.difficulty)
        # Only a duration estimate based on difficulty is visible to the policy.
        route=choose(policy,estimates,waits,times,rng,threshold,deadline)
        latency=waits[route]+times[route]
        free[route]=r.arrival+latency
        rows.append(dict(route=route,correct=int((r.small_ok,r.large_ok)[route]),
                         latency=latency,compute=times[route],confidence=estimates[route]))
    if not rows: raise ValueError('empty workload')
    n=len(rows); lat=sorted(r['latency'] for r in rows)
    mean=lambda key:sum(r[key] for r in rows)/n
    return dict(n=n,accuracy=mean('correct'),compute_seconds=mean('compute'),
                mean_latency=mean('latency'),p95_latency=lat[math.ceil(.95*n)-1],
                deadline_miss=sum(r['latency']>deadline for r in rows)/n,
                large_fraction=mean('route'),brier=sum((r['confidence']-r['correct'])**2 for r in rows)/n)
