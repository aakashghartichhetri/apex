"""Optional synchronized device microbenchmark, not an LLM benchmark."""
import argparse
import json
import statistics
import time
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu","cuda","mps"],default="cpu")
    p.add_argument("--size",type=int,default=1024)
    p.add_argument("--runs",type=int,default=20)
    a=p.parse_args()
    if not 1<=a.size<=4096 or not 1<=a.runs<=1000: p.error("size 1..4096, runs 1..1000")
    import torch
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA unavailable")
    if a.device=="mps" and not torch.backends.mps.is_available(): p.error("MPS unavailable")
    torch.manual_seed(42)
    x=torch.randn(a.size,a.size,device=a.device)
    y=torch.randn_like(x)
    def sync():
        if a.device=="cuda": torch.cuda.synchronize()
        elif a.device=="mps": torch.mps.synchronize()
    with torch.inference_mode():
        for _ in range(5): z=x@y
        sync()
        times=[]
        for _ in range(a.runs):
            sync(); start=time.perf_counter(); z=x@y; sync()
            times.append(time.perf_counter()-start)
    print(json.dumps(dict(device=a.device,torch=torch.__version__,size=a.size,
        dtype=str(x.dtype),runs=a.runs,median_seconds=statistics.median(times),
        samples_seconds=times,kind="matrix multiplication microbenchmark"),indent=2))
if __name__=="__main__": main()
