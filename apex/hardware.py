"""Report available PyTorch devices; no throughput claims."""
import json
import platform
def main():
    report={"platform":platform.platform(),"python":platform.python_version(),"cpu":platform.processor()}
    try:
        import torch
        report.update(torch=torch.__version__,cuda=torch.cuda.is_available(),
                      mps=bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()))
        report["cuda_devices"]=[{"name":torch.cuda.get_device_name(i),"memory_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    except ImportError:
        report["torch"]="not installed"
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
