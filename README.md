# APEX 

Phase 01 / v0.1.0 — experimental research prototype, not a production inference engine.

**Question:** Can a calibrated quality estimate and queue-aware routing improve the trade-off between answer accuracy, service time and deadline misses compared with fixed or single-signal routing?

APEX contains a deterministic two-server discrete-event simulator, six routing policies, empirical calibration on a separate training set, and an optional real-model trace collector and offline replay tool. It deliberately does not claim a novel algorithm, GPU acceleration or production performance.

## Quick start (Python 3.10+, no dependencies)

```sh
python3 -m unittest discover -s tests -v
python3 -m apex.bench --n 1000 --seeds 5 --out results
```

The output is a CSV (90 policy/seed/regime combinations), a manifest, and a console summary. Checked-in results are **synthetic**, not LLM measurements. See [research protocol](docs/RESEARCH.md) and [results](results/summary.txt).

## Architecture

```text
Separate calibration data --> empirical probability estimates
                                      |
Request stream --> features + queue state --> routing policy
                                      |
                             small / large server
                                      |
                  accuracy, service time, latency, misses
```

The simulator models two independent FIFO servers, deterministic service times conditional on a synthetic difficulty feature, and coupled Bernoulli correctness. APEX maximizes estimated correctness minus service-time and predicted deadline-overrun penalties. The policies are small-only, large-only, random, quality-only, load-only and combined APEX.

## Real-model experiment (optional; not yet executed)

Install the optional dependencies into a virtual environment. Choose two compatible causal language models that fit your hardware; model downloads may require several GB. Use the same model revisions, hardware and generation settings for both splits. Pin revisions before formal experiments. The sample commands use example Qwen models; compatibility and runtime are unverified here.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-live.txt
mkdir -p traces
python -m apex.live --small Qwen/Qwen2.5-0.5B-Instruct --large Qwen/Qwen2.5-1.5B-Instruct --split train --n 100 --out traces/train.json
python -m apex.live --small Qwen/Qwen2.5-0.5B-Instruct --large Qwen/Qwen2.5-1.5B-Instruct --split test --n 100 --out traces/test.json
python -m apex.replay --train traces/train.json --test traces/test.json --rate 0.1 --deadline 30 --out results/replay.csv
```

The collector uses greedy decoding, a plain prompt, strict numerical answer parsing, and synchronized GPU timing when CUDA is available. It records paired outcomes and generation time. Offline replay uses measured serial service times and simulated arrivals/queues; it is **not** live concurrent inference. The live replay feature is question length, a weak proxy for difficulty, not privileged test labels. The real-model pipeline has not been executed in this environment. Raw traces are ignored by Git; review model and dataset terms before publishing them.

## Next steps

1. Validate real-model collection and answer normalization; add model/dataset revision hashes and hardware manifest.
2. Fit and tune routing policies using training/validation only, with fixed held-out evaluation.
3. Compare calibration and latency trade-offs across workloads, include confidence intervals and an existing learned router.
4. Run live concurrent serving on real hardware and seek independent reproduction.

See [research protocol](docs/RESEARCH.md) for the week-one plan and limitations.
