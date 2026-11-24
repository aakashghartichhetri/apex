# Phase 01 research protocol

## Hypothesis

At matched accuracy or matched compute, combining calibrated estimates of model correctness with queue-aware penalties reduces deadline violations relative to quality-only routing during bursts. The hypothesis is falsifiable; a negative result is useful. The current prototype does not establish novelty.

## Methods

For each candidate route a, select the largest score:

    U(a) = estimated P(correct | feature, a) - 0.35 * service_seconds
           - 0.8 * max(0, queue_wait + service_seconds - deadline)

These coefficients are illustrative, not tuned. Calibration uses ten bins and Laplace smoothing on separate training data. Synthetic experiments use 5,000 training requests (seed 90001), 1,000 test requests per seed (0–4), three workload regimes, and common workloads across policies. Deadline: 0.8 simulated seconds. The shift regime changes both load and feature distribution; separate them in follow-up ablations. The simulator knows service time from the synthetic difficulty feature; real trace replay uses training-only mean duration estimates. The same coefficients are not necessarily appropriate for real durations.

## Preliminary observations

See the CSV and summary for actual executed synthetic results. In burst traffic, APEX averages accuracy 0.791 and p95 latency 0.990 simulated seconds, versus quality-only accuracy 0.855 and p95 latency 21.325 seconds. This is a trade-off, not a matched-quality win. Load-only has lower latency but lower accuracy. Results are averages over five seeds, not confidence intervals. No real GPU, token throughput or financial cost claims are supported.

## Evaluation checklist

- Record exact model and dataset revisions, hardware, software versions and generation settings.
- Use disjoint train/validation/test sets; tune thresholds and penalty coefficients only on validation.
- Sweep policies to produce matched-accuracy and matched-cost comparisons; report per-seed variability and paired confidence intervals.
- Ablate calibration, load penalties and duration prediction separately; compare an established learned routing baseline.
- Evaluate parsing failures separately; add robust numeric normalization and check human-labeled examples.
- Report p50/p95/p99 latency, deadline misses, accuracy, token count, throughput and GPU memory only when measured.
- Distinguish serial trace replay from live concurrent serving; measure the latter before claiming deployment performance.

## Related work to read

- RouteLLM: https://arxiv.org/abs/2406.18665
- Scaling LLM Test-Time Compute Optimally: https://arxiv.org/abs/2408.03314
- PagedAttention: https://arxiv.org/abs/2309.06180
