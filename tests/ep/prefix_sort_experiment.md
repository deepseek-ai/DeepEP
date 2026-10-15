# Valid-prefix sorting and count-prefetch experiment

This benchmark separates two changes to deterministic compact BF16 dispatch: sorting only valid receive rows, and copying receive counts before independent computation. It adds no production option or default change.

Four factorial arms (`late_full`, `late_prefix`, `early_full`, `early_prefix`) retain the same capacity allocation and copy epilogue. Public `upstream_syncfree` and `upstream_defer` are additional references. Deferred also changes allocation shape and epilogue timing, so it is a practical reference rather than a one-factor contrast.

The prototype replaces the event's sort hook, copies same-dtype counts into pinned host memory, and waits before consuming them. Prefix views retain the full allocation. Scope is compact BF16 in one NVLink domain; no hybrid routing, expanded layout, FP8, convergence or training-speed claim. Here prefix means valid receive rows.

## Results

Eight H200 GPUs; Torch 2.13.0+cu132, CUDA 13.2, loaded NCCL 2.32.3. Production source `93eb6eb`, top-2, two local experts/rank, 8 communication SMs, 2048 tokens/rank, hidden 4096, seed 42. The extension was unchanged. Current base `b5a33eef` only removes format/lint machinery relative to the measured source.

Three fresh unprofiled processes each cover 8 cases × 6 arms × 12 balanced blocks, with 5 samples/block and 5 warmup calls/arm. Samples measure synchronized wall time from dispatch through the fixed GEMM island and receive readiness. Barriers and max-rank aggregation are outside the timer. Values below are milliseconds: the mean of 36 block medians, each over five maximum-rank samples. GEMMs=8 means eight independent 2048³ BF16 matrix multiplies, identical across arms.

| Capacity | Routes | GEMMs | Late full | Late prefix | Early full | Early prefix | Native syncfree | Native deferred |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | balanced | 0 | 0.6718 | 0.5342 | 0.6779 | 0.5437 | 0.6217 | 0.3518 |
| 1 | balanced | 8 | 0.8277 | 0.6997 | 0.8318 | 0.7065 | 0.7995 | 0.4574 |
| 1 | skewed | 0 | 0.9181 | 0.8967 | 0.9220 | 0.8753 | 0.7311 | 0.7102 |
| 1 | skewed | 8 | 0.9639 | 0.8897 | 0.9564 | 0.9067 | 0.8949 | 0.7164 |
| 4 | balanced | 0 | 1.1327 | 0.5336 | 1.1433 | 0.5300 | 0.9750 | 0.3419 |
| 4 | balanced | 8 | 1.3153 | 0.7097 | 1.3137 | 0.7122 | 1.1336 | 0.4570 |
| 4 | skewed | 0 | 1.3985 | 0.8753 | 1.4082 | 0.8726 | 1.1941 | 0.7126 |
| 4 | skewed | 8 | 1.4176 | 0.8842 | 1.4137 | 0.9005 | 1.3559 | 0.7209 |

Native deferred is faster than every prototype arm in these eight cases. Valid-prefix sorting helps relative to the matching full-capacity arm, especially at capacity factor 4; early count copies provide no consistent additional benefit. These results support an experimental benchmark rather than a new default. They include reaching host-visible exact counts, rather than timing the sort kernel alone.

All per-process case means and all six arms are in [prefix_sort_results.json](prefix_sort_results.json). Profiling was separate and does not supply these performance figures.

## Reproduce

From the repository root with the intended package/native extension on the import path:

```bash
export EP_DISABLE_GIN=1 EP_REUSE_NCCL_COMM=0 EP_BUFFER_DEBUG=0
export NCCL_NET_PLUGIN=none NCCL_DEBUG=WARN CUBLAS_WORKSPACE_CONFIG=:4096:8
EP_AVOID_RECORD_STREAM=0 python tests/ep/bench_prefix_sort.py --phase correctness --num-processes 8 --output /tmp/deepep-prefix-check
EP_AVOID_RECORD_STREAM=1 python tests/ep/bench_prefix_sort.py --phase correctness --num-processes 8 --output /tmp/deepep-prefix-check
EP_AVOID_RECORD_STREAM=1 python tests/ep/bench_prefix_sort.py --phase performance --num-processes 8 --tokens 2048 --hidden 4096 --blocks 12 --samples 5 --warmup 5 --output /tmp/deepep-prefix-results
```

Run the performance command three times in fresh processes and save each stdout separately. Correctness covers independent dispatch-reference equality, counts, combine, cached replay, and saved-view backward after later dispatches, with both allocation streams and uneven/balanced/duplicate/skewed/masked inputs. The retained correctness/smoke study has 1920 rank/cases; the full performance study adds 1152 preflight rank/cases.

Execution code matches the GPU-tested script; only the docstring and historical-source log label changed. `tested_source` denotes the original measurement; public-sort/harness hashes identify loaded code. Frozen script SHA256: `89bece2cd31a0b3000fd04b7b7f390e39a49a7ec44b9f1481cbccae3ab6378ad`.
