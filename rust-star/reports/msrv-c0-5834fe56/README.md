# Current Rust 1.74 full-model correctness and profile

The current Rust 1.74 executable passes two fresh 32K/128 full-output C0 runs: normal and profiled. Every one of the 33,354,240 checked FP32 logit values matches the immutable oracle fixture bit for bit. Runtime source remains unchanged from the compatibility repair; this closes its full-model correctness gap at 32K.

The executable is separately pinned from the older Rust 1.91 performance baseline. This is not a compiler speed comparison, higher-frontier validation or DwarfStar benchmark.

## Evidence

Both fresh processes exit successfully, reproduce the entire expected report and check all 128 decode steps. The profiled run also passes unchanged coverage, order, positivity and attribution checks for all 4,466 component intervals. All 42 source files, 7,669 tracked fixture files, the complete model, prompt, oracle source/binaries and current executable are pinned before and after execution. No failed run or excluded sample occurred.

The model is DeepSeek-V4-Flash IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8 chat-v2 imatrix-0731. Oracle source is ccbd16d36895ef9fe8edf7f793014988ff1ab7f4. The compatibility build and source-inverse provenance remain in the adjacent msrv-174-179e0c24 report. Model, fixtures and executables are omitted from this portable report; their identities remain in the manifests.

## Refreshed diagnostic profile

Counter instrumentation perturbs execution. These values locate work inside this one profiled run and do not establish a paired throughput improvement.

| Stage | GPU ms | Recorded-stage share |
|---|---:|---:|
| tail | 75867.613 | 37.39% |
| attention | 57880.545 | 28.52% |
| bootstrap_ingress | 32939.910 | 16.23% |
| prefix_layer | 17683.661 | 8.71% |
| ingress | 6995.473 | 3.45% |
| boundary | 6192.858 | 3.05% |
| initial_bootstrap | 4946.540 | 2.44% |
| compressor | 408.887 | 0.20% |
| bind | 0.000 | 0.00% |
| host_scratch | 0.000 | 0.00% |

The complete tail/attention component breakdown, per-packet coverage and raw counters are preserved in profile-audit.json and current-32k-profiled-components.json. Bootstrap ingress accounts for 16.23% of recorded GPU stage time. A later optimization needs its own prospective source, exactness, memory and whole-engine gates. The next-bootstrap-design.md document is conditional planning, not an accepted implementation.

## Reproduce the evidence audit

Run `python3 rust-star/reports/msrv-c0-5834fe56/verify-report.py`. This model-free Python audit verifies the artifact manifest, source/build/input pins, both complete-output reports and all exact decode confirmations, then reruns the frozen component audit on the preserved profile. Keep the adjacent msrv-174-179e0c24 report for the build provenance. Private paths are redacted with original and stored hashes.

Both routed-half attempts and the 4K bootstrap rejection remain closed. No runtime change, publication or automatic 256K resumption occurred. Remote CI, source-dependent higher frontiers and direct reference performance remain separate open requirements.
