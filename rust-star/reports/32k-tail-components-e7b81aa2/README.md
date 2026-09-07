# Native 32K tail component profile

Candidate `e7b81aa2e468f8fa8fd7b173d7c22fb3a51be46d` adds diagnostic component attribution to the accepted
native-tail implementation. It does not introduce an inference optimization or
claim a throughput improvement. Both the profiled and normal output-only runs
match all 129,280 prefill logits and all 16,547,840 decode words at 32K against
oracle-v5. All 312 Rust tests pass.

## Component evidence

The profile covers 287 native tail calls (147 even, 140 odd), each with nine
compute encoder intervals: 2,583 independently checked rows. Counter shares
are scaled to each completed tail command GPU interval. The figures below are
attributed estimates from a single perturbed profile, not standalone kernel
latencies or a paired benchmark. They partition the parent tail totals and
must not be added to them. Prefix/bootstrap work is outside this breakdown.

| Component | Attributed GPU seconds | Share of native tail |
| --- | ---: | ---: |
| attention_output | 36.173 | 36.75% |
| routed_gate_up | 34.285 | 34.83% |
| routed_down | 19.267 | 19.57% |
| shared_expert | 5.733 | 5.82% |
| final_hc | 1.286 | 1.31% |
| ffn_ingress | 0.796 | 0.81% |
| routed_map | 0.332 | 0.34% |
| router | 0.298 | 0.30% |
| routed_sum | 0.270 | 0.27% |

Normal execution retains its one compute encoder and 25 dispatches per native
tail. Diagnostic execution uses nine compute encoders, with no additional host
waits or arithmetic changes. Timestamp storage is reused per completed call,
consumed once, and released at the existing prefill/decode or reset boundary.

## Reproduce and verify

Run the frozen executable with:

```sh
rust-star long-prefill-chunk-probe MODEL.gguf --context 32768 --output-only --json profiled.json --profile-json components.json
rust-star long-prefill-chunk-probe MODEL.gguf --context 32768 --output-only --json normal.json
python3 verify.py
```

The audit validates full-output target checksums, exact coverage, ordered raw
timestamps and every component's attribution against its parent interval.
`identity.json`, `build.json` and `implementation.patch` bind the source and
executable. The executable remains in local ignored evidence; the prior five-pair
performance report remains [unchanged](../32k-tail-89e32745/README.md).

The first development build observed a partially updated CLI/API and failed
with an unresolved export. `tests.log` preserves that failure. The complete
source then passed `tests-final.log` and the release build; no failed binary
was used for live evidence. No rejected model run or retry is omitted.
