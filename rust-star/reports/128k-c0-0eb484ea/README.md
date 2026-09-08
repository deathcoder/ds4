# Native 128K prefill and decode correctness

**Accepted:** two fresh native runs match every prefill logit and every logit over 128 greedy decode steps at 131,072 prompt tokens. Two fresh unchanged oracle-v5 processes produce identical reference traces. Repeated oracle prefill at 64K/96K and native intermediate-frontier checks also pass.

All 314 Rust tests, the release build and seven established shorter/32K/lifecycle/profile controls pass. The profile audit validates 287 native tails and 2,583 component intervals. No larger-context throughput claim is made.

| Native check | Full prefill words | Full decode rows | Peak process RSS (GiB) | macOS peak footprint (GiB) |
|---|---:|---:|---:|---:|
| native-65536-a | 129,280 | 0 | 6.656 | 14.391 |
| native-98304-a | 129,280 | 0 | 10.209 | 17.853 |
| native-131072-a | 129,280 | 128 | 13.880 | 21.910 |
| native-131072-b | 129,280 | 128 | 13.878 | 21.891 |

Peak RSS comes from macOS `/usr/bin/time -l` for each fresh native process. It is observed process memory, not a full Metal allocation accounting or a universal capacity guarantee. These diagnostic commands do not call `warm_tensor_pages`, unlike the earlier warmed engine benchmark. Their RSS does not represent total model residency and must not be compared directly with that warmed 83.850GiB measurement. macOS peak-footprint counters are also retained separately; neither counter establishes a total Metal/model memory bound. Runs include correctness readbacks and are not paired benchmarks. The previous 93.173GiB planning estimate remains historical evidence, not an asserted bound.

| Oracle frontier | Fresh repeats | Output rows per repeat | Prefill selected token | Prefill SHA-256 |
|---|---:|---:|---:|---|
| 65,536 | 2 | 1 | 39 | `f785a567330b39edf7fd4f5f5c5cc0f2ebbb5b178d7b5bebe18dcca0852b243d` |
| 98,304 | 2 | 1 | 918 | `ad55f6707d0be338cb13296873e69818ca155169eb6a6efbe8502433f792c6a7` |
| 131,072 | 2 | 129 | 1110 | `a640e526b0f7f701424402c70358536df33e9df4cb3599585d7e314b4a37ea7f` |

At 128K each run checks 129,280 prefill FP32 words and 16,547,840 decode FP32 words, plus every greedy selection. The 64K and 96K checks validate full prefill outputs only. New internal HC targets are not established; their accepted scope still ends at 24K.

## Implementation and scope

Frozen candidate `0eb484ea14e7ae5f8d34f32cb372bcd245ecf7b8` (executable SHA-256 `85081aa01de6d305db92557fb94335694e3bdbdf2dbbe4e8e1ca5e9f038205fc`) builds on the accepted memory-efficient prefill baseline. The change adds validated external vectors, extends native boundary/output/adoption guards through 128K, and parameterizes decode position/capacity while preserving kernel arithmetic and 4K scratch storage. See `source-audit.md` and `implementation.patch`.

`long-prefill-reference-probe MODEL TOKENS SELECTED LOGITS REPORT` compares explicit external vectors. The input checker rejects malformed lengths, out-of-range tokens, non-finite values and inconsistent greedy selections, and requires all 129 rows for 128K. Prompts must extend the pinned 32K corpus. Its JSON explicitly says oracle provenance is not authenticated by the runtime; this report authenticates the vectors against the pinned captures. Existing `engine-measure`, retained measurement and built-in fixture commands remain at their previously validated frontiers.

The next milestone is to admit 128K into the measurement/retained-engine protocol and benchmark it against the same pinned oracle with explicit memory and performance criteria. This report establishes correctness through 128K, not 256K, arbitrary-input correctness, serving readiness or coding-agent corpus performance.

## Provenance and verification

Oracle source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4` and executable `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0` are unchanged. Model SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` and prompt SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` are pinned. CLI tokenization extends the immutable 32K prefix. Every captured row is finite, its recorded argmax is independently checked, and no row selects EOS, so the benchmark EOS policy does not alter these transcripts.

All six oracle attempts and eleven native/regression checks are retained, with no failed attempts or retries in this accepted sequence. Raw reference vectors from both repetitions, capture metadata/logs, native output checks/logs, source snapshots, build/test logs and host observations are included. Bulk JSON logit dumps remain in the local work directory; equivalent exact FP32 vectors and per-row hashes are retained here. The model and executables are excluded.

Run `python3 verify.py` in this report directory. The portable verifier checks the full artifact inventory, repeated oracle vectors, all native checksums/selections, complete chunk progression, old 32K C0 anchors, component intervals and source provenance. No model or Metal access is needed. Host observations began during the early capture sequence; they are not a complete monitor of its initial setup. The machine was shared with interactive applications.

Host monitor retained 222 samples, 2026-09-08T11:52:28.254249+00:00 through 2026-09-08T14:38:29.436789+00:00. All thermal/performance queries report no recorded warnings: True. Observed swap use spans 27.06–35.94MiB. These observations do not establish an idle or controlled benchmark host.

Journal and project/runtime documentation are updated. Stop at this milestone.
