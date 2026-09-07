# Rust Star: upstream refresh and oracle-v5 baseline

The upstream-refresh milestone is complete: pinned upstream, an explicitly versioned repeatable oracle, the applicable M1 selector port, complete Rust correctness/lifecycle gates, five fresh benchmark pairs and an isolated profile. This is a 32K development baseline, not the project’s 256K headline workload.

## Measured baseline

Five alternating pairs (AB, BA, AB, BA, AB), following one warmup per engine. Plain greedy generation is 128 steps; initial selection is charged to prefill and every decode interval includes evaluation, selection and commitment. Correctness dumping and profiling are disabled in the paired block.

| Metric | Rust median | Oracle-v5 median |
|---|---:|---:|
| Prefill | 423.152 s | 172.875 s |
| Steady decode | 20.484 tokens/s | 19.507 tokens/s |
| First decode step | 55.114 ms | 50.252 ms |
| Peak process memory | 92.300 GiB | 80.893 GiB |

Timing distributions (median, MAD, minimum, maximum) are retained in `results/paired-summary.json`, with memory distributions in `independent-audit.json`. The independent audit recomputes them from all raw measurements. Paired ratios below are computed directly from those same raw pairs.

- Prefill latency, paired Rust/reference ratio: median 2.43919, MAD 0.00855, range 2.41443–2.46094.
- Steady decode throughput, paired Rust/reference ratio: median 1.04952, MAD 0.00606, range 1.01479–1.06317.

Host monitoring found 0 unusual thermal observations. The machine was not otherwise isolated: the load observer records processes reaching 5% CPU every 45 seconds, including the separate profile period. CPU percentages below are process CPU, not GPU utilization; these observations do not establish a causal timing effect.

- `Browser Helper (Renderer)`: 98 observed samples; peak 88.9% CPU.
- `logioptionsplus_updater`: 91 observed samples; peak 35.1% CPU.
- `WindowServer`: 64 observed samples; peak 43.3% CPU.
- `logioptionsplus_agent`: 31 observed samples; peak 51.8% CPU.
- `backgroundtaskmanagementd`: 17 observed samples; peak 7.3% CPU.

## Change from the previous v4 campaign

- Rust median prefill: 521.130 → 423.152 s (-18.80% latency).
- Reference median prefill: 184.333 → 172.875 s (-6.22% latency).

Rust median peak process memory rose from 87.585 to 92.300 GiB (+4.715 GiB); the reference remains 80.893 GiB. This is a measured tradeoff of the refreshed baseline and should be tracked in the next optimization.

These are separate campaigns with different oracle versions. The previous v4 campaign included a disclosed host restart; this comparison is descriptive, not an isolated causal speedup claim. The prompt tokens and complete greedy transcript are unchanged. Historical reports and fixtures retain their original producer identities.

## Correctness and port scope

- Oracle-v5 source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, tree `a0a33f4a00c3c78cd40350785485203e7c6b0b1d`, based on upstream `c0a6119f363ef82125877142f13fb3fe491cba14`.
- Official Metal kernels and logprob-vector tests pass; four fresh full-vocabulary captures match at each 2K/32K. Final-source 8K also matches the historical fixture.
- Two fresh 32K/128 traces match all 129 full FP32 rows, including all 16,547,840 decode words. Every row is finite and global argmax equals EOS-excluding selection. Raw tokenization independently matches all 32,768 prompt IDs.
- A v4 control with only the streaming-selector branch removed reproduces the new 32K prefill tensor exactly. This isolates the prefill target change to selector behavior; no numerical tolerance was introduced.
- Against historical v4, 16,547,704 decode words change, while all 129 selected tokens remain identical. The complete v5 tensors are included under `correctness/fixtures/`.
- Rust uses upstream’s exact argsort kernel and block-sort/merge scheduling at both prefill selector sites. Existing batched tails and fused compressor score/APE work remain intact.
- Rust already uses layer-local inverse-RoPE fusion state and separate KV-store dispatches, avoiding the two mechanisms repaired upstream. Current M3/M5-only fusions do not constitute M1 performance ports. GLM/MXFP4, other accelerators, distributed and SSD work remain outside this fixed resident M1/Q2 runtime scope.
- The first pristine upstream kernel test failed because tests required optional M3/M5 fusions on M1. The pinned test repair checks refusal on M1, runs production fallbacks and preserves full output checks. Production admission is unchanged; the failed log is retained.
- The first diagnostic Rust run stopped on the old 16K layer 34 HC target. Direct oracle-v5 capture reproduced Rust exactly. Two fresh full 43-layer HC inventories at each 12K/16K/20K/24K establish separately versioned v5 internal targets; historical inventories and the stopped attempt are retained. Temporary HC payloads were compacted only after complete SHA256/FNV inventories.
- The detailed 8K diagnostic also exposed four reordered selector slots. Two direct v5 captures confirm identical score inputs and selected key sets, with only equal-score ordering changed. A separate v5 selector fixture replaces that index list; other historical tile tensors keep their original producer identities. A subsequent stale full layer 4 selector checksum was also replaced only after two all-even-layer captures reproduced it; its full tensor and stopped attempt are retained.
- The remaining-layer 8K diagnostic driver also needed updated merge dispatch accounting (37 per tile, 4736 per continuation, 4852 per even layer). The stopped metadata check is retained; inference and tensor targets did not change for that correction.
- Rust passes 310 tests and 93 Python tests, 12K internal control, fresh 32K output-only and diagnostic full-output gates, 2K/8K controls, and two retained 32K reset/reuse cycles. Internal HC targets remain accepted only through 24K; complete final logits remain mandatory.

The default benchmark builder now selects v5. Its final smoke test initially exposed omitted upstream image-decoder headers; the exporter now includes pinned `third_party/` sources and licenses. A fresh build passes, all 72 exported files match the pinned tree, and all 69 source files inventoried for the measured reference remain identical. The failed log, successful build proof, exporter and final 93-test Python run are retained. This packaging correction does not change either measured executable.

## Profile and next milestone

The isolated profile runs after the complete paired block and is explicitly ineligible for throughput claims. Coverage includes all 43 layers in every continuation chunk.

| Stage group | GPU seconds | Host seconds | Calls |
|---|---:|---:|---:|
| even/attention | 66.470 | 110.698 | 18816 |
| odd/ingress | 47.897 | 70.040 | 140 |
| even/tail | 42.677 | 55.369 | 147 |
| odd/tail | 40.061 | 52.169 | 140 |
| bootstrap/bootstrap_ingress | 31.620 | 33.455 | 7 |
| prefix/initial_bootstrap | 23.928 | 24.157 | 1 |
| odd/attention | 22.750 | 26.454 | 140 |
| even/boundary | 20.307 | 27.049 | 147 |
| prefix/prefix_layer | 17.294 | 24.462 | 40 |
| odd/compressor | 0.711 | 3.215 | 140 |
| even/bind | 0.000 | 2.267 | 147 |
| even/host_scratch | 0.000 | 0.009 | 147 |
| odd/bind | 0.000 | 0.472 | 140 |

The largest measured GPU stage group is `even/attention`. The next milestone should target that measured cost while preserving v5’s complete output contract and the retained-engine lifecycle. Extend broader context coverage separately; this refresh does not establish 256K throughput.

## Reproduce and audit

Rust source `cc032078748ad4c450717b1cfd4e734fd57cf9e6`; executable SHA `e87354b9f0ed3e2f30ac1b3584d7f03a255bbb715664dc452e29953b205bd706`.
Reference executable SHA `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`. Original frozen plan SHA `915d68eda1ebf4f619f3767a2003af7baa01a21d616612dc283e221ddf94e8cc`.

Run `python3 verify.py` from this directory. The verifier checks the artifact inventory, raw paired measurements, warmups, independent statistics, complete v5 tensors and repeated traces, selector attribution, correctness gates and profile coverage without loading the model. `inputs/plan-template.json` replaces private local paths with placeholders; `portability.json` records every derived file. Model, toolchain, configuration, source and fixture hashes are pinned in the evidence.
