from pathlib import Path
import hashlib,json,statistics
w=Path(__file__).resolve().parent;root=w.parents[2];d=root/'rust-star/reports/32k-attention-1711856e'
read=lambda p:json.loads(p.read_text());a=read(w/'audit.json');p=read(w/'compressed-grid-probe/audit.json');n=read(w/'compressed-grid-probe/aa-control/audit.json');run=read(w/'validation-acceptance.json')['runs'][0]
components='\n'.join(f"| {r['component']} | {r['gpu_ms']/1000:.6f} | {100*r['attention_share']:.2f}% |" for r in a['ranked_attention_components'])
trials='\n'.join(f"| {s['chunk_start']} | {s['key_rows']} | {'yes' if s['padded'] else 'no'} | {s['statistics']['median_paired_ratio']:.6f} | {s['statistics']['mad_paired_ratio']:.6f} | {s['statistics']['faster_pairs']}/7 | FAIL |" for s in p['shapes'] if s['statistics'])
controls='\n'.join(f"| {s['chunk_start']} | {s['statistics']['ab_ratio_median']:.6f} | {s['statistics']['ba_ratio_median']:.6f} | {s['statistics']['first_vs_second_median']:.6f} |" for s in n['shapes'] if s['statistics'])
old=read(root/'rust-star/reports/32k-route-order-45520435/candidate-32k-profiled-components.json');old_odd=sum(r['gpu_ms'] for r in old['rows'] if r['stage']=='attention' and r['first_layer']%2);old_even=sum(r['gpu_ms'] for r in old['rows'] if r['stage']=='attention' and r['first_layer']%2==0)
attn=a['profiles']['attention'];coverage=attn['counter_coverage']
md=f'''# 32K attention breakdown and compressed-grid screen

Completed 2026-09-11 on Apple M1 Ultra, 48 GPU cores, 128 GB. Diagnostic runtime: `1711856ea309f70877102d15facb4e1243a9fdcc`, based on `ce94f3fb` / runtime `45520435`.

## Outcome

**Retain the diagnostic attention profiler. Do not promote the head-first kernel candidate.** The profiler passes all 16,677,120 full FP32 output comparisons at 32K/128 and the strict component audit. A standalone wrapper preserves all 4,697,620,480 FP32 comparisons but fails every predeclared speed screen. A separate A/A null control reproduces strong first/second order sensitivity. This establishes no kernel or engine speedup; it does not prove the candidate is intrinsically slower.

The new attribution identifies the actual attention computations as the largest components: compressed attention 26.158s, indexed attention 18.197s, indexer scoring 11.389s. Top-k selection contributes 1.601s. These are deliberately perturbed diagnostic attributions, not normal engine latency. The earlier routed gate/up candidate remains correctness-validated only. No larger frontier, engine pairing, push or packaged release is part of this step; 256K remains paused.

## Full-model diagnostic

Indexed attention now records nine components: indexer Q projection, Q transformation, weight projection, scores, top-k selection, chronological ordering, KV conversion, attention and inverse RoPE. Compressed attention records four: KV conversion, mask/block preparation, attention and inverse RoPE. Profiling is enabled after the initial prefix rebuild. Only the native continuation paths use these sample buffers; absent profiling, the original encoder sequence is preserved. Kernels, arithmetic, dispatch geometry and immutable fixtures are unchanged.

Each packet must come from a completed native attention call, match its expected sample count and be consumed before reuse. Every counter boundary must increase strictly and differ from the error sentinel. Invalid packets retain the raw timestamps in the error. Existing MTLFence ordering is used only between diagnostic encoders. New profiling objects follow the existing request-state release boundary.

The single declared model run passes 129,280 prefill FP32 words plus 128 full decode rows, totaling **16,677,120 exact words** against oracle-v5. Process elapsed is {run['elapsed_seconds']:.3f}s; diagnostic peak RSS is {run['peak_rss_bytes']/2**30:.3f}GiB. These values include conformance/startup work and omit normal engine residency warming; they are neither throughput nor full-memory claims. No internal-HC or larger-context contract is extended.

The audit checks all seven chunks from 4096 through 28672 and layers 2 through 42: **147 indexed packets plus 140 compressed packets, 1,883 attention intervals**. It also checks the existing **287 tail packets / 2,583 intervals**, initial bootstrap layers 0–2 and all 40 prefix layers. Component sums reconcile to their parent GPU intervals.

| Component | Attributed GPU seconds | Share of attention GPU |
| --- | ---: | ---: |
{components}

The new profile records {a['stages']['attention']['gpu_ms']/1000:.6f}s in attention and {sum(v['gpu_ms'] for v in a['stages'].values())/1000:.6f}s across all recorded stages. Parent rows and child attributions must not be added together.

**Instrumentation limits:** compared with the previous profile that split only tail components, compressed attention's parent changes from {old_odd/1000:.6f}s to {attn['path_totals_ms']['compressed']/1000:.6f}s ({100*(attn['path_totals_ms']['compressed']/old_odd-1):.2f}% higher); indexed attention changes from {old_even/1000:.6f}s to {attn['path_totals_ms']['indexed']/1000:.6f}s. These are separate processes, so this is an observed difference rather than an isolated estimate of profiler overhead. Sampled spans cover {100*coverage['minimum']:.2f}–{100*coverage['maximum']:.2f}% of each packet's first-to-last counter span, median {100*coverage['median']:.2f}%. Gaps are preserved. Scaling assigns parent time in proportion to sampled spans and is an attribution model, not a direct stopwatch for each component. This uncertainty precludes interpreting the table as production speed.

## Bounded scheduling experiment

The largest measured component motivated one scheduling hypothesis: visiting neighboring heads at a fixed query tile could improve query/KV/mask locality. The standalone wrapper forwards `uint3(tgpig.y, tgpig.x, tgpig.z)` to the unchanged original FlashAttention implementation while transposing the dispatch grid. All per-output arithmetic and threadgroup geometry remain unchanged: 4,096 queries, 64 heads, width 512, 256 threads and 28,672 bytes of shared memory. No optimized kernel was integrated into the runtime.

The probe uses deterministic finite synthetic F32 queries, shared F16 K/V, per-head sinks and production-shaped causal masks. It covers all seven 32K continuation chunk shapes, including both padded and unpadded KV. GPU mask blocks and padding are independently checked against CPU calculations; the report verifier independently recomputes mask-block coverage too. Full output words, finite/nondegenerate values, destination guards and unchanged input digests pass. Every warm-up pair and every timed pair is checked; no sample is discarded or retried.

There are two warm-ups per variant per shape and seven alternating pairs at three predeclared shapes. **All 4,697,620,480 compared FP32 words match.** This is isolated synthetic kernel exactness, not another full-model C0 result. The frozen screen requires median paired GPU ratio below 1.0 and at least 5/7 faster pairs at every timed shape.

| Chunk start | Key rows | KV padded | Median candidate/baseline | Ratio MAD | Faster pairs | Screen |
| --- | ---: | --- | ---: | ---: | ---: | --- |
{trials}

The candidate is slower in all nine BA pairs and faster in 11 of 12 AB pairs. The order sensitivity prevents interpreting favorable medians as a reliable improvement. All raw durations and orders remain in `compressed-grid-probe/probe.jsonl` and `audit.json`. The standalone process takes 26.151s and loads no model.

## Separate A/A control

After the failed screen exposed order sensitivity, a **new, separately bound null control** runs the exact same original pipeline object and original dispatch grid in both slots. The data, output checks, buffer poisoning and timing sequence remain unchanged. This control is not a retry or a correction to the candidate's acceptance criteria; it cannot be used to normalize the failed screen retroactively.

It passes another 4,697,620,480 FP32 comparisons, with identical per-shape output checksums, in 25.154s. B/A below is only a slot ratio: both slots execute the original kernel.

| Chunk start | A/A B/A median in AB order | A/A B/A median in BA order | Median first-run / second-run |
| --- | ---: | ---: | ---: |
{controls}

The identical kernel's first execution is about 3–7% slower on the paired median. The standalone setup poisons and verifies large output buffers on the CPU between pairs; cache and GPU-clock contributions were not separately measured, so the cause remains unproven. Seven alternating pairs contain four AB and three BA orders, which also biases a pooled median when order effects are this large. The 5/7 guard rejects every candidate case, but this harness needs an A/A-neutral protocol with balanced order and equal preconditioning before another small performance claim. Preserve the failed screen; do not reinterpret it as a win after changing the method.

## Build and integrity

All 320 release Rust tests pass, including proportional attribution for both packet shapes and rejection of malformed, overlapping, sentinel or invalid-duration samples. The source is formatted, and the release executable builds. Source audits show that removing only diagnostic guards restores both original attention executors; every Metal source hash matches the prior candidate. The standalone wrapper audit restores the original wrapper exactly, and the null-control source differs only in pipeline/grid identity and labels.

Model SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` (86,720,111,488 bytes). Oracle-v5 source: `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. Diagnostic executable SHA-256: `a174c1e06ff399056b5946bbac1a34583e43ec7aaac0b078c280e4f04a81f3c8`.

Fresh model/prompt/oracle/baseline hashes pass before the model run and afterward. The frozen diagnostic executable and all 40 source inputs still match after both isolated probes, which do not read the model. Plans, source/build/toolchain records, every process result and host load/thermal/swap observations are retained. No failed build, failed numerical run or discarded GPU attempt occurred in this step; the performance screen failed its declared criteria.

## Reproduction and next action

Run `python3 verify.py` here to verify the artifact inventory, source/input/build bindings, full-output receipts, strict counter coverage, component arithmetic, all paired statistics and the A/A control. It does not rerun a GPU or prove execution from hashes alone. `artifacts.json` hashes every other report file. `redactions.json` binds source and stored digests where local paths are replaced by `<repo>`, `<model>` or `<user>`. Binaries and model weights are omitted; their captured hashes remain. Probe source inputs are included so the exact library assembly can be reconstructed without the repository.

The full-model command on the pinned target/build is:

```sh
"$DIAGNOSTIC" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json attention-32k-profiled.json --profile-json attention-32k-profiled-components.json
```

Each standalone probe can be built from its own directory with:

```sh
xcrun clang -O3 -fobjc-arc -framework Foundation -framework Metal probe.m -o probe
./probe combined.metal
```

Run model and GPU experiments sequentially on the target with the recorded configuration. Archived runner/input scripts contain redacted local paths and existing-output guards; they are provenance, not turnkey live rerun scripts.

The next technical action is to establish A/A-neutral standalone timing before selecting another compressed/indexed attention optimization. The newly validated component breakdown supplies the target; it does not yet supply an accepted speed improvement. Keep 256K paused and the earlier grid candidate correctness-only.
'''
(d/'README.md').write_text(md)
# Add the narrow hardware check performed after the experiments.
raw=(w/'hardware-verification.json').read_bytes();(d/'hardware-verification.json').write_bytes(raw)
redactions=read(d/'redactions.json');h=hashlib.sha256(raw).hexdigest();redactions['hardware-verification.json']=dict(original_sha256=h,stored_sha256=h,changed=False);(d/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
files={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file() and p.name!='artifacts.json' and '__pycache__' not in p.parts}
(d/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-research-artifact-manifest-v1',files=files),indent=2)+'\n')
print('Report written;',len(files),'artifacts')
