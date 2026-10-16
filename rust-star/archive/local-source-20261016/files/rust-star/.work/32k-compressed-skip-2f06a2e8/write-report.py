from pathlib import Path
import hashlib,json
w=Path(__file__).resolve().parent;root=w.parents[2];d=root/'rust-star/reports/32k-compressed-skip-9d35953f'
a=json.loads((w/'audit.json').read_text());s=a['selection'];b=json.loads((w/'candidate-build.json').read_text());accept=json.loads((w/'validation-acceptance.json').read_text())
assert s['criteria_pass'], 'This report template is for a selected candidate; write the rejection report explicitly instead'
base=a['runs']['baseline-32k-profiled'];cand=a['runs']['candidate-32k-profiled']
rows='\n'.join(f"| {x['chunk_start']:,} | {x['baseline_target_ms']/1000:.6f} | {x['candidate_target_ms']/1000:.6f} | {x['raw_ratio']:.6f} | {x['normalized_aggregate_ratio']:.6f} | {x['median_normalized_ratio']:.6f} |" for x in a['per_chunk'])
runs='\n'.join(f"| {r['name']} | pass | {r['elapsed_seconds']:.3f} | {r['peak_rss_bytes']:,} |" for r in accept['runs'])
stages='\n'.join(f"| {k} | {v['baseline_gpu_ms']/1000:.6f} | {v['candidate_gpu_ms']/1000:.6f} | {v['ratio']:.6f} |" for k,v in a['stages'].items() if v['ratio'] is not None)
obs=[json.loads(line) for line in (w/'host-observations.jsonl').read_text().splitlines()]
text=f'''# Native compressed run-skip: full 32K correctness and component check

Candidate `{b['commit']}` passes all three full-output 32K/128 C0 runs,
checking **50,031,360 FP32 words**, and passes the prospectively declared
component selection. Retain it as a correctness-validated candidate.
**This is not an accepted engine speedup or a comparison against DwarfStar.**

Compressed-attention attributed GPU time changes from
{s['baseline_target_ms']/1000:.6f}s to {s['candidate_target_ms']/1000:.6f}s
({100*(1-s['raw_aggregate_ratio']):.2f}% lower). The control-normalized aggregate
ratio is {s['normalized_aggregate_ratio']:.6f}, and the median of 140 matched
normalized ratios is {s['median_matched_normalized_ratio']:.6f}.
The saved {((s['baseline_target_ms']-s['candidate_target_ms'])/1000):.6f}s is only
{100*s['recorded_stage_time_saving_fraction']:.3f}% of the baseline recorded GPU
stage sum. That fraction is diagnostic context, not an end-to-end prediction.

## Scope and prior evidence

Starting worktree commit: `2f06a2e8`. Frozen comparator:
`1711856ea309f70877102d15facb4e1243a9fdcc`, executable SHA-256
`a174c1e06ff399056b5946bbac1a34583e43ec7aaac0b078c280e4f04a81f3c8`.
Candidate executable SHA-256: `{b['executable_sha256']}`.

The Rust-owned `compressed_prefill.metal` is the isolated candidate from
[the run-skip study](../compressed-skip-runs-fbbea533/README.md) with symbol
renaming only. It scans four zero block markers at a time, handles scalar
leftovers, and consolidates mask-pointer advances. The original active-block
arithmetic, grid and partial padded tail remain intact. The prior study's
119,150 integer traces, 120 GPU edge cases and controlled isolated timing are
reused as prior evidence, not claimed as rerun here.

The new source is appended only to the attention-ingress library. Two lazy
NSG8/512 pipeline specializations handle padded/unpadded inputs only when
`prefillLayerChunkStart && position_start == chunk_start && position_start != 0`.
All initial-prefix, legacy continuation and decode dispatches remain unchanged,
as do all imported kernel sources. Removing the scoped additions restores the entire original
host and build files byte for byte; 38 other frozen inputs are unchanged.
The standalone compilation uses the same assembled source, default Metal
compile options and function constants as production. Both pipelines pass.

## Predeclared validation and results

`plan.json` predates the frozen executable and all inference runs. It declares
exactly three fresh processes: normal candidate C0, baseline profiled C0, and
candidate profiled C0. Every run compares all 129,280 logits for prefill plus
128 decode steps against immutable FP32 targets: 16,677,120 words per run.
All output JSON fields, full-logit receipts and decode checksums pass.

| Run | Full C0 | Observed elapsed seconds | Peak process RSS bytes |
|---|---|---:|---:|
{runs}

Elapsed times include the correctness harness and startup and are not paired
performance measurements. `/usr/bin/time -l` RSS is a process diagnostic, not
a claim about full model or unified-memory residency. All 320 release Rust
host tests, format check and release build pass. No additional tests, model
runs, retiming or threshold changes were added after results were observed.

Both profiles contain all 287 native attention packets (147 indexed, 140
compressed; 1,883 component intervals) and all 287 tail packets (2,583
intervals). Every interval is strictly ordered, finite and positive; packet
attributions sum to their recorded parents. Initial bootstrap layers 0–2
and all 40 prefix layers are present. `audit.py` recomputes coverage,
component sums, every matched ratio and the selection from the raw profiles.

Selection requires all three ratios below one: raw compressed-attention
aggregate, target/control aggregate, and median matched target/control ratio.
Controls are `attention_output`, `routed_down` and `shared_expert` from the
same 140 compressed-layer tail packets. All three criteria pass.
{s['matched_improved']}/140 normalized packet ratios improve; median absolute
deviation is {s['matched_normalized_mad']:.6f}.

| Chunk start | Baseline target s | Candidate target s | Raw ratio | Normalized aggregate | Matched normalized median |
|---:|---:|---:|---:|---:|---:|
{rows}

The effect is not uniform across chunks: normalized aggregates are slightly
worse at starts 4,096 (+0.89%) and 12,288 (+0.24%). The preset overall and
matched-median criteria still pass; there was no per-chunk acceptance rule.
The unchanged control sum itself is {100*(1-s['candidate_control_ms']/s['baseline_control_ms']):.2f}%
lower in the candidate process. This broad variation is why the raw 6.76%
component reduction must not be treated as an isolated causal engine gain.

For context, the full stage attribution is below. Components and stages include
encoder splitting, counter collection and fences; they perturb execution.

| Stage | Baseline GPU s | Candidate GPU s | Ratio |
|---|---:|---:|---:|
{stages}

## Limits and next decision

There is only one baseline/candidate profile pair, always in that order.
Control normalization can expose broad drift but cannot remove every source
of timing variation. The isolated equal-warm A/A method is a separate result;
it is not applied retroactively to these real-model profiles. No unprofiled
engine pair, total-prefill speedup, DwarfStar parity or new frontier is accepted.
Assess these small real-component savings before spending on broader engine
timing; keep any later engine criteria and run count prospective and bounded.

The hardware was checked live as M1 Ultra, 48 GPU cores and 128 GB memory.
Periodic `pmset`, swap and VM observations are preserved in
`host-observations.jsonl`; GPU clocks are not measured. First observed swap:
`{obs[0]['swap']}`. Last observed swap: `{obs[-1]['swap']}`.
Full model, prompt, oracle source/executables, baseline/candidate executables
and all 41 runtime/imported input hashes pass final verification. The oracle
tracked tree remains clean. Model/GPU work is finished. No push or 256K run.

## Portable evidence

Run `python3 verify.py` here. It verifies the artifact manifest, original/stored
redaction bindings, source restoration and assembly, build/input/executable
bindings, all three full-output records and both strict profiles. The candidate
full host source is losslessly stored as `candidate-host.m.gz`. Binaries and
model weights are omitted; their hashes remain in the manifests. User-specific
paths are replaced by `<repo>`, `<model>` and `<user>` with original/stored
SHA-256 bindings. Replay scripts are provenance, not directly runnable with
those placeholders; the portable verifier needs only Python's standard library.

For reproduction, build the pinned runtime and run
`rust-star long-prefill-chunk-probe "$MODEL" --context 32768 --output-only
--json output.json`; append `--profile-json components.json` for the diagnostic
mode. Use the pinned oracle fixtures and prompt in the source checkout.
'''
(d/'README.md').write_text(text)
files={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifacts.json'}
(d/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-report-artifacts-v1',files=files),indent=2)+'\n')
print('Report written:',len(files),'artifacts')
