from pathlib import Path
import json,statistics,re,hashlib
w=Path(__file__).resolve().parent; root=w.parents[2]; d=root/'rust-star/reports/32k-route-order-45520435'
a=json.loads((w/'audit.json').read_text()); b=a['profiles']['baseline']; c=a['profiles']['candidate']; q=a['diagnostic_comparison']; runs=json.loads((w/'validation-acceptance.json').read_text())['runs']; passed=a['later_engine_timing_justified']
component_table='\n'.join(f"| {n} | {b['components'][n]/1000:.6f} | {c['components'][n]/1000:.6f} | {r:.6f} |" for n,r in q['component_total_ratios'].items())
stage_table='\n'.join(f"| {n} | {v['calls']} | {v['gpu_ms']/1000:.6f} | {c['stages'][n]['gpu_ms']/1000:.6f} | {format(c['stages'][n]['gpu_ms']/v['gpu_ms'], '.6f') if v['gpu_ms'] else 'n/a'} |" for n,v in b['stages'].items())
run_table='\n'.join(f"| {r['name']} | PASS | {r['elapsed_seconds']:.3f} | {r['peak_rss_bytes']/2**30:.3f} |" for r in runs)
obs=[json.loads(s) for s in (w/'host-observations.jsonl').read_text().splitlines()]
swap=[float(re.search(r'used = ([\d.]+)M',s['swap'])[1]) for s in obs]
md=f'''# 32K routed-prefill grid-order validation

Candidate `{json.loads((w/'candidate-build.json').read_text())['commit']}` versus frozen Rust baseline `d6c14fa0918a5130da72aa7be6e56b82696fb8b0`, 2026-09-11, Apple M1 Ultra (48 GPU cores, 128 GB).

## Decision

All three complete 32K/128 C0 runs pass. **{'Retain as a correctness-validated candidate for later engine timing' if passed else 'Reject the candidate and restore the prior runtime'}** under the predeclared selection rule. This is a single diagnostic comparison with deliberately perturbed profiling, not accepted engine performance. No Rust-versus-DwarfStar throughput result is updated, no longer frontier is validated, and 256K remains paused.

Routed gate/up totals change from {b['components']['routed_gate_up']/1000:.6f}s to {c['components']['routed_gate_up']/1000:.6f}s (candidate/baseline {q['component_total_ratios']['routed_gate_up']:.6f}). Normalizing against unchanged attention-output, routed-down and shared-expert work gives aggregate ratio **{q['aggregate_normalized_gate_ratio']:.6f}**, and median matched-tail ratio **{q['matched_normalized_gate_ratio']['median']:.6f}**. Both must be below 1.0 to select further timing. The raw gate/up saving is only about 0.20% of baseline summed stage GPU time, so this change alone cannot materially close the reference gap. {q['matched_normalized_gate_ratio']['improved']} of 287 matched tails improve on that diagnostic. Median absolute deviation is {q['matched_normalized_gate_ratio']['mad']:.6f}; all per-tail values are retained in `audit.json`.

## Scope and correctness

Only native 4K continuation-tail fused IQ2 gate/up grid order changes. `build.rs` appends a Rust-owned shader to the existing MoE library. The shared tail lazily creates/selects its pipeline under the existing `native_chunk` predicate and swaps grid axes. Initial prefill, legacy tiles, decode and imported root Metal are unchanged. Reversing two coordinate expressions and the function rename restores the baseline kernel body byte for byte; the IQ2 template parameter remains 16. `implementation.patch`, both kernel sources and `verify.py` preserve/check that identity.

The pinned GGUF is `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` (86,720,111,488 bytes), oracle-v5 source `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. Each run checks 129,280 prefill FP32 words and 128 full decode rows, **16,677,120 words per run; 50,031,360 total**. Tokens alone are insufficient. All immutable output targets remain unchanged. This does not extend the accepted internal-HC frontier.

| Run | Full-output C0 | Process elapsed (s) | Diagnostic peak RSS (GiB) |
| --- | --- | ---: | ---: |
{run_table}

These process durations include conformance/startup work and are not headline timing. Diagnostic RSS omits full engine page warming and does not establish total model/Metal residency or a memory improvement. No timed samples were selected or discarded; these are the three declared corrected-candidate runs.

## Diagnostic attribution

Both profiles pass 287 ordered native-tail packets (chunks 4096 through 28672, layers 2 through 42), each with nine strictly increasing counter intervals: 2,583 intervals per run. The audit rejects missing/duplicate coverage, invalid counters and inconsistent parent attribution; it checks the unchanged bootstrap/prefix coverage as well. Encoder splitting and MTLFence serialization perturb execution. Attributed component times distribute parent GPU duration according to counter deltas; they are not independent end-to-end stopwatches.

| Tail component | Baseline GPU attribution (s) | Candidate (s) | Candidate/baseline |
| --- | ---: | ---: | ---: |
{component_table}

| Stage | Calls | Baseline GPU (s) | Candidate GPU (s) | Candidate/baseline |
| --- | ---: | ---: | ---: | ---: |
{stage_table}

The stage tables retain absolute changes in untouched work so ordinary run variation remains visible. The aggregate ratio uses sums of the three named controls; the matched statistic independently normalizes every corresponding tail. The experiment was sequential baseline then candidate, without alternating repeats; it supplies no confidence interval or production speed guarantee. The initial standalone synthetic screen is separate evidence in `../prefill-route-order-d6c14fa0/`.

## Build, failure and provenance

The first candidate `9171518a` failed library compilation before inference: appended source referenced `QK_NL` after the imported file undefines it. The failed run, source, build metadata, plan and host observations remain under `rejected/`. No profiles ran on that failed candidate. Correction `45520435` substitutes the original constant 16 in two template arguments; it changes no arithmetic or acceptance criterion.

All 318 release Rust tests and formatting passed on the first candidate. Their host-test evidence is explicitly reused for the two-constant Metal correction (`reused-host-tests.json`). The corrected release executable builds; the actual assembled Metal library and new pipeline compile on M1 Ultra (SIMD width 32, maximum 512 threads). All live C0/profile results above belong to that corrected frozen executable. Final verification rehashes the complete model, prompt, oracle source/executables, frozen Rust baseline/candidate and all 40 source inputs.

The sequential runner clears DS4/Rust-Star/Metal override variables, checks frozen source/executable hashes before each run and rejects concurrent model processes. It records host load, thermal status, VM state and swap every 30 seconds. There are {len(obs)} host observations; reported swap usage ranges {min(swap):.2f}–{max(swap):.2f} MiB. Raw observations remain available; this is not proof of exclusive system load.

## Reproduction

Run `python3 verify.py` here for model-free artifact hashes, source/build/input bindings, exact-output receipts and independent profile/statistics recomputation. This verifies retained evidence; it does not rerun the GPU. `artifacts.json` hashes every other report file. `redactions.json` binds original and stored digests for source work artifacts; local absolute paths are replaced with `<repo>`, `<model>` and `<user>`. The portable audit authenticates stored files before using their original digests. Binaries and weights are excluded.

Live reproduction requires the exact target machine/toolchain, pinned source/build and model. Build the runtime from the candidate commit and run sequentially:

```sh
"$CANDIDATE" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json candidate-32k-output.json
"$BASELINE" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json baseline-32k-profiled.json --profile-json baseline-32k-profiled-components.json
"$CANDIDATE" long-prefill-chunk-probe "$MODEL" --context 32768 --output-only --json candidate-32k-profiled.json --profile-json candidate-32k-profiled-components.json
```

`validate.py` is the archived original runner with redacted local paths, not a turnkey rerun script. Supply paths and the same environment/input checks when reproducing. The preserved assembled library can be checked with `compile-shader.m` using Foundation and Metal frameworks.

The next candidate-specific step {'is a bounded alternating unprofiled 32K engine timing comparison against the frozen Rust baseline, with an end-to-end prefill criterion fixed before running' if passed else 'is complete; investigate another profile-led lower-context bottleneck'}. Do not automatically start a larger campaign. No push or release is part of this research validation.
'''
(d/'README.md').write_text(md)
files={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file() and p.name!='artifacts.json' and '__pycache__' not in p.parts}
(d/'artifacts.json').write_text(json.dumps(dict(schema='rust-star-research-artifact-manifest-v1',files=files),indent=2)+'\n')
print(f'Wrote README and {len(files)} artifact hashes')
