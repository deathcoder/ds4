from pathlib import Path
import json,re,statistics
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());p=Path(read(w/'paired-location.json')['work']);study=root/'rust-star/.work/profile128-9b4d7d0d'
b=read(w/'candidate-build.json');base=read(w/'baseline.json');v=read(p/'optimization-verdict.json');raw=read(p/'results/paired-raw.json');ext=read(p/'extended-summary.json')['contexts'][0];gates=read(w/'validation-acceptance.json')['checks'];profile=read(study/'diagnostic-profile-audit.json');candidate_profile=read(w/'checks/components-128k-audit.json');micro=read(study/'score-tile-audit.json');obs=[json.loads(line) for line in (study/'host-observations.jsonl').read_text().splitlines()]
assert read(p/'independent-audit.json')['passed'] and read(w/'gates-audit.json')['passed'] and read(w/'final-model-verification.json')['passed']
retained=v['accepted'];title='retained' if retained else 'not retained';ratios=ext['candidate_over_oracle'];change=lambda metric:(ratios[metric]['median']-1)*100
rows=[]
for metric,label,scale,unit in [('prefill_ms','Prefill',1000,'s'),('gen_steady_tps','Steady decode',1,'tokens/s'),('gen_first_ms','First decode step',1,'ms'),('process_peak_bytes','Fresh peak RSS',2**30,'GiB'),('process_wall_ms','Process wall time',1000,'s')]:
 a=ext['candidate'][metric];z=ext['oracle'][metric];rows.append(f"| {label} | {a['median']/scale:.3f} {unit} | {z['median']/scale:.3f} {unit} | {ratios[metric]['median']:.6f} | {a['mad']/scale:.3f} / {z['mad']/scale:.3f} |")
pairs='\n'.join(f"| {x['repetition']} | {x['order']} | {x['candidate']['prefill_ms']/1000:.3f} | {x['oracle']['prefill_ms']/1000:.3f} | {x['candidate']['prefill_ms']/x['oracle']['prefill_ms']:.6f} | {x['candidate']['gen_steady_tps']:.3f} | {x['oracle']['gen_steady_tps']:.3f} |" for x in raw['pairs'])
# The portable verifier binds every full metric distribution, not just these rows.
gate_table='\n'.join(f"| {x['name']} | pass | {x['seconds']:.3f} |" for x in gates)
profile_rows='\n'.join(f"| {name} | {profile['stages'][name]['gpu_ms']/1000:.3f} | {candidate_profile['stages'][name]['gpu_ms']/1000:.3f} |" for name in profile['stages'] if profile['stages'][name]['gpu_ms']>0)
micro_rows='\n'.join(f"| {context} | {micro['isolated_pairs'][shape]['baseline_median_ms']:.3f} | {micro['isolated_pairs'][shape]['candidate_median_ms']:.3f} | {micro['isolated_pairs'][shape]['paired_ratio_median']:.6f} |" for context,shape in [('32K','5'),('128K','6')])
thermal=[x for x in obs if x.get('thermal_exit')!=0 or any(line and not line.startswith('Note: No ') for line in x.get('thermal','').splitlines())]
swaps=[float(m[1]) for x in obs if (m:=re.search(r'used = ([0-9.]+)M',x.get('swap','')))]
outcome=(f"The candidate meets the prebound criteria: median paired prefill latency is {abs(change('prefill_ms')):.2f}% lower, with {v['prefill_faster_pairs']}/5 pairs faster. The score tiling change is retained." if retained else "The candidate did not meet every prebound improvement/regression criterion. Its measured result is preserved and the score tiling change is not retained; see disposition.json for the restoration evidence.")
text=f'''# Rust Star: 128K indexer score tiling — {title}

{outcome} This is a Rust-versus-Rust development comparison against frozen accepted `ef37e566`. Oracle-v5 remains the independent correctness oracle; this study does not update the Rust-versus-DwarfStar baseline.

The implementation widens prefill indexer score tiles from 8×32 to 8×64. Eight SIMD groups share query staging across twice as many compressed columns, using 256 threads and 20KiB of shared memory. Half conversion, each 8×8 dot product and ordered FP32 accumulation over heads are preserved. One geometry helper covers all three prefix/continuation call sites. No runtime selector flag or alternate numerical contract was added. Decoder scoring is unchanged.

## Five-pair engine evidence

All five alternating pairs and one warm-up per engine use fresh processes, the same 131,072-token prompt and 128 decode steps. Timed runs disable full-output collection and profiling, then require the exact pinned greedy transcript. Every attempt and all outliers are retained. The unchanged runner schema calls steady decode its generic primary metric; this study's separately frozen acceptance plan evaluates prefill latency and guards decode and RSS. Both scopes are recorded in `inputs/protocol-scope.json`.

| Metric | Candidate median | Accepted Rust median | Median paired ratio | Candidate / baseline MAD |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Steady throughput changes {change('gen_steady_tps'):+.2f}%, first-step latency {change('gen_first_ms'):+.2f}% and fresh peak RSS {change('process_peak_bytes'):+.2f}% on paired medians. Ratios are computed within pairs; a ratio of separate medians is a different statistic. Full minima, maxima, MADs and all numeric metrics are in `extended-summary.json`.

| Pair | Order | Candidate prefill s | Baseline prefill s | Latency ratio | Candidate steady tokens/s | Baseline steady tokens/s |
|---|---|---:|---:|---:|---:|---:|
{pairs}

Acceptance was frozen before measurement: median paired prefill ratio below 1.0, at least four faster pairs, steady throughput ratio at least 0.97 and peak-RSS ratio at most 1.03. First-step latency is reported as a tradeoff. `optimization-verdict.json` records each criterion independently. No favorable retry is permitted.

## Correctness and lifecycle

All 317 Rust tests, 94 Python tests and the release build pass. The ten frozen-candidate live gates below pass. Fresh normal and profiled 128K executions each match all 16,677,120 FP32 words across prefill and 128 decode rows. The 32K modes match the complete vocabulary and decode trace; accepted internal HC targets remain limited to 24K. The 12K gate is an internal-state control, not a new complete-output claim.

| Gate | Result | Diagnostic process seconds |
|---|---|---:|
{gate_table}

Retained 32K and 128K checks each verify two exact transcripts, request-state reset and model/context/residency reuse. They do not collect every logit or count as paired samples; fresh full-logit C0 remains independent. Their timings and diagnostic RSS do not replace warmed fresh-process engine measurements.

## Validated 128K profiling

The original baseline profile stopped at chunk 57,344 / layer 41 with a non-increasing timestamp. All nine individual component spans were positive; the shared-expert span lay inside routed-down. A nine-pass Metal probe produced overlap in 100/100 unfenced runs and reported 100/100 ordered packets with fences. Raw successful fenced microprobe packets were not emitted; the later full-model profiles do retain all their raw intervals. Apple documents that independent passes may overlap ([Metal 3 session](https://developer.apple.com/videos/play/wwdc2022/10101/?time=413)).

Diagnostic `ec5db25e` adds fences only between the nine tail profile encoders and keeps the strict timestamp guard. It passes full 128K output C0 plus all 1,271 packets / 11,439 intervals. Counter-span coverage is {profile['counter_span_coverage']['minimum']*100:.3f}–{profile['counter_span_coverage']['maximum']*100:.3f}% (median {profile['counter_span_coverage']['median']*100:.3f}%). The score candidate also passes the full 128K profile and the 32K 287-packet / 2,583-interval control. No 128K decoder-stage profile is claimed.

The baseline diagnostic assigns 49.04% of recorded stage GPU time to attention, with even-layer indexed attention the largest subgroup at 383.761s. This motivated the score-tile experiment. These are separate perturbed diagnostic runs; scaled tail component shares must not be added to their parent times or used as headline kernel latency.

| Stage | Baseline diagnostic GPU s | Candidate diagnostic GPU s |
|---|---:|---:|
{profile_rows}

## Isolated score control

Seven shapes, including partial tiles and causal masks, match every one of 167,815,044 score words. Seven alternating timing pairs at each large shape all improve. These synthetic kernel measurements justified full-model testing; they do not establish an engine speedup by themselves. `profile-study/probe-commands.json` records standalone reproduction commands for macOS/Metal.

| Shape | Original kernel median ms | Wider kernel median ms | Median paired ratio |
|---|---:|---:|---:|
{micro_rows}

## Integrity, host conditions and disposition

The first freeze-helper attempt rejected different numbers of empty lines left after removing comments. It occurred before model execution; nonempty kernel code lines were identical. Its record is preserved alongside the original rejected profile. Neither numerical nor timestamp validity was relaxed.

There are {len(obs)} host observations spanning profiling, validation and measurement, including {len(thermal)} unusual thermal-query observations. Observed swap use ranges from {min(swaps):.2f} to {max(swaps):.2f}MiB. This was an interactive host; isolated probes and model jobs ran sequentially. Host-side build/audit/report work during ineligible diagnostics is distinct from the paired block. RSS values are observations under the recorded warming policy, not universal model/Metal memory bounds.

Candidate source: `{b['commit']}`; executable SHA-256 `{b['executable_sha256']}`. Timed Rust baseline: `{base['baseline_source_commit']}`; executable SHA-256 `{base['executable_sha256']}`. Oracle-v5 remains `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. The final model, prompt and oracle hash checks pass.

Run `python3 verify.py` without the model to rehash the complete package, validate oracle row hashes and full-output receipts, recompute profile attribution and all paired distributions, and check the prebound verdict. Kernel probes require the documented Metal device. `candidate-source/`, `diagnostic-source/` and the two implementation patches preserve the exact source states; compiled binaries, the model and private absolute-path plan are excluded. Hash-bound measurement JSON is unchanged. `redactions.json` records raw/stored digests for redacted local paths elsewhere.

Stop at this milestone. 256K measurements and the coding-agent corpus remain future work.
'''
(w/'report-README.md').write_text(text)
