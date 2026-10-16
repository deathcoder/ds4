from pathlib import Path
import json,statistics
source=Path(__file__).resolve().parent;target=Path(json.loads((source/'report-location.json').read_text())['path'])
def read(name):return json.loads((target/name).read_text())
a=read('independent-audit.json');assert a['passed'];summary=read('results/paired-summary.json');c=summary['contexts'][0];b=read('baseline-summary.json')['contexts'][0];before=read('correctness/profile-before-analysis.json');after=read('profile32k-analysis.json');build=read('correctness/batch-build.json')
old=b['candidate']['prefill_ms']['median']/1000;new=c['candidate']['prefill_ms']['median']/1000;ref=c['oracle']['prefill_ms']['median']/1000
speed=old/new;reduction=100*(1-new/old);normalized=c['candidate_over_oracle']['prefill_tps']['median']/b['candidate_over_oracle']['prefill_tps']['median']
s=f'''# Native32K prefill tail batching

The prefill optimization milestone is complete for candidate `{build['commit']}` against unchanged oracle-v4 `{summary['oracle']['source_commit']}`. Five complete pairs and both warmups pass; exact full-logit gates, retained reset/reuse, and independent audit pass.

Median Rust prefill falls from **{old:.3f} s to {new:.3f} s**: **{reduction:.2f}% less latency ({speed:.3f}× throughput)** relative to the previous audited baseline. The current reference median is {ref:.3f} s. These are fixed 32K/128 development results on the M1 Ultra (48 GPU cores, 128 GB memory), not the project's 256K headline.

## Measured problem and change

A profile of the unoptimized scheduler reconciled 1,483 stage rows with aggregate GPU/command timers. It identified even-layer tails as the largest GPU cost: 294.625 s across 18,816 calls, versus 40.337 s across 140 odd-layer batched calls. Binding itself took only 3.875 s, so binding allocation was not the explanation for the aggregate setup cost.

Each even layer previously ran attention and its output/FFN tail in 128 separate 32-token tiles per chunk. The new schedule retains each tile's inverse-RoPE attention heads with a blit in the existing command, then executes the same tail kernels once on all 4,096 rows. This reduces even-tail calls to 147 across the full request. Guards require consecutive tiles and a complete chunk before the tail can execute. The shared 512 MiB attention buffer participates in the existing prefill ownership and reset cleanup.

Shared Metal kernels, arithmetic kernels, oracle source, model and immutable targets are unchanged. Batch grouping changes scheduling; exactness was established by full-output comparison rather than assumed from this design.

## Correctness

The candidate passed an early internal12K localization gate; two fresh full32K gates (output-only and internal) matching all 129 full-logit rows and GPU selections; exact 2K/8K controls; and both 32K/128 cycles in a retained Metal context. Established internal checkpoints through 24K pass; later internal hashes remain diagnostic. All 310 Rust tests and 91 Python tests pass.

Oracle-v4 acceptance and complete trace evidence remain pinned in [the baseline report](../32k-baseline-cbb078bb/README.md). The exact same oracle executable is used here. Model and prompt were freshly rehashed before freezing the new plan. Sampling remains included in both engines' intervals; the fixed full-logit trace establishes identical non-EOS selections despite their different general EOS policies.

## Paired results

One fresh-process warmup per engine precedes five pairs in AB/BA/AB/BA/AB order. Pairs 1–4 passed on first attempts. A host restart interrupted pair 5, leaving its candidate without a completed result. Uptime reset from over 16 hours to minutes and swap reset to zero on resumption; the restart trigger is unavailable. The original state and reference measurement are retained; the complete fifth pair was rerun under the protocol’s documented external-interruption rule. No result was excluded for speed. The campaign therefore spans an interruption and is not a continuously controlled host session. Values are median ± median absolute deviation (MAD), not confidence intervals.

| Metric | Optimized Rust | Oracle-v4 |
| --- | ---: | ---: |
'''
for label,key,scale,unit in [('Steady decode','gen_steady_tps',1,'tokens/s'),('All 128 decode steps','gen_tps',1,'tokens/s'),('First decode step','gen_first_ms',1,'ms'),('Prefill','prefill_ms',.001,'s'),('Whole process','process_wall_ms',.001,'s')]:
 s+='| '+label+' | '+' | '.join(f"{c[role][key]['median']*scale:.3f} ± {c[role][key]['mad']*scale:.3f} {unit}" for role in ['candidate','oracle'])+' |\n'
s+='| Peak process memory | '+' | '.join(f"{a['memory_distributions'][role]['median']:.3f} ± {a['memory_distributions'][role]['mad']:.3f} GiB" for role in ['candidate','oracle'])+' |\n'
r=c['candidate_over_oracle']['gen_steady_tps'];pref=c['candidate_over_oracle']['prefill_tps']
valid=[p for p in read('results/paired-raw.json')['pairs'] if p['valid']];pre_restart=valid[:4]
pre_ratio=statistics.median(p['candidate']['prefill_tps']/p['oracle']['prefill_tps'] for p in pre_restart)
last_ratio=valid[-1]['candidate']['prefill_tps']/valid[-1]['oracle']['prefill_tps']
s+=f'\nRestart sensitivity: the first four pairs have median paired prefill-throughput ratio {pre_ratio:.5f}; the complete post-restart pair has ratio {last_ratio:.5f}. All five valid pairs remain in the declared aggregate.\n'

s+=f'''
The median paired Rust/reference steady-decode ratio is {r['median']:.5f}, range {r['min']:.5f}–{r['max']:.5f}. The median paired prefill-throughput ratio is {pref['median']:.5f}, range {pref['min']:.5f}–{pref['max']:.5f}. Rust prefill remains {new/ref:.3f}× slower by the ratio of latency medians.

The old and new Rust runs are separate benchmark campaigns, not direct old/new pairs. The reference-normalized prefill-throughput improvement is {normalized:.3f}×, compared with {speed:.3f}× from Rust latency medians. The reference prefill median changed from {b['oracle']['prefill_ms']['median']/1000:.3f} s to {ref:.3f} s; retained host/load observations allow that comparison to be assessed. First-decode-step latency excludes prefill and is not time to first output.

## Separate profiles

Both profiles match the fixed transcript and are paired-ineligible. The before profile was collected before choosing the change; the after profile was collected after all timed pairs. Existing command timings and host timers add no prefill GPU waits. Bootstrap work is explicitly grouped by layer range, prefix work by layer, and continuation work by stage. These are stage observations rather than per-kernel attribution. Nested intervals and independently sampled medians must not be summed into an exact wall-time partition.

| Continuation stage | Before GPU seconds | After GPU seconds | Before / after calls |
| --- | ---: | ---: | ---: |
'''
for name in sorted(after['group_totals'],key=lambda n:after['group_totals'][n]['gpu_ms'],reverse=True):
 if name.startswith(('even/','odd/')):
  x=before['group_totals'][name];y=after['group_totals'][name];s+=f"| {name} | {x['gpu_ms']/1000:.3f} | {y['gpu_ms']/1000:.3f} | {x['calls']} / {y['calls']} |\n"
largest=max(after['group_totals'],key=lambda n:after['group_totals'][n]['gpu_ms'])
s+=f'''
The largest remaining measured GPU stage is `{largest}` at {after['group_totals'][largest]['gpu_ms']/1000:.3f} s in the after profile. The next milestone should investigate that stage's batching, allocation and command costs while preserving the same exact gates, then validate any change with this paired protocol. Broader native contexts and agent co-design remain future work.

## Evidence and audit

`verify.py` rehashes the portable evidence, checks all twelve accepted warmup/timed measurement bundles plus the interrupted pair’s completed reference bundle and recomputes the paired statistics. `independent-audit.json` also records verification of the frozen executable hashes, exact transcripts and byte-identical official summary regeneration. `profile32k-analysis.json` and `correctness/profile-before-analysis.json` reconcile profile coverage and interval sums. `correctness/` retains the full live gates and build identities.

There are {a['host_observations']} timed-block host observations and {len(a['unusual_thermal_observations'])} unusual thermal observations. Desktop applications remained open; this is not a dedicated idle-host claim. Profiling remains excluded from throughput samples. The summary's `headline_eligible` flag means protocol eligibility, not completion of the 256K project target.

`portability.json` records the plan template and test-log path rendering; raw measurements retain original bytes. Run `python3 {target.relative_to(source.parents[2])}/verify.py` from the repository root.
'''
(target/'README.md').write_text(s)
print(target/'README.md')
