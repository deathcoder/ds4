from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];prior=root/'rust-star/.work/selective-clear-db1de4a9'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(w/n).write_text(json.dumps(x,indent=2)+'\n')
state=read(prior/'final-runtime-state.json');assert all(sha(root/p)==h for p,h in state['sources'].items())
assert sha(root/'rust-star/.work/runtime-target/release/rust-star')==state['default_executable_sha256']
for n in ['final-runtime-state.json','candidate-build.json','validation-acceptance.json','profile-audit.json','audit-profile.py','candidate-32k-profiled.json','candidate-32k-profiled.log','candidate-32k-profiled-profile.json','oracle-32k.json']:
 p=prior/n
 if p.exists():shutil.copy2(p,w/n)
shutil.copy2(prior/'plan.json',w/'prior-plan.json')
ref=root/'rust-star/.work/oracle-v5-source'
prov=root/'rust-star/.work/upstream-refresh-1190237b/oracle-v5-benchmark-build.json';shutil.copy2(prov,w/prov.name)
assert sha(ref/'ds4-bench')=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
assert all(sha(ref/p)==h for p,h in read(prov)['source_sha256'].items())
(w/'adapters').mkdir()
for n in ['measure_ruststar.py','measure_dwarfstar.py','ruststar_measurement_lib.py','dwarfstar_measurement_lib.py']:
 shutil.copy2(root/'rust-star'/n,w/'adapters'/n)
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
plan=dict(schema='rust-star-direct-reference-comparison-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runtime_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),context=32768,gen_tokens=128,orders=['AB','BA','AB','BA','AB'],warmups=['A','B'],role_A='pinned DwarfStar oracle-v5',role_B='accepted Rust Star selective initialization, Rust1.74',executables={'A':str(ref/'ds4-bench'),'B':str(root/'rust-star/.work/runtime-target/release/rust-star')},executable_sha256={'A':sha(ref/'ds4-bench'),'B':state['default_executable_sha256']},model=str(model),prompt=str(root/'speed-bench/promessi_sposi.txt'),model_stat={'size':model.stat().st_size,'mtime_ns':model.stat().st_mtime_ns},primary_comparison='Rust/DwarfStar paired prefill latency ratio; no optimization retention decision',secondary_metrics=['gen_steady_tps','gen_tps','gen_first_ms','process_wall_ms','process_peak_bytes'],timing='Both sampling inclusive; prefill includes initial selection, each decode eval includes resulting token selection. Exact single frontier; built-in weight warming. No profiling/correctness collection in timed runs.',correctness='Unchanged oracle-v5 source/executable and accepted Rust full C0 evidence (50031360 exact FP32 comparisons) reused. Rust timed runs check fixed transcript. DwarfStar CSV does not expose a per-run transcript: correctness rests on pinned producer acceptance, not a new full-logit capture.',failure_policy='Stop on any process, parser, pin or correctness failure; preserve all evidence. No discretionary repeats or exclusions. Warmups excluded prospectively. Every completed pair included regardless of speed. No parity claim from transcript alone; no statistical significance claim from five pairs.',reporting='Median/MAD/min/max, all paired ratios, separate AB and BA medians, raw host observations. Reference zero snapshot bytes is unavailable KV memory, not zero.',no_256k=True,no_push=True,stop_after_milestone=True)
write('plan.json',plan)
host={}
for name,cmd in [('os',['sw_vers']),('hardware',['system_profiler','SPHardwareDataType','SPDisplaysDataType']),('uptime',['uptime']),('thermal',['pmset','-g','therm'])]:
 x=subprocess.run(cmd,capture_output=True,text=True);host[name]={'exit_code':x.returncode,'stdout':x.stdout}
# Omit hardware serials and UUIDs from portable record.
host['hardware']['stdout']='\n'.join(s for s in host['hardware']['stdout'].splitlines() if not any(k in s for k in ['Serial Number','Hardware UUID','Provisioning UDID']))
write('host.json',host)
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();entry='''## 2026-09-12 — Start direct 32K comparison against pinned DwarfStar

- D authorized the next single milestone. Measurement only: unchanged accepted
  Rust e0bec446 versus oracle-v5 ccbd16d3, same model and 32K/128 workload.
- Frozen protocol: one excluded warmup per engine, five AB/BA/AB/BA/AB pairs,
  strict existing adapters, sampling-inclusive timing and passive host records.
  Reuse verified full-C0 evidence; do not call timed transcript checks full C0.
- Evidence: rust-star/.work/dwarfstar-32k-e0bec446. Verify complete input pins
  before and after; stop on failure, preserve all runs, no speed exclusions.
- Finish portable report, next bottleneck proposal, local commit and update,
  then STOP. No optimization edits, larger frontiers or publication.

''';s=s.replace('## 2026-09-12 — Complete selective',entry+'## 2026-09-12 — Complete selective',1);s=s.replace('- COMPLETED SINGLE MILESTONE: selective scratch initialization retained.','- ACTIVE SINGLE MILESTONE: direct 32K DwarfStar comparison in\n  rust-star/.work/dwarfstar-32k-e0bec446; measurement only, then STOP.\n- COMPLETED: selective scratch initialization retained.',1);s=s.replace('  Current work is complete; report the result and next proposal, then STOP.','  Accepted runtime is unchanged during the direct comparison.',1)
a=s.index('## Immediate Next Actions\n');b=s.index('\n## Entries',a);s=s[:a]+'''## Immediate Next Actions

1. Complete frozen direct 32K/128 DwarfStar comparison, including full input
   pins before/after and all twelve fresh engine processes.
2. Audit portable evidence and report all five pairs and order sensitivity.
3. Choose next bottleneck from the accepted profile; no optimization yet.
4. Commit locally, send milestone update and next proposal, then STOP.
   Keep oracle-v5 immutable, 256K paused and publication deferred.
'''+s[b:];p.write_text(s)
print('Prepared',w)
