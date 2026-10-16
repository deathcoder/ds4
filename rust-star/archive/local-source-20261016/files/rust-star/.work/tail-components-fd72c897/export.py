from pathlib import Path
import json,hashlib,subprocess
w=Path(__file__).resolve().parent;r=w.parents[2];b=json.loads((w/'build.json').read_text());a=json.loads((w/'audit.json').read_text());assert a['passed']
t=r/'rust-star/reports'/('32k-tail-components-'+b['commit'][:8]);t.mkdir(exist_ok=False)
for name in ['components.json','profiled.json','normal.json','profiled.log','normal.log','audit.json','audit.py','build.json','build.log','tests-final.log','tests.log','run.log']:
 text=(w/name).read_text().replace(str(r),'<worktree>');(t/name).write_text(text)
(t/'implementation.patch').write_bytes(subprocess.check_output(['git','diff','fd72c897',b['commit'],'--','rust-star/runtime'],cwd=r))
meta={'baseline_commit':'fd72c897','candidate_commit':b['commit'],'executable_sha256':b['executable_sha256'],'oracle_source':'ccbd16d36895ef9fe8edf7f793014988ff1ab7f4','oracle_executable_sha256':'a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0','model_sha256':'ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0','model_identity_note':'Same resident model used by the accepted tail baseline; not freshly rehashed for this diagnostic-only milestone.','paired_performance_claim':False,'tests_passed':312}
(t/'identity.json').write_text(json.dumps(meta,indent=2)+'\n')
totals={}
for k,v in a['summary'].items():
 name=k.split('/')[1];totals[name]=totals.get(name,0)+v['attributed_gpu_ms']
total=sum(totals.values());table='\n'.join(f'| {k} | {v/1000:.3f} | {v/total*100:.2f}% |' for k,v in sorted(totals.items(),key=lambda x:-x[1]))
(t/'README.md').write_text(f'''# Native 32K tail component profile

Candidate `{b['commit']}` adds diagnostic component attribution to the accepted
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
{table}

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
''')
(t/'verify.py').write_text('''from pathlib import Path
import hashlib,json,subprocess,sys
r=Path(__file__).resolve().parent
manifest=json.loads((r/'artifact-hashes.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
subprocess.run([sys.executable,str(r/'audit.py')],check=True,stdout=subprocess.DEVNULL)
assert hashlib.sha256((r/'audit.json').read_bytes()).hexdigest()==manifest['audit.json']
print(f"PASS: {len(manifest)} artifacts; both full-output gates and all 2583 component rows verified")
''')
manifest={str(p.relative_to(t)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(t.rglob('*')) if p.is_file()};(t/'artifact-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
(w/'report.json').write_text(json.dumps({'report':str(t),'passed':True},indent=2)+'\n');print(t)
