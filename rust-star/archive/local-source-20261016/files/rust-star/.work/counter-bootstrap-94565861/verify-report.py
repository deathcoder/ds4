"""Verify the complete portable bootstrap study without Metal or model access."""
from pathlib import Path
import json,hashlib,subprocess,sys,struct,collections
r=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(p,*args):subprocess.run([sys.executable,str(p),*map(str,args)],check=True,stdout=subprocess.DEVNULL)
m=read(r/'artifact-hashes.json')['artifacts']
actual={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file() and p!=r/'artifact-hashes.json' and '__pycache__' not in p.parts}
assert actual==set(m)
for n,x in m.items():assert (r/n).stat().st_size==x['bytes'] and sha(r/n)==x['sha256'],n
run(r/'reference-evidence/verify.py');run(r/'accepted-profile-study/verify.py')
b=read(r/'batch-build.json');base=read(r/'accepted-build.json');freeze=read(r/'inputs/freeze.json')
assert b['executable_sha256']==freeze['candidate_sha256']
assert base['executable_sha256']==freeze['oracle_sha256']==read(r/'reference-evidence/restored-runtime-check.json')['rebuilt_executable_sha256']
assert base['executable_sha256']==read(r/'accepted-profile-study/build.json')['executable_sha256']
for n,d in b['sources'].items():assert sha(r/'candidate-source'/n)==d
proof=r/'correctness';old=r/'reference-evidence/correctness'
checks=read(proof/'batch-validation.json');assert len(checks)==7 and all(x['exit']==0 for x in checks)
manifest=read(r/'inputs/correctness-manifest.json')
for n,digest in manifest['measurement_tools_sha256'].items():assert sha(r/'measurement-tools'/n)==digest
for x in checks:assert sha(proof/(x['name']+'.json'))==manifest['candidate_gates'][x['name']]
assert '312 passed; 0 failed' in (proof/'batch-tests-fixed.log').read_text()
assert 'Finished `release` profile' in (proof/'batch-release.log').read_text()
assert manifest['c0_oracle_executable_sha256']==read(old/'conformance-v5.json')['executable_sha256']
def checksum(p):
 x=0xcbf29ce484222325
 for (v,) in struct.iter_unpack('<I',p):x=((x^v)*0x100000001b3)&0xffffffffffffffff
 return x
f=old/'fixtures/prefill-frontier-32768-v5';d=old/'fixtures/prefill-decode-32768-v5'
prefill=checksum((f/'batch-prefill-logits.f32le.bin').read_bytes());decode=checksum((d/'decode-logits.f32le.bin').read_bytes());tokens=checksum((d/'token-ids.u32le.bin').read_bytes()[4:])
for name in ['batch-32k-output','batch-32k-internal','batch-32k-profiled']:
 x=read(proof/(name+'.json'));assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['checked_logits']==129280 and not x['throughput_claim']
 assert (x['logits_checksum'],x['decode_logits_checksum'],x['decode_tokens_checksum'])==(prefill,decode,tokens)
 assert (proof/(name+'.log')).read_text().count('all 129280 logits exact, token ')==128
internal=read(proof/'batch-32k-internal.json')
for end in [12288,16384,20480,24576]:
 target=read(old/f'long-prefill-chunks-v5/frontier-{end}-a.json');x=next(x for x in internal['chunks'] if x['end']==end)
 assert x['hc_c0_match'] and x['hc_checked_layers']==43
 assert x['hc_checksums']==[int(target[f'oracle_hc_ffn_post-{layer}_pos{end-4096}.bin']['fnv1a_u32'],16) for layer in range(43)]
assert read(proof/'batch-retained.json')['selection']['both_oracle_transcripts_match']
assert not read(proof/'batch-retained.json')['paired_protocol_eligible']
assert read(proof/'batch-8k.json')['second_consecutive_tile']['selector_fixture']=='dwarfstar-oracle-v5-prefill-indexer-selector-4128'
run(proof/'audit-components.py');assert sha(proof/'component-audit.json')==manifest['candidate_profile_audit_sha256']
run(r/'audit-comparison.py',r);assert sha(r/'independent-audit.json')==m['independent-audit.json']['sha256']
decision=read(r/'retention-decision.json');audit=read(r/'independent-audit.json')
assert decision['independent_audit_sha256']==sha(r/'independent-audit.json')
assert decision['paired_plan_sha256']==freeze['plan_sha256']
assert decision['prefill_acceptance_passed']==audit['prefill_acceptance_passed']
assert decision['counter_diagnostics_retained'] and decision['oracle_v5_unchanged'] and decision['stop_after_milestone']
if decision['bootstrap_retained']:assert audit['prefill_acceptance_passed']
if (r/'profile32k.json').exists():
 run(r/'analyze-profile.py',r/'profile32k.json');assert sha(r/'profile32k-analysis.json')==m['profile32k-analysis.json']['sha256']
else:
 x=read(r/'profile-failure.json');assert x['status']=='failed' and x['attempts']==1 and sha(r/'profile32k.log')==x['log_sha256']
 # An invalid sample must carry the raw packet and exact execution location.
 log=(r/'profile32k.log').read_text()
 if 'invalid tail timestamps' in log:assert 'raw_ticks=[' in log and 'chunk=' in log and 'layer=' in log
probe=read(r/'counter-probe-final.json');prov=read(r/'counter-probe-final-provenance.json')
assert sha(r/'tail_counter_probe.m')==prov['source_sha256'] and sha(r/'counter-probe-final.json')==prov['output_sha256']
assert len(probe['records'])==256 and probe['wrong_gpu_output_words']==0
counts=dict(error_values=0,zero_duration_spans=0,backward_spans=0,touching_boundaries=0,overlapping_boundaries=0)
assert {(x['elements'],x['repetition']) for x in probe['records']}=={(n,i) for n in [1,256,4096,65536] for i in range(64)}
for x in probe['records']:
 ticks=x['ticks'];assert len(ticks)==18
 for i,value in enumerate(ticks):
  if value==2**64-1:counts['error_values']+=1
  elif i and ticks[i-1]!=2**64-1:
   a=ticks[i-1]
   if i%2:counts['zero_duration_spans']+=value==a;counts['backward_spans']+=value<a
   else:counts['touching_boundaries']+=value==a;counts['overlapping_boundaries']+=value<a
assert all(probe[k]==v for k,v in counts.items())
print(f'PASS: {len(actual)} artifacts; frozen Rust comparison, full C0, scoped coverage, probe and historical evidence')
