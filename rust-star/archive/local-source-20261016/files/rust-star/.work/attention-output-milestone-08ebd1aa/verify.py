from pathlib import Path
import json,hashlib,subprocess,sys,re,datetime,os
w=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());digest=lambda b:hashlib.sha256(b).hexdigest()
manifest=read(w/'artifacts.json')['files']
for name,e in manifest.items():
 p=w/name;raw=p.read_bytes();assert len(raw)==e['stored_bytes'] and digest(raw)==e['stored_sha256'],name
 assert (b'/Users/'+b'deathcodevision') not in raw,name
 if not e['redacted']:assert e['original_sha256']==e['stored_sha256'] and e['original_bytes']==e['stored_bytes'],name
def original(name):return manifest[name]['original_sha256']
plan=read(w/'plan.json');build=read(w/'integration/candidate-build.json');bins=read(w/'binary-identities.json')
for name,h in read(w/'freeze.json')['files'].items():
 assert (bins[name]['sha256'] if name in bins else original(name))==h,name
assert bins['integration/candidate-rust-star']['sha256']==build['executable_sha256']
assert bins['integration/baseline-rust-star']['sha256']==read(w/'engine-plan.json')['baseline_executable_sha256']
for name,h in build['sources'].items():assert original('candidate-sources/'+name)==h,name
for name,h in plan['runtime_sources'].items():
 source='baseline-sources/'+name if name in ['rust-star/runtime/build.rs','rust-star/runtime/src/metal_shim.m'] else 'candidate-sources/'+name
 assert original(source)==h,name
for field,name in [('plan_sha256','plan.json'),('tests_log_sha256','tests.log'),('build_log_sha256','build.log'),('preflight_sha256','preflight.json'),('source_audit_sha256','source-audit.json'),('shader_compilation_sha256','shader-compile.json'),('runner_sha256','validate.py')]:assert original('integration/'+name)==build[field],field
assert original('engine-plan.json')==build['engine_plan_sha256']
assert original('plan.json')==build['parent_plan_sha256'] and original('timing/audit.json')==build['candidate_selection_audit_sha256']
for n,h in read(w/'engine/freeze.json')['files'].items():assert original(n)==h,n
for n,h in read(w/'engine/freeze.json')['adapter_files'].items():assert original('support/'+Path(n).name)==h,n
# Full source scope proof: only the declared properties and single tail function
# change. Invert recorded edits and compare every byte of the complete host.
p='rust-star/runtime/src/metal_shim.m';host=(w/'candidate-sources'/p).read_text();baseline=(w/'baseline-sources'/p).read_text();restored=host
for op in reversed(read(w/'integration/host-edits.json')):
 assert restored.count(op['after'])==1;restored=restored.replace(op['after'],op['before'])
assert restored==baseline
start=lambda s:s.index('int rust_star_metal_run_prefill_layer2_continuation_tail(')
def exterior(s):
 a=start(s);b=s.index('\nint ',a+5);return s[:a]+s[b:]
outer=exterior(host)
for line in ['@property(nonatomic, strong) id<MTLComputePipelineState> attentionNativeHalfLowPipeline;\n','@property(nonatomic, strong) id<MTLComputePipelineState> attentionNativeHalfOutputPipeline;\n']:
 assert outer.count(line)==1;outer=outer.replace(line,'')
assert outer==exterior(baseline)
assert 'native_chunk && chunk_start != 0u &&\n            tile_start == chunk_start' in host
b=read(w/'integration/build-edits.json');assert b['before']==(w/'baseline-sources/rust-star/runtime/build.rs').read_text() and b['after']==(w/'candidate-sources/rust-star/runtime/build.rs').read_text()
shader=(w/'candidate-sources/rust-star/runtime/src/attention_output_half.metal').read_text()
assert shader.replace('rust_star_attention_output_half_t','probe_q8_half_t').replace('rust_star_attention_output_half','probe_q8_output_half')==(w/'candidate-instantiation.metal').read_text()
# Both synthetic and production shader assemblies can be rebuilt from inputs.
sourceaudit=read(w/'source-audit.json');parts=[]
for n,h in sourceaudit['sources'].items():
 assert original('candidate-sources/'+n)==h;parts.append((w/'candidate-sources'/n).read_text())
assert '\n'.join(parts)+'\n'+(w/'candidate-instantiation.metal').read_text()==(w/'combined.metal').read_text()
for field,name in [('combined_sha256','combined.metal'),('candidate_instantiation_sha256','candidate-instantiation.metal'),('arguments_sha256','arguments.h')]:assert original(name)==sourceaudit[field]
# Rebuild the standalone ABI declarations and original production chain.
names=['ds4_metal_args_mul_mm','ds4_metal_args_mul_mm_id_map0','ds4_metal_args_mul_mm_id','ds4_metal_args_dsv4_hc_expand']
structs=[]
for name in names:
 match=re.search(r'struct '+name+r' \{.*?\n\};','\n'.join(parts),re.S);assert match;structs.append(match[0])
assert '#include <stdint.h>\n'+'\n'.join(structs)+'\n'==(w/'arguments.h').read_text()
fragment=baseline[start(baseline):];i=fragment.index('        [encoder setComputePipelineState:context.attentionOutputBatchMapPipeline]');j=fragment.index('        if (tail_samples) {',i)
assert fragment[i:j]==(w/'prior-screen/production-encoding.m').read_text()
assert original('prior-screen/production-encoding.m')==sourceaudit['production_encoding_sha256']
paths=read(w/'integration/assembly.json')['paths'];text=''.join(line+'\n' for p in paths for line in (w/'candidate-sources/rust-star/runtime'/p).read_text().splitlines())
assert text==(w/'integration/assembled.metal').read_text()
assert read(w/'integration/shader-compile.json')['passed'] and build['production_assembly_matches_build']
assert '320 passed; 0 failed' in (w/'integration/tests.log').read_text() and 'Finished `release`' in (w/'integration/build.log').read_text()
assert (w/'integration/fmt.log').read_text()==''
assert plan['reference_checksums']==read(w/'prior-screen/audit.json')['output_checksums']
assert original('prior-screen/plan.json')==plan['parent_plan_sha256']
# Recompute every isolated interval, pair order, exactness count and gate.
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for mode in ['aa-only','timing']:
 subprocess.run([sys.executable,str(w/'audit.py'),mode],stdout=subprocess.DEVNULL,check=True,env=env)
 assert read(w/mode/'audit.json')['contemporary_null_screen_pass']
assert read(w/'timing/audit.json')['candidate_selected_for_later_c0']
# Full-output C0 evidence and pinned input identities.
c0=read(w/'integration/candidate-32k-output.json')
expected=dict(frontier_tokens=32768,checked_logits=129280,checked_decode_steps=128,logits_checksum=2996662110144943749,decode_logits_checksum=1716816788919264169,decode_tokens_checksum=11977321378026793466,selected_token=85166,complete_32k_c0_claim=True,throughput_claim=False,internal_hc_collection=False)
for k,v in expected.items():assert c0[k]==v,k
assert (w/'integration/candidate-32k-output.log').read_text().count('all 129280 logits exact, token ')==128
accept=read(w/'integration/validation-acceptance.json');assert accept['passed'] and accept['full_fp32_words_per_run']==16677120 and len(accept['runs'])==1
assert accept['runs'][0]['exit_code']==0 and accept['runs'][0]['executable_sha256']==build['executable_sha256']
preflight=read(w/'integration/preflight.json');final=read(w/'completion-pins.json');assert preflight['passed'] and final['passed']
assert preflight['model_sha256']==read(w/'engine-plan.json')['model_sha256']
# The portable engine audit is exactly the recorded source with two explicit
# portability edits; original hash lookups require stored-file verification.
s=(w/'engine-audit.py').read_text().replace("sys.path.insert(0,str(root/'rust-star'))","sys.path.insert(0,str(w/'support'))")
s=s.replace("read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()", "read=lambda p:json.loads(p.read_text())\nmanifest=read(w/'artifacts.json')['files']\ndef sha(p):\n key=str(p.relative_to(w));entry=manifest[key]\n assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['stored_sha256']\n return entry['original_sha256']")
assert s==(w/'portable-engine-audit.py').read_text()
subprocess.run([sys.executable,str(w/'portable-engine-audit.py')],stdout=subprocess.DEVNULL,check=True,env=env)
a=read(w/'engine/audit.json');assert a['execution_passed']
progress=read(w/'engine/progress.json')
for prev,following in zip(progress,progress[1:]):assert datetime.datetime.fromisoformat(prev['finished_utc'])<datetime.datetime.fromisoformat(following['started_utc'])
for row in progress:
 run=w/'engine'/row['name'];check=read(run/'preflight.json')
 assert check['passed'] and check['source_count']==42
 observations=[json.loads(line) for line in (run/'host-observations.jsonl').read_text().splitlines()]
 assert observations[0]['phase']=='before' and observations[-1]['phase']=='after'
 assert all(x['phase']=='active' for x in observations[1:-1])
 times=[datetime.datetime.fromisoformat(x['utc']) for x in observations]
 assert times==sorted(times) and times[0]<datetime.datetime.fromisoformat(row['started_utc'])
 assert times[-1]<datetime.datetime.fromisoformat(row['finished_utc'])
 assert all(x[k+'_exit']==0 for x in observations for k in ['thermal','swap','vm_stat'])

for key in ['model_sha256','model_bytes','prompt_sha256','oracle_executable_sha256','cli_sha256','oracle_source','baseline_executable_sha256']:assert final[key]==preflight[key],key
assert final['sources']==build['sources'] and final['candidate_executable_sha256']==build['executable_sha256']
print(json.dumps(dict(passed=True,artifacts=len(manifest),isolated_fp32_words=sum(read(w/m/'audit.json')['fp32_words_compared'] for m in ['aa-only','timing']),full_model_fp32_words=16677120,engine_runs=a['raw_runs'],timed_pairs=a['timed_pairs'],retained=a['candidate_retained_as_engine_improvement']),indent=2))
