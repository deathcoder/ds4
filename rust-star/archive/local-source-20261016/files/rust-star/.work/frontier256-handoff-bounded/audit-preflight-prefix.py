from pathlib import Path
import json,hashlib,struct,math,re,subprocess,sys,datetime
w=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
root=w.parents[2];read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def original_sha(p):
 if (w/'redactions.json').exists():
  redactions=read(w/'redactions.json');key=str(p.relative_to(w))
  if key in redactions:
   assert sha(p)==redactions[key]['stored_sha256']
   return redactions[key]['raw_sha256']
 return sha(p)

def fnv(raw):
 h=0xcbf29ce484222325
 for (v,) in struct.iter_unpack('<I',raw):h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
b=read(w/'candidate-build.json');i=read(w/'inputs.json');pre=read(w/'preflight.json')
assert b['commit']=='d6c14fa0918a5130da72aa7be6e56b82696fb8b0'
assert b['executable_sha256']=='8bdcf022811eca2c2cf0b5c7aca10bc51705a1a18d988129cffbdfd43bb58d11'
source_root=w/'candidate-source' if (w/'candidate-source').exists() else root
for n,h in b['sources'].items():assert sha(source_root/n)==h,n
for field in ['tests','build','python_tests']:
 file={'tests':'tests.log','build':'build.log','python_tests':'python-tests.log'}[field]
 assert original_sha(w/file)==b[field+'_log_sha256']
assert '318 passed; 0 failed' in (w/'tests.log').read_text()
assert 'Ran 94 tests' in (w/'python-tests.log').read_text() and (w/'python-tests.log').read_text().rstrip().endswith('OK')
assert sha(w/'preflight.json')==b['preflight_sha256']
assert sha(w/'milestone-plan.json')==b['milestone_plan_sha256']
assert i['oracle_source']==pre['oracle_source']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'
assert i['oracle_executable_sha256']==pre['oracle_executable_sha256']=='a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0'
assert i['model_sha256']==pre['model_sha256']=='ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
for field in ['model_bytes','prompt_sha256','cli_sha256']:assert i[field]==pre[field]
assert i['eos_token']==1 and i['available_prompt_tokens']>=262144
assert read(w/'cli-checks.json')['passed'] and read(w/'capacity-probe.json')['passed']
assert read(w/'host-provenance.json')['passed']
assert original_sha(w/'host-current.json')==read(w/'host-provenance.json')['current_host_sha256']
probe=read(w/'capacity-probe.json');capacity=read(w/'capacity-analysis.json')
assert probe['context_tokens']==262144 and probe['chunk_rows']==4096 and probe['physical_memory_bytes']==137438953472
assert len({x['role'] for x in probe['buffers']})==len(probe['buffers'])
assert probe['total_requested_bytes']==sum(x['bytes'] for x in probe['buffers'])
assert probe['largest_requested_bytes']==max(x['bytes'] for x in probe['buffers'])<=probe['max_buffer_length']
assert probe['checked_page_and_end_bytes']==sum((x['bytes']+16383)//16384+1 for x in probe['buffers'])
for row in capacity['history_and_selector_budgets']:
 n=row['tokens'];h={'FullKv':43*n*512*4,'AttnCompressed':sum((n//(128 if layer%2 else 4)+1)*512*4 for layer in range(2,43)),'IndexerCompressed':21*(n//4+1)*128*4}
 assert row['history_bytes']==h and row['total_history_bytes']==sum(h.values())
 assert row['odd_attention_mask_bytes']==4096*(4224+n//128)*2
 assert row['even_indexer_scores_full_chunk_bytes']==4096*(n//4)*4 and row['topk_scratch_upper_bytes']==2*4096*(n//4)*4
prior_host=w/'prior-host.json';prior_raw=w/'prior-128k-paired-raw.json';prior_manifest=w/'prior-128k-artifact-hashes.json'
if not prior_host.exists():
 prior_host=root/'rust-star/reports/128k-score64-b6b0f125/inputs/host-before.json'
 prior_raw=root/'rust-star/reports/128k-score64-b6b0f125/results/paired-raw.json'
 prior_manifest=root/'rust-star/reports/128k-score64-b6b0f125/artifact-hashes.json'
# Packaged redactions preserve original digests; workspace prior files are immutable.
raw_sha=original_sha(prior_raw) if prior_raw.is_relative_to(w) else sha(prior_raw)
assert raw_sha==read(prior_manifest)['artifacts']['results/paired-raw.json']['sha256']
peak=max(x['candidate']['process_peak_bytes'] for x in read(prior_raw)['pairs']);assert peak==capacity['measured_128k_max_peak_bytes']
rows=capacity['history_and_selector_budgets'];estimate=peak+rows[-1]['total_history_bytes']-rows[0]['total_history_bytes']+sum(rows[-1][k]-rows[0][k] for k in ['even_indexer_scores_full_chunk_bytes','topk_scratch_upper_bytes'])
assert estimate==capacity['planning_estimate_128k_peak_plus_history_and_selector_delta_bytes'] and capacity['planning_estimate_is_bound'] is False
host=read(w/'host-current.json');prior=read(prior_host)
for field in ['macos','machine','system_profiler_filtered']:assert host[field]==prior[field]
for field in ['uname','xcodebuild','clang']:assert host['commands'][field]['output']==prior['commands'][field]['output']
assert read(w/'capture-acceptance.json')['passed']
captures=read(w/'capture-progress.json');assert len(captures)==4
assert [(c['context'],c['repetition']) for c in captures]==[(196608,'a'),(196608,'b'),(262144,'a'),(262144,'b')]
for c in captures:
 assert c['exit']==0 and c['process_peak_bytes']>0 and math.isfinite(c['seconds']) and c['seconds']>0
 log=(w/f"oracle-{c['context']}-{c['repetition']}"/'console.log').read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log);assert rss and int(rss[1])==c['process_peak_bytes']
 cmd=c['command'];assert cmd[cmd.index('--ctx-start')+1]==str(c['context'])==cmd[cmd.index('--ctx-max')+1]
 assert cmd[cmd.index('--gen-tokens')+1]==('128' if c['context']==262144 else '0') and '--warm-weights' in cmd
references={}
for n in [131072,196608,262144]:
 d=w/f'reference-{n}'
 if n==131072 and not d.exists():d=root/'rust-star/reports/128k-c0-0eb484ea/reference-131072'
 m=read(d/'manifest.json');assert m['repeat_processes']==2 and m['frontier']==n
 for name,info in m['artifacts'].items():assert sha(d/name)==info['sha256'] and (d/name).stat().st_size==info['bytes']
 raw=(d/'logits.f32le.bin').read_bytes();selected=(d/'selected.u32le.bin').read_bytes();tokens=list(struct.unpack('<'+'I'*(len(selected)//4),selected));count=1 if n==196608 else 129
 assert len(tokens)==count and len(raw)==count*129280*4
 prompt=(d/'prompt.u32le.bin').read_bytes();assert len(prompt)==n*4
 assert all(v[0]<129280 for v in struct.iter_unpack('<I',prompt))
 assert hashlib.sha256(prompt[:32768*4]).hexdigest()=='874c479e2cde7b231dfa4a963d5f1fbe9dd9e60c2842f869e2a117d4860902a0'
 if n==131072:prefix128=prompt
 else:assert prompt[:131072*4]==prefix128
 row_hashes=[]
 for j,token in enumerate(tokens):
  row=raw[j*129280*4:(j+1)*129280*4];values=struct.unpack('<129280f',row)
  assert all(map(math.isfinite,values)) and max(range(len(values)),key=values.__getitem__)==token!=i['eos_token']
  row_hashes.append(hashlib.sha256(row).hexdigest())
 if n!=131072:
  for rep in ['a','b']:
   oracle=w/f'oracle-{n}-{rep}';v=read(oracle/'verification.json')
   assert (oracle/'all-logits.f32le.bin').read_bytes()==raw and (oracle/'selected.u32le.bin').read_bytes()==selected
   assert v['row_sha256']==row_hashes and v['selected_tokens']==tokens and v['all_logits_sha256']==sha(d/'logits.f32le.bin')
   assert v['rows']==count and v['words']==count*129280 and not v['paired_eligible']
 else:
  for rep in ['a','b']:
   receipt=w/f'oracle-131072-{rep}-verification.json'
   if not receipt.exists():receipt=root/f'rust-star/reports/128k-c0-0eb484ea/oracle-131072-{rep}/verification.json'
   v=read(receipt);assert v['row_sha256']==row_hashes and v['selected_tokens']==tokens and v['all_logits_sha256']==sha(d/'logits.f32le.bin')
 references[n]={'frontier_tokens':n,'checked_logits':129280,'checked_decode_steps':count-1,'prefill_logits_checksum':fnv(raw[:129280*4]),'decode_logits_checksum':fnv(raw[129280*4:]) if count>1 else 0,'decode_tokens_checksum':fnv(selected[4:]) if count>1 else 0,'selected_token':tokens[0],'manifest_sha256':sha(d/'manifest.json')}
memory_observations=[json.loads(line) for line in (w/'memory-observations.jsonl').read_text().splitlines()]
assert memory_observations and memory_observations[0]['phase']=='post-rejection baseline before corrected validation'
# Development prefix: future gate coverage is checked by the unmodified final audit.
assert [x['utc'] for x in memory_observations]==sorted(x['utc'] for x in memory_observations)
for x in memory_observations:
 assert x['exit']==0 and x['command']==['vm_stat'] and x['page_size_bytes']==16384
 assert x['scope']=='system-wide; post-rejection baseline before corrected native validation; not fresh boot; no per-process attribution'
 parsed={k:int(v) for k,v in re.findall(r'^([^\n:]+):\s+(\d+)\.',x['raw_stdout'],re.M)}
 assert parsed==x['pages'] and all(v>=0 for v in parsed.values())
 assert {'Pages occupied by compressor','Pages stored in compressor','File-backed pages','Anonymous pages'}<=parsed.keys()
 assert 'page size of 16384 bytes' in x['raw_stdout'] and not x['raw_stderr']
setup=read(w/'runner-setup-correction.json');assert setup['source_candidate_unchanged'] and original_sha(w/'runner-setup-reference-missing.log')==setup['log_sha256']
assert 'FileNotFoundError' in (w/'runner-setup-reference-missing.log').read_text()
# The independent synthetic GPU regression uses separately compiled oracle shaders.
reg=read(w/'rejected-0d5f50ff/regression/probe-verification.json');assert reg['passed'] and reg['shapes']==9 and reg['exact_f32_words']==294912
assert original_sha(w/'rejected-0d5f50ff/regression/probe-verification.json')==read(w/'rejected-0d5f50ff/candidate-build.json')['regression_verification_sha256']
for filename,key in [('decode-nsg-probe.m','probe_source_sha256'),('probe-results.jsonl','results_sha256'),('probe-build.log','build_log_sha256'),('oracle-flash_attn.metal','oracle_flash_sha256')]:
 assert original_sha(w/'rejected-0d5f50ff/regression'/filename)==reg[key]
assert reg['native_shim_sha256']==read(w/'rejected-0d5f50ff/candidate-build.json')['sources']['rust-star/runtime/src/metal_shim.m']
assert reg['native_flash_sha256']==b['sources']['metal/flash_attn.metal']
rows=[json.loads(x) for x in (w/'rejected-0d5f50ff/regression/probe-results.jsonl').read_text().splitlines()];assert rows[-1]=={'passed':True}
assert [x['keys'] for x in rows[1:-1]]==[128,1152,1664,2048,2049,2176,2177,4096,4097]
for x in rows[1:-1]:
 assert x['words']==32768 and x['exact'] and x['nsg']==(1 if x['keys']<=2048 else 2 if x['keys']<=4096 else 4)
 assert (x['old_nsg1_different_words']==0)==(x['keys']<=2048)
# The new production handoff/reset regression checks every copied word at full shapes.
reg=read(w/'regression/probe-verification.json');assert reg['passed'] and reg['all_copy_checks_and_ranges_preserved']
assert original_sha(w/'regression/probe-verification.json')==b['regression_verification_sha256']
assert reg['production_shim_sha256']==b['sources']['rust-star/runtime/src/metal_shim.m']
for name,h in reg['files'].items():assert original_sha(w/'regression'/name)==h,name
cases=reg['cases'];assert [(x['rows'],x['cycle']) for x in cases]==[(131072,1),(131072,2),(262144,1),(262144,2)]
logged=[json.loads(x) for x in (w/'regression/probe-results.jsonl').read_text().splitlines()]
assert [x for x in logged if 'copied_words' in x]==cases and logged[-1]['passed']
for x in cases:
 n=x['rows'];words=43*128*512
 for l in range(2,43):
  ratio=128 if l%2 else 4;words+=(n//ratio)*512+2*(65536 if l%2 else 8192)
  if ratio==4:words+=(n//4)*128+2*2048
 assert x['copied_words']==words and x['untouched_tail_words']==928000 and x['copies']==229
 assert x['commands']==x['waits']==(43 if n>131072 else 1)
 for key in ['exact','per_command_source_release','residency_preserved','reset_passed']:assert x[key] is True
# Ensure the prior decode regression still covers the current unchanged dispatch/shaders.
oldshim=(w/'rejected-0d5f50ff/candidate-source/rust-star/runtime/src/metal_shim.m').read_text()
newshim=(source_root/'rust-star/runtime/src/metal_shim.m').read_text()
start='/* Encode the same exact KV/state copies for both handoff schedules. */'
oldstart='int rust_star_metal_adopt_prefill_decoder_state('
end='int rust_star_metal_copy_completed_output_logits('
assert oldshim[:oldshim.index(oldstart)]==newshim[:newshim.index(start)]
assert oldshim[oldshim.index(end):]==newshim[newshim.index(end):]
oldbody=oldshim[oldshim.index('            id<MTLBuffer> source_raw',oldshim.index(oldstart)):]
oldbody=oldbody[:oldbody.index('\n        [blit endEncoding];\n        const double wall_start')]
assert oldbody.endswith('\n        }');oldbody=oldbody[:-len('\n        }')]
oldbody=oldbody.replace('[blit endEncoding];','').replace('if (layer < 2u) continue;','if (layer < 2u) return 1;').replace('if (ratio != 4u) continue;','if (ratio != 4u) return 1;').replace('copies++;','(*copies)++;').replace('copies += 3u;','*copies += 3u;')
newbody=newshim[newshim.index('    id<MTLBuffer> source_raw',newshim.index(start)):];newbody=newbody[:newbody.index('\n    return 1;\n}')]
assert re.sub(r'\s+','',oldbody)==re.sub(r'\s+','',newbody)
# Preserve and verify the failed predecessor instead of presenting a retry-free run.
r=w/'rejected-c1047756';failure=read(r/'failed-attempt.json');rb=read(r/'candidate-build.json')
assert failure['candidate']==rb['commit']=='c1047756a9186b1c60b6ee5dd323ae81ffbbabc3'
assert failure['candidate_sha256']==rb['executable_sha256']=='f427ceee303921482d3b4550195ee346cb39367525f98916ae4e4a356481d280'
assert failure['passed_native_gates']==8 and failure['failed_gate']=='c0-256k' and not failure['retained_gates_started']
assert failure['log_sha256']==original_sha(r/'checks/c0-256k.log')
assert failure['validation_progress_sha256']==original_sha(r/'validation-progress.json')
assert not (r/'checks/c0-256k.json').exists() and not (r/'validation-acceptance.json').exists()
for name,h in rb['sources'].items():assert sha(r/'candidate-source'/name)==h
for field,file in [('tests','tests.log'),('build','build.log'),('python_tests','python-tests.log')]:assert original_sha(r/file)==rb[field+'_log_sha256']
failed_log=(r/'checks/c0-256k.log').read_text()
assert 'c16ca0ca' in failed_log and 'c16ea2a8' in failed_log and '262144' in failed_log
previous=read(r/'validation-progress.json');assert len(previous)==9 and [x['exit'] for x in previous]==[0]*8+[1]
assert previous[-1]['name']=='c0-256k'
for c in previous:
 log=(r/'checks'/(c['name']+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log);assert rss and int(rss[1])==c['peak_rss_bytes']
old_memory=[json.loads(x) for x in (r/'memory-observations.jsonl').read_text().splitlines()]
assert old_memory and {x['phase'] for x in old_memory}=={'START c0-256k',"AssertionError: ('c0-256k', 1)"}
initial=read(r/'memory-initial-snapshot.json');assert initial['raw_stdout']==(r/'memory-initial-snapshot.txt').read_text()
assert initial['observed_after_utc']<initial['observed_before_utc']==old_memory[0]['utc']
for x in old_memory:
 assert x['exit']==0 and x['page_size_bytes']==16384
 assert x['pages']=={k:int(v) for k,v in re.findall(r'^([^\n:]+):\s+(\d+)\.',x['raw_stdout'],re.M)}
# Preserve the second rejection: first retained trace passes, second handoff OOMs.
r2=w/'rejected-0d5f50ff';f2=read(r2/'failed-attempt.json');b2=read(r2/'candidate-build.json')
assert f2['candidate']==b2['commit']=='0d5f50ff0fc1d725fc2fdd5a23579427647d627a'
assert f2['candidate_sha256']==b2['executable_sha256']=='1ace3c5031d832d34eb0064fdfd71b3feeb0a0c3000d568496b4828eb82548c5'
assert f2['passed_native_gates']==10 and f2['failed_gate']=='retained-256k'
assert f2['retained_256k_full_output_cycles_passed']==1 and f2['second_cycle_prefill_full_output_comparison_passed_before_handoff']
assert f2['second_cycle_decode_steps_completed']==0 and not f2['retry_performed']
assert original_sha(r2/'checks/retained-256k.log')==f2['log_sha256']
assert original_sha(r2/'validation-progress.json')==f2['validation_progress_sha256']
assert not (r2/'checks/retained-256k.json').exists() and not (r2/'validation-acceptance.json').exists()
for name,h in b2['sources'].items():assert sha(r2/'candidate-source'/name)==h
for field,file in [('tests','tests.log'),('build','build.log'),('python_tests','python-tests.log')]:assert original_sha(r2/file)==b2[field+'_log_sha256']
p2=read(r2/'validation-progress.json');assert len(p2)==11 and [x['exit'] for x in p2]==[0]*10+[1]
for c in p2:
 log=(r2/'checks'/(c['name']+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log);assert rss and int(rss[1])==c['peak_rss_bytes']
failed2=(r2/'checks/retained-256k.log').read_text();assert f2['error'] in failed2
positions=[(int(p),int(t)) for p,t in re.findall(r'decode position (\d+) all 129280 logits exact, token (\d+)',failed2)]
ts=struct.unpack('<129I',(w/'reference-262144/selected.u32le.bin').read_bytes());assert positions==list(zip(range(262144,262272),ts[1:]))
assert 'rust_star_metal_adopt_prefill_decoder_state' in oldshim
for x in [json.loads(line) for line in (r2/'memory-observations.jsonl').read_text().splitlines()]:
 assert x['exit']==0 and x['page_size_bytes']==16384
 assert x['pages']=={k:int(v) for k,v in re.findall(r'^([^\n:]+):\s+(\d+)\.',x['raw_stdout'],re.M)}
for name,v in read(w/'reused-artifact-provenance.json').items():assert original_sha(w/name)==v['sha256']

print('PASS: independent reference, source, regression and both rejected-attempt provenance checks; live gates still pending')
