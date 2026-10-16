from pathlib import Path
import json,hashlib,struct,math,re,subprocess,sys
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
assert b['commit']=='c1047756a9186b1c60b6ee5dd323ae81ffbbabc3'
assert b['executable_sha256']=='f427ceee303921482d3b4550195ee346cb39367525f98916ae4e4a356481d280'
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
initial_memory=read(w/'memory-initial-snapshot.json')
assert initial_memory['raw_stdout']==(w/'memory-initial-snapshot.txt').read_text()
assert initial_memory['phase']=='START c0-256k' and initial_memory['exit']==0 and initial_memory['command']==['vm_stat']
assert initial_memory['page_size_bytes']==16384 and not initial_memory['raw_stderr']
assert initial_memory['pages']=={k:int(v) for k,v in re.findall(r'^([^\n:]+):\s+(\d+)\.',initial_memory['raw_stdout'],re.M)}
assert initial_memory['pages']['Pages occupied by compressor']==1338211 and initial_memory['pages']['Pages stored in compressor']==1660526
memory_observations=[json.loads(line) for line in (w/'memory-observations.jsonl').read_text().splitlines()]
assert memory_observations and memory_observations[0]['phase']=='START c0-256k'
assert initial_memory['observed_after_utc']<initial_memory['observed_before_utc']==memory_observations[0]['utc']
assert {'START c0-256k','START retained-128k','START retained-256k'}<={x['phase'] for x in memory_observations}
assert [x['utc'] for x in memory_observations]==sorted(x['utc'] for x in memory_observations)
for x in memory_observations:
 assert x['exit']==0 and x['command']==['vm_stat'] and x['page_size_bytes']==16384
 assert x['scope']=='system-wide; began during late fresh 256K validation, no earlier baseline'
 parsed={k:int(v) for k,v in re.findall(r'^([^\n:]+):\s+(\d+)\.',x['raw_stdout'],re.M)}
 assert parsed==x['pages'] and all(v>=0 for v in parsed.values())
 assert {'Pages occupied by compressor','Pages stored in compressor','File-backed pages','Anonymous pages'}<=parsed.keys()
 assert 'page size of 16384 bytes' in x['raw_stdout'] and not x['raw_stderr']
assert max(x['pages']['Pages occupied by compressor'] for x in memory_observations)>0
checks=w/'checks';a=read(w/'validation-acceptance.json');expected_names=['batch-2k','batch-8k','batch-12k','batch-32k-output','batch-32k-internal','batch-32k-profiled','c0-128k','c0-192k','c0-256k','retained-128k','retained-256k']
assert a['passed'] and a['candidate']==b['commit'] and [x['name'] for x in a['checks']]==expected_names
assert a['checks']==read(w/'validation-progress.json') and a['references']=={str(k):v for k,v in references.items()}
for c in a['checks']:
 assert c['exit']==0 and c['peak_rss_bytes']>0 and math.isfinite(c['seconds']) and c['seconds']>0
 name=c['name'];log=(checks/(name+'.log')).read_text();rss=re.search(r'(\d+)\s+maximum resident set size',log);assert rss and int(rss[1])==c['peak_rss_bytes']
 x=read(checks/(name+'.json'))
 if name.startswith(('c0-','retained-')):
  n=int(name.split('-')[1][:-1])*1024;retained=name.startswith('retained-')
  if retained:
   assert x['schema']=='rust-star-retained-frontier-check-v1' and x['cycles']==2 and len(x['checks'])==2
   for key in ['same_process','same_model_mapping','same_metal_context','request_state_reset_between_cycles','model_warm_reused_on_retained_cycle','retained_model_residency_reused']:assert x[key] is True,key
   assert x['first_model_residency_reused'] is False and x['paired_protocol_eligible'] is False and x['throughput_claim'] is False
   assert '--retained' in c['command'];cycles=x['checks'];assert cycles[0]==cycles[1]
  else:cycles=[x]
  for cycle in cycles:
   for key,value in references[n].items():
    if key!='manifest_sha256':assert cycle[key]==value,(name,key)
   assert cycle['external_reference_match'] and not cycle['oracle_provenance_verified_by_runtime'] and not cycle['throughput_claim']
  positions=[(int(p),int(t)) for p,t in re.findall(r'decode position (\d+) all 129280 logits exact, token (\d+)',log)]
  assert len(positions)==len(cycles)*references[n]['checked_decode_steps']
  if positions:
   d=w/f'reference-{n}' if n!=131072 or (w/f'reference-{n}').exists() else root/'rust-star/reports/128k-c0-0eb484ea/reference-131072'
   ts=struct.unpack('<129I',(d/'selected.u32le.bin').read_bytes());assert positions==list(zip(range(n,n+128),ts[1:]))*len(cycles)
 elif name.startswith('batch-32k'):
  assert x['complete_32k_c0_claim'] and x['checked_decode_steps']==128 and x['checked_logits']==129280
  assert x['logits_checksum']==2996662110144943749 and x['decode_logits_checksum']==1716816788919264169 and x['decode_tokens_checksum']==11977321378026793466
  assert log.count('all 129280 logits exact, token ')==128
 elif name=='batch-8k':
  assert x['complete_8k_transformer_claim'] and x['output_logits_c0_claim']
  assert x['sparse_transition']['score_kernel']=='rust_star_indexer_scores_tiled64'
  assert x['layer4_prefix_complete']['sparse_attention']['kernels'][0]=='rust_star_indexer_scores_tiled64'
 elif name=='batch-2k':assert x['output_head']['full_logits_c0_bitwise_match']
for name in ['batch-12k','batch-32k-internal']:
 prior=w/'prior-controls'/(name+'.json')
 if not prior.exists():prior=root/'rust-star/reports/32k-memory-de6d2c11/correctness'/(name+'.json')
 current=read(checks/(name+'.json'));original=read(prior)
 assert current['internal_hc_collection'] and len(current['chunks'])==len(original['chunks'])
 for actual,expected in zip(current['chunks'],original['chunks']):
  if actual['end']<=24576:assert actual==expected
before=sha(checks/'component-audit.json');subprocess.run([sys.executable,str(checks/'audit-components.py')],check=True,stdout=subprocess.DEVNULL);assert sha(checks/'component-audit.json')==before
result={'passed':True,'candidate':b['commit'],'oracle_captures':4,'native_gates':11,'fresh_256k_full_c0_processes':1,'retained_contexts':[131072,262144],'retained_full_c0_cycles_per_context':2,'full_256k_native_output_cycles':3,'profile32_packets':287,'all_output_word_receipts_recomputed':True,'all_oracle_rows_finite_and_argmax_checked':True,'source_hashes_verified':True,'all_process_memory_receipts_checked':True,'no_performance_claim':True,'late_system_memory_observations_checked':True,'no_uncompressed_residency_claim':True}
(w/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS independent oracle/native/lifecycle/source/memory audit')
