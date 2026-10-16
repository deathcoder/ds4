from pathlib import Path
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
paired=Path(read(w/'paired-location.json')['work']);assert read(paired/'independent-audit.json')['passed']
base=root/'rust-star/reports/32k-tail-89e32745';commit=read(w/'batch-build.json')['commit'];target=base.with_name('32k-dense-output-'+commit[:8]);assert not target.exists()
def clone(src,dst):subprocess.run(['cp','-c',str(src),str(dst)],check=True);return str(dst)
shutil.copytree(base,target,copy_function=clone)
(target/'README.md').write_text('# Dense output candidate: report assessment pending\n\nThis directory contains newly exported evidence. The candidate has not been accepted as a performance improvement. Report rendering, artifact verification and the retention decision must finish before this becomes a current result. If rendering stopped, preserve all measurements and inspect the failure; do not interpret the copied baseline artifacts as this candidate’s result.\n')
port=read(target/'portability.json');derived=port['derived_files']
for name in ['profile32k.json','profile32k-analysis.json']:
 (target/name).unlink();derived.pop(name,None)
# The baseline reference/fixture and historical-failure evidence is immutable.
# Replace only current candidate, measurement, profile and host records.
for name in ['results','inputs']:shutil.rmtree(target/name)
shutil.copytree(paired/'results',target/'results',copy_function=clone);(target/'inputs').mkdir()
for key in list(derived):
 if key.startswith(('results/','inputs/')):del derived[key]
def portable(src,dst):
 text=src.read_text();new=text.replace(str(root),'$REPO').replace('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf','$MODEL');dst.write_text(new)
 key=str(dst.relative_to(target));derived.pop(key,None)
 if text!=new:derived[key]={'original_sha256':sha(src),'transformation':'private repository/model paths replaced with placeholders'}
for name in ['independent-audit.json','profile-failure.json','profile32k.log','host-observations.jsonl','load-observations.jsonl']:portable(paired/name,target/name)
for p in (paired/'inputs').glob('*.json'):
 if p.name!='plan.json':portable(p,target/'inputs'/p.name)
plan=read(paired/'inputs/plan.json')
for role in ['candidate','oracle']:
 e=plan[role];e['working_directory']='$CANDIDATE_BUILD' if role=='candidate' else '$ORACLE_BUILD'
 for i,v in enumerate(e['adapter_command']):
  if v.startswith(str(root)):e['adapter_command'][i]='$REPO/'+v[len(str(root))+1:]
  elif v.startswith('/Users/'):e['adapter_command'][i]='$MODEL' if i and e['adapter_command'][i-1]=='--model' else '$LOCAL_PATH'
write(target/'inputs/plan-template.json',plan);derived['inputs/plan-template.json']={'original_sha256':sha(paired/'inputs/plan.json'),'transformation':'local paths replaced with placeholders'}
for name in ['batch-build.json','batch-validation.json','batch-12k.json','batch-8k.json','batch-2k.json','batch-32k-output.json','batch-32k-internal.json','batch-retained.json','batch-tests-fixed.log','batch-32k-profiled.json','batch-32k-output.log','batch-32k-profiled.log','components.json','component-audit.json','audit-components.py']:portable(w/name,target/'correctness'/name)
for src,dst in [('results/paired-summary.json','baseline-tail-summary.json'),('independent-audit.json','baseline-tail-audit.json'),('profile32k-analysis.json','baseline-tail-profile.json')]:clone(base/src,target/dst)
patch=subprocess.check_output(['git','diff','9c1d5009',commit,'--','rust-star/runtime'],cwd=root);(target/'implementation.patch').write_bytes(patch)
port.update(derived_files=derived,baseline_report='../'+base.name+'/',reference_evidence_reused_without_change=True);write(target/'portability.json',port)
# Existing verifier recomputes full v5 outputs, candidate gates and paired stats.
# Add the native attention call-count/direct-output contract to profile evidence.
# Update the current candidate contract, preserving historical evidence checks.
p=target/'verify.py';text=p.read_text().replace('len(checks)==6','len(checks)==7').replace('310 passed; 0 failed','312 passed; 0 failed').replace("config['native_tail_dispatches']==25","config['native_tail_dispatches']==31");assert text.count("assert all(pair['valid'] and pair['attempt'] == 1 for pair in raw['pairs'])")==1
text=text.replace("assert all(pair['valid'] and pair['attempt'] == 1 for pair in raw['pairs'])","assert audit['invalid_attempts']==1\nassert all(pair['valid'] and pair['attempt']==(2 if pair['repetition']==4 else 1) for pair in raw['pairs'])\ninvalid=[a for a in state['attempts'] if a['status']=='invalid']\nassert len(invalid)==1\nbad=invalid[0]\nassert (bad['context'],bad['repetition'],bad['attempt'])==(32768,4,1)\nassert bad['invalid_reason'].startswith('Host shutdown/restart interrupted pair 4 attempt 1')\ninterruption=read(ROOT/'inputs/restart-interruption.json')\nsnapshot=ROOT/'inputs/state-before-restart-recovery.json'\nassert digest(snapshot)==interruption['interrupted_state_sha256']\nassert interruption['confirmed_no_benchmark_processes'] and interruption['candidate_and_oracle_binaries_rehashed']\nbefore=read(snapshot)['attempts'][-1]\nassert before['status']=='running' and before['engines']['oracle']['status']=='running'\nassert {k:v for k,v in bad.items() if k not in ('status','invalid_reason','invalidated_at_utc')}=={k:v for k,v in before.items() if k!='status'}\ne=bad['engines']['candidate'];directory=results/e['output'];measurement=directory/'measurement.json'\nassert digest(measurement)==e['measurement_sha256']\nm=read(measurement);assert m['status']=='passed' and m['metrics']==e['metrics']\nfor artifact in m['artifacts'].values():\n file=directory/artifact['path'];assert file.stat().st_size==artifact['bytes'] and digest(file)==artifact['sha256']\n",1)
text=text.replace("raw['pairs']",'valid_pairs')
marker="raw = read(results / 'paired-raw.json')"
assert text.count(marker)==1
text=text.replace(marker,marker+'\n'+"valid_pairs=[pair for pair in raw['pairs'] if pair['valid']]\ninvalid_raw=[pair for pair in raw['pairs'] if not pair['valid']]\nassert len(valid_pairs)==5 and len(invalid_raw)==1\nassert (invalid_raw[0]['context'],invalid_raw[0]['repetition'],invalid_raw[0]['attempt'])==(32768,4,1)\nassert invalid_raw[0]['invalid_reason'].startswith('Host shutdown/restart interrupted pair 4 attempt 1')\n",1)
p.write_text(text)
with (target/'verify.py').open('a') as f:f.write('''\nattention=read(ROOT/'profile32k-analysis.json')['group_totals']['even/attention']\nassert attention['calls']==147\nconfig=raw['candidate']['runtime_configuration']\nassert config['native_tail_scratch_reuse'] and config['native_tail_direct_hc_output']\nassert not config['native_tail_routed_mid_f32_conversion'] and config['native_tail_dispatches']==31\nassert raw['candidate']['runtime_configuration']['odd_prefill_ingress_rows']==4096\nassert raw['candidate']['runtime_configuration']['even_prefill_attention_tile_rows']==4096\nassert raw['oracle']['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'\nassert digest(ROOT/'baseline-tail-summary.json')==read(ROOT/'baseline-tail-audit.json')['summary_sha256']\nassert read(ROOT/'baseline-tail-profile.json')['profile_sha256']==read(ROOT/'baseline-tail-audit.json')['profile_sha256']\nassert read(ROOT/'correctness/ingress-metadata-attempt-05447277/batch-validation.json')[0]['exit']!=0\nprint('PASS: native tail scratch/output contract and unchanged oracle-v5')\n''')
with (target/'verify.py').open('a') as f:f.write("\nassert config['native_tail_scratch_buffers']==20\nassert config['native_output_low_projection']=='eight aligned dense Q8 groups with direct strided stores'\nimport subprocess,sys\nsubprocess.run([sys.executable,str(proof/'audit-components.py')],check=True,stdout=subprocess.DEVNULL)\nassert read(proof/'component-audit.json')['passed']\nassert read(proof/'batch-32k-profiled.json')['complete_32k_c0_claim']\nprint('PASS: direct dense Q8 projection and exact-gated component profile')\n")
with (target/'verify.py').open('a') as f:f.write("\nimport collections\nlp=read(ROOT/'profile32k.json')['long_prefill_profile']\nparents={(r['chunk_start'],r['first_layer']):r for r in lp['rows'] if r['stage']=='tail'}\nparts=collections.defaultdict(list)\nfor r in lp['tail_components']['rows']:parts[r['chunk_start'],r['layer']].append(r)\nassert set(parts)==set(parents)=={(c,l) for c in range(4096,32768,4096) for l in range(2,43)}\nnames=['attention_output','ffn_ingress','router','routed_map','routed_gate_up','routed_down','routed_sum','shared_expert','final_hc']\nfor key,group in parts.items():\n assert [r['component'] for r in group]==names\n assert all(isinstance(r['start_tick'],int) and isinstance(r['end_tick'],int) and r['end_tick']>r['start_tick'] for r in group)\n assert all(b['start_tick']>a['end_tick'] for a,b in zip(group,group[1:]))\n total=sum(r['end_tick']-r['start_tick'] for r in group)\n assert abs(sum(r['attributed_gpu_ms'] for r in group)-parents[key]['gpu_ms'])<1e-6\n for r in group:assert abs(r['attributed_gpu_ms']-parents[key]['gpu_ms']*(r['end_tick']-r['start_tick'])/total)<1e-6\nassert read(ROOT/'profile32k-analysis.json')['tail_component_rows']==2583\nprint('PASS: isolated engine profile component accounting')\n")
p=target/'verify.py';text=p.read_text()
a=text.index("profile = read(ROOT / 'profile32k.json')");b=text.index('print(f"PASS:',a)
failure_check="""failure=read(ROOT/'profile-failure.json')
assert audit['profile_status']=='failed' and audit['profile_sha256'] is None
assert not audit['profile_stage_shares_verified']
assert digest(ROOT/'profile-failure.json')==audit['profile_failure_sha256']
assert failure['exit_code']==1 and not failure['profile_artifact_written']
assert digest(ROOT/'profile32k.log')==failure['log_sha256']
assert 'invalid tail timestamps' in (ROOT/'profile32k.log').read_text()
assert not (ROOT/'profile32k.json').exists()
"""
text=text[:a]+failure_check+text[b:]
text=text.replace("assert read(ROOT/'profile32k-analysis.json')['passed']", "assert read(ROOT/'profile-failure.json')['status']=='failed'")
text=text.replace("rows=profile['long_prefill_profile']['rows']", "rows=read(proof/'components.json')['rows']")
old="attention=read(ROOT/'profile32k-analysis.json')['group_totals']['even/attention']\nassert attention['calls']==147"
new="assert sum(r['calls'] for r in read(proof/'components.json')['rows'] if r['stage']=='attention' and r['first_layer']==r['last_layer'] and r['first_layer']%2==0)==147"
assert text.count(old)==4;text=text.replace(old,new)
text=text.replace("lp=read(ROOT/'profile32k.json')['long_prefill_profile']", "lp=read(ROOT/'correctness/components.json')")
text=text.replace("assert read(ROOT/'profile32k-analysis.json')['tail_component_rows']==2583", "assert len(lp['tail_components']['rows'])==2583")
text=text.replace('PASS: isolated engine profile component accounting', 'PASS: premeasurement exact-gated component accounting; isolated profile failure preserved')
compile(text,str(p),'exec');p.write_text(text)
for p in target.rglob('*'):
 if p.is_file():
  try:assert '/Users/' not in p.read_text(),str(p)
  except UnicodeDecodeError:pass
write(w/'report-location.json',{'path':str(target)});print(target)
