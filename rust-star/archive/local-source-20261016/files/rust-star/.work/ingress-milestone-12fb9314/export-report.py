from pathlib import Path
import hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
paired=Path(read(w/'paired-location.json')['work']);assert read(paired/'independent-audit.json')['passed']
base=root/'rust-star/reports/32k-attention-1eb20ddb';commit=read(w/'batch-build.json')['commit'];target=base.with_name('32k-ingress-'+commit[:8]);assert not target.exists()
def clone(src,dst):subprocess.run(['cp','-c',str(src),str(dst)],check=True);return str(dst)
shutil.copytree(base,target,copy_function=clone)
port=read(target/'portability.json');derived=port['derived_files']
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
for name in ['independent-audit.json','profile32k.json','profile32k-analysis.json','host-observations.jsonl','load-observations.jsonl']:portable(paired/name,target/name)
for p in (paired/'inputs').glob('*.json'):
 if p.name!='plan.json':portable(p,target/'inputs'/p.name)
plan=read(paired/'inputs/plan.json')
for role in ['candidate','oracle']:
 e=plan[role];e['working_directory']='$CANDIDATE_BUILD' if role=='candidate' else '$ORACLE_BUILD'
 for i,v in enumerate(e['adapter_command']):
  if v.startswith(str(root)):e['adapter_command'][i]='$REPO/'+v[len(str(root))+1:]
  elif v.startswith('/Users/'):e['adapter_command'][i]='$MODEL' if i and e['adapter_command'][i-1]=='--model' else '$LOCAL_PATH'
write(target/'inputs/plan-template.json',plan);derived['inputs/plan-template.json']={'original_sha256':sha(paired/'inputs/plan.json'),'transformation':'local paths replaced with placeholders'}
for name in ['batch-build.json','batch-validation.json','batch-12k.json','batch-8k.json','batch-2k.json','batch-32k-output.json','batch-32k-internal.json','batch-retained.json','batch-tests-fixed.log']:portable(w/name,target/'correctness'/name)
failure=target/'correctness/ingress-metadata-attempt-05447277';failure.mkdir()
for src in (w/'metadata-attempt-05447277').iterdir():
 if src.suffix in ('.json','.log'):portable(src,failure/src.name)
for src,dst in [('results/paired-summary.json','baseline-attention-summary.json'),('independent-audit.json','baseline-attention-audit.json'),('profile32k-analysis.json','baseline-attention-profile.json')]:clone(base/src,target/dst)
patch=subprocess.check_output(['git','diff','12fb9314',commit,'--','rust-star/runtime'],cwd=root);(target/'implementation.patch').write_bytes(patch)
port.update(derived_files=derived,baseline_report='../'+base.name+'/',reference_evidence_reused_without_change=True);write(target/'portability.json',port)
# Existing verifier recomputes full v5 outputs, candidate gates and paired stats.
# Add the native attention call-count/direct-output contract to profile evidence.
with (target/'verify.py').open('a') as f:f.write('''\nattention=read(ROOT/'profile32k-analysis.json')['group_totals']['even/attention']\nassert attention['calls']==147\nassert raw['candidate']['runtime_configuration']['odd_prefill_ingress_rows']==4096\nassert raw['candidate']['runtime_configuration']['even_prefill_attention_tile_rows']==4096\nassert raw['oracle']['source_commit']=='ccbd16d36895ef9fe8edf7f793014988ff1ab7f4'\nassert digest(ROOT/'baseline-attention-summary.json')==read(ROOT/'baseline-attention-audit.json')['summary_sha256']\nassert read(ROOT/'baseline-attention-profile.json')['profile_sha256']==read(ROOT/'baseline-attention-audit.json')['profile_sha256']\nassert read(ROOT/'correctness/ingress-metadata-attempt-05447277/batch-validation.json')[0]['exit']!=0\nprint('PASS: complete-chunk ingress scheduling, retained metadata failure and unchanged oracle-v5')\n''')
for p in target.rglob('*'):
 if p.is_file():
  try:assert '/Users/' not in p.read_text(),str(p)
  except UnicodeDecodeError:pass
write(w/'report-location.json',{'path':str(target)});print(target)
