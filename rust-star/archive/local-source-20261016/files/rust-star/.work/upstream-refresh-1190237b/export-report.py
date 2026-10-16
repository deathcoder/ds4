from pathlib import Path
import hashlib,json,shutil,subprocess
source=Path(__file__).resolve().parent;root=source.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
w=Path(read(source/'paired-location.json')['work']);assert read(w/'independent-audit.json')['passed'];assert read(w/'profile32k-analysis.json')['passed'];commit=read(source/'batch-build.json')['commit'];target=root/'rust-star/reports'/('32k-upstream-v5-'+commit[:8]);assert not target.exists();target.mkdir(parents=True)
shutil.copytree(w/'results',target/'results');(target/'inputs').mkdir();proof=target/'correctness';proof.mkdir();derived={}
def copy_portable(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 try:
  text=src.read_text();portable=text.replace(str(root),'$REPO').replace('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf','$MODEL')
  if text!=portable:
   dst.write_text(portable);derived[str(dst.relative_to(target))]={'original_sha256':sha(src),'transformation':'private repository/model paths replaced with placeholders'};return
 except UnicodeDecodeError:pass
 shutil.copyfile(src,dst)
for name in ['independent-audit.json','profile32k.json','profile32k-analysis.json','host-observations.jsonl','load-observations.jsonl']:
 copy_portable(w/name,target/name)
for p in (w/'inputs').glob('*.json'):
 if p.name!='plan.json':copy_portable(p,target/'inputs'/p.name)
plan=read(w/'inputs/plan.json')
for role in ['candidate','oracle']:
 e=plan[role];e['working_directory']='$CANDIDATE_BUILD' if role=='candidate' else '$ORACLE_BUILD';args=e['adapter_command']
 for i,v in enumerate(args):
  if v.startswith(str(root)):args[i]='$REPO/'+v[len(str(root))+1:]
  elif v.startswith('/Users/'):args[i]='$MODEL' if i and args[i-1]=='--model' else '$LOCAL_PATH'
write(target/'inputs/plan-template.json',plan);derived['inputs/plan-template.json']={'original_sha256':sha(w/'inputs/plan.json'),'transformation':'local executable, repository and model paths replaced with placeholders'}
for name in ['toolchain-verification.json','batch-build.json','batch-validation.json','batch-12k.json','batch-32k-output.json','batch-32k-internal.json','batch-2k.json','batch-8k.json','batch-retained.json','python-tests.log','batch-tests-fixed.log','oracle-v5-verification.json','oracle-v5-benchmark-build.json','conformance-v5.json','v4-v5-trace-comparison.json','tokenization-v5.json','eos-token-v5.json','trace-metadata-validation.json','selector-attribution.json','selector-control-change.json','selector-control.patch','selector-control-32768.f32le.bin','selector-control-8192-comparison.json','pristine-repeatability.json','pristine-comparison.json','pristine-build.json','pristine-metal-kernels.log','m1-kernels-corrected.log','benchmark-sampling-test.log','upstream-port-scope.json']:
 copy_portable(source/name,proof/name)
shutil.copytree(source/'oracle-v5-acceptance',proof/'oracle-v5-acceptance')
for name in ['hc-v5-acceptance.json','hc-v5-changes.json']:copy_portable(source/name,proof/name)
shutil.copytree(root/'rust-star/fixtures/long-prefill-chunks-v5',proof/'long-prefill-chunks-v5')
for name in ['batch-validation.json','batch-build.json','batch-32k-internal.log','batch-32k-output.json']:
 copy_portable(source/'old-hc-attempt-9a519a36'/name,proof/'historical-hc-mismatch'/name)
for rep in ['a','b']:copy_portable(source/f'conformance-v5-{rep}/verification.json',proof/f'trace-{rep}.json')
copy_portable(source/'oracle-v5-8192/verification.json',proof/'oracle-v5-8192.json')
for kind in ['prefill-frontier','prefill-decode']:copy_portable(root/f'rust-star/fixtures/{kind}-32768-v5/manifest.json',proof/f'{kind}-fixture.json')
def clone_file(src,dst):
 subprocess.run(['cp','-c',str(src),str(dst)],check=True);return str(dst)
for kind in ['prefill-frontier','prefill-decode']:
 shutil.copytree(root/f'rust-star/fixtures/{kind}-32768-v5',proof/'fixtures'/f'{kind}-32768-v5',copy_function=clone_file)
shutil.copytree(root/'rust-star/fixtures/prefill-indexer-selector-4128-v5',proof/'fixtures/prefill-indexer-selector-4128-v5',copy_function=clone_file)
for name in ['selector-v5-acceptance.json','selector-v5-tie-attribution.json','capture-selector-v5.log']:copy_portable(source/name,proof/name)
for name in ['batch-build.json','batch-validation.json','batch-8k.log','batch-32k-output.json','batch-32k-internal.json']:
 copy_portable(source/'old-selector-attempt-8aa4d13d'/name,proof/'selector-ordering-mismatch'/name)
copy_portable(root/'rust-star/fixtures/prefill-layer23-continuation-second-tile-4128-v1/layer2-indexer-topk-tile2.i32le.bin',proof/'historical-second-tile-topk.i32le.bin')
shutil.copytree(root/'rust-star/fixtures/prefill-indexer-selectors-8192-v5',proof/'fixtures/prefill-indexer-selectors-8192-v5',copy_function=clone_file)
for name in ['batch-build.json','batch-validation.json','batch-8k.log']:
 copy_portable(source/'old-full-selector-attempt-053c5593'/name,proof/'full-selector-ordering-mismatch'/name)
for name in ['batch-build.json','batch-validation.json','batch-8k.log']:
 copy_portable(source/'old-dispatch-attempt-6aaa1192'/name,proof/'remaining-dispatch-mismatch'/name)
previous=root/'rust-star/reports/32k-prefill-dd0d26c5';shutil.copyfile(previous/'results/paired-summary.json',target/'previous-v4-summary.json');shutil.copyfile(previous/'independent-audit.json',target/'previous-v4-audit.json')
write(target/'portability.json',{'schema':'rust-star-report-portability-v1','derived_files':derived,'historical_report':'../32k-prefill-dd0d26c5/','reference_acceptance':'correctness/oracle-v5-acceptance/'})
shutil.copyfile(root/'rust-star/reports/32k-baseline-cbb078bb/verify.py',target/'verify.py')
with (target/'verify.py').open('a') as f:f.write((source/'verify-extension.py').read_text())
for p in target.rglob('*'):
 if p.is_file():
  try:assert '/Users/' not in p.read_text(),str(p)
  except UnicodeDecodeError:pass
write(source/'report-location.json',{'path':str(target)});print(target)
