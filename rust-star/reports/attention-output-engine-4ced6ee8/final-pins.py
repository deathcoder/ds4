from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];d=w/'integration'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert read(w/'engine/execution-complete.json')['passed']
initial=read(d/'preflight.json');build=read(d/'candidate-build.json');o=root/'rust-star/.work/oracle-v5-source'
model=Path('$MODEL')
r=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip(),baseline_executable_sha256=sha(d/'baseline-rust-star'))
for k,v in r.items():assert initial[k]==v,k
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
r['candidate_executable_sha256']=sha(d/'candidate-rust-star');assert r['candidate_executable_sha256']==build['executable_sha256']
r['sources']={p:sha(root/p) for p in build['sources']};assert r['sources']==build['sources']
for name,h in read(w/'freeze.json')['files'].items():assert sha(w/name)==h,name
for name,h in read(w/'engine/freeze.json')['files'].items():assert sha(w/name)==h,name
for name,h in read(w/'engine/freeze.json')['adapter_files'].items():assert sha(root/name)==h,name
r.update(passed=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_commit=build['commit'],scope='Exact tested sources and binaries after all execution, before final retention decision')
(w/'completion-pins.json').write_text(json.dumps(r,indent=2)+'\n');print('Full model, prompt, oracle, both engines, all 42 sources and frozen protocols pass final pins')
