from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[3];read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert read(w/'engine/audit.json')['execution_passed']
processes=subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
assert not [s for s in processes if Path(s.strip().split(maxsplit=1)[-1]).name in ['rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe']]
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');o=root/'rust-star/.work/oracle-v5-source';initial=read(w.parent/'preflight.json');build=read(w/'candidate-build.json')
inputs=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip());assert inputs==initial['inputs'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
for n,h in build['sources'].items():assert sha(root/n)==h,n
for n,h in build['files'].items():assert sha(w/n)==h,n
assert sha(w/'candidate-rust-star')==build['executable_sha256'];assert sha(w/'baseline-rust-star')==build['baseline_executable_sha256']==initial['executable_sha256']
r=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=inputs,sources=build['sources'],candidate_executable_sha256=build['executable_sha256'],baseline_executable_sha256=build['baseline_executable_sha256'],candidate_build_sha256=sha(w/'candidate-build.json'));p=w/'completion-pins.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print('Final complete input/source/binary verification passed')
