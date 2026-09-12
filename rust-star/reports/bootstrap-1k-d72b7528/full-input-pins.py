from pathlib import Path
import json,hashlib,subprocess,datetime,sys
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());plan=read(w/'plan.json');build=read(w/'candidate-build.json');ref=plan['source_reference']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sys.argv[1] in ['input-before.json','input-after.json']
assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','probe','router-probe','ds4','ds4-bench','ds4-server'}]
model=Path('$USER/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');o=root/'rust-star/.work/oracle-v5-source';print('Hashing full model with GPU idle',flush=True)
inputs=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip());assert inputs==ref['inputs'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
sources={n:sha(root/n) for n in build['sources']};assert sources==build['sources'];fixtures={n:sha(root/n) for n in ref['fixtures']};assert fixtures==ref['fixtures'];assert sha(w/'candidate-rust-star')==build['executable_sha256'];assert sha(w/'baseline-rust-star')==build['baseline_executable_sha256']==ref['executable_sha256']
for n,h in build['files'].items():assert sha(w/n)==h,n
result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=inputs,sources=sources,fixtures=fixtures,candidate_executable_sha256=build['executable_sha256'],baseline_executable_sha256=build['baseline_executable_sha256'],model_stat=dict(size=model.stat().st_size,mtime_ns=model.stat().st_mtime_ns))
with (w/sys.argv[1]).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print('All complete model, oracle, fixture, source and executable pins pass',flush=True)
