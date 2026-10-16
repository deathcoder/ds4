from pathlib import Path
import json,hashlib,sys,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def idle():
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','candidate-rust-star','baseline-rust-star','current-rust-star','ds4','ds4-bench','ds4-server','probe'}]
def pins(full=False):
 p=read(w/'plan.json');state=read(w/'final-runtime-state.json');old=read(w/'prior-plan.json')['source_reference'];model=Path(p['model']);assert {'size':model.stat().st_size,'mtime_ns':model.stat().st_mtime_ns}==p['model_stat']
 sources={n:sha(root/n) for n in state['sources']};assert sources==state['sources']
 exes={k:sha(Path(n)) for k,n in p['executables'].items()};assert exes==p['executable_sha256']
 freeze=read(w/'freeze.json');assert all(sha(w/n)==h for n,h in freeze['files'].items())
 result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,executables=exes,freeze_sha256=sha(w/'freeze.json'))
 if full:
  idle();print('Full input hash with GPU idle',flush=True)
  o=root/'rust-star/.work/oracle-v5-source'
  inputs=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(Path(p['prompt'])),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip());assert inputs==old['inputs']
  assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
  fixtures={n:sha(root/n) for n in old['fixtures']};assert fixtures==old['fixtures']
  result.update(inputs=inputs,fixtures=fixtures)
 return result
if __name__=='__main__':
 assert sys.argv[1] in ['input-before.json','input-after.json']
 with (w/sys.argv[1]).open('x') as f:f.write(json.dumps(pins(True),indent=2)+'\n')
 print('All input pins pass',flush=True)
