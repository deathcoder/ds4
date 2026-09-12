from pathlib import Path
import json,hashlib,datetime,subprocess
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text());utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
ref=read(w/'source-reference.json');model=Path('$USER/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf');o=root/'rust-star/.work/oracle-v5-source'
def full_pins():
 assert not [s for s in subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines() if Path(s.strip().split(maxsplit=1)[-1]).name in {'rust-star','current-rust-star','candidate-rust-star','baseline-rust-star','probe','ds4','ds4-bench','ds4-server'}]
 print('Hashing full model with GPU idle',utc(),flush=True)
 inputs=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip());assert inputs==ref['inputs'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
 sources={n:sha(root/n) for n in ref['sources']};assert sources==ref['sources'];assert sha(w/'current-rust-star')==ref['executable_sha256']
 fixture_names=subprocess.check_output(['git','ls-files','rust-star/fixtures'],cwd=root,text=True).splitlines();fixtures={n:sha(root/n) for n in fixture_names}
 assert not subprocess.check_output(['git','diff','--name-only','HEAD','--','rust-star/fixtures'],cwd=root,text=True).strip()
 return dict(passed=True,utc=utc(),inputs=inputs,sources=sources,fixtures=fixtures,executable_sha256=ref['executable_sha256'],model_stat=dict(size=model.stat().st_size,mtime_ns=model.stat().st_mtime_ns))
if __name__=='__main__':
 pins=full_pins();pins['files']={n:sha(w/n) for n in ['plan.json','source-reference.json','expected-output.json','audit-profile.py','profile-audit-derivation.json','run.py','preflight.py']}
 with (w/'freeze.json').open('x') as f:f.write(json.dumps(pins,indent=2)+'\n')
 print('Frozen',len(pins['sources']),'sources and',len(pins['fixtures']),'fixture files',flush=True)
