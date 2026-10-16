from pathlib import Path
import hashlib,json,subprocess,datetime
w=Path(__file__).resolve().parent; root=w.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(p.read_text())
assert read(w/'validation-acceptance.json')['passed']
build=read(w/'candidate-build.json'); pre=read(w/'preflight.json')
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'); o=root/'rust-star/.work/oracle-v5-source'
r=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip(),baseline_executable_sha256=sha(root/'rust-star/.work/prefill-route-order-32k-iq2-bound/candidate-rust-star'))
for k,v in r.items():assert pre[k]==v,k
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
r['candidate_executable_sha256']=sha(w/'candidate-rust-star'); assert r['candidate_executable_sha256']==build['executable_sha256']
r['sources']={p:sha(root/p) for p in build['sources']}; assert r['sources']==build['sources']
r.update(passed=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(w/'final-pins.json').write_text(json.dumps(r,indent=2)+'\n'); print('Final model, prompt, oracle, baseline, candidate and 40 source pins pass')
