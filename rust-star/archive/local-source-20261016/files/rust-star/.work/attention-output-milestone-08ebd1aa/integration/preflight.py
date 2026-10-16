from pathlib import Path
import hashlib,json,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
model=Path('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'); o=root/'rust-star/.work/oracle-v5-source'
prior=json.loads((root/'rust-star/reports/32k-route-order-45520435/final-pins.json').read_text())
r=dict(model_sha256=sha(model),model_bytes=model.stat().st_size,prompt_sha256=sha(root/'speed-bench/promessi_sposi.txt'),oracle_executable_sha256=sha(o/'ds4-bench'),cli_sha256=sha(o/'ds4'),oracle_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip())
for k,v in r.items():assert prior[k]==v,k
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=o,text=True).strip()
baseline=root/'rust-star/.work/32k-compressed-skip-2f06a2e8/candidate-rust-star'
r['baseline_executable_sha256']=sha(baseline);assert r['baseline_executable_sha256']=='631b4844599d814ebfcae6706d26c0854169d4573feed86391c41880f71cfd99'
r.update(passed=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(w/'preflight.json').write_text(json.dumps(r,indent=2)+'\n'); print('Full model/prompt/oracle/baseline pins pass')
commands={'os':['sw_vers'],'clang':['xcrun','clang','--version'],'rustc':['rustc','--version'],'cargo':['cargo','--version'],'machine':['sysctl','hw.model','hw.memsize','hw.physicalcpu','hw.logicalcpu','machdep.cpu.brand_string']}
toolchain={k:subprocess.check_output(cmd,text=True) for k,cmd in commands.items()};(w/'toolchain.json').write_text(json.dumps(toolchain,indent=2)+'\n')
