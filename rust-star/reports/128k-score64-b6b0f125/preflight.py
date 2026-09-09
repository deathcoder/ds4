from pathlib import Path
import hashlib,json,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[2]
i=json.loads((root/'rust-star/reports/128k-c0-0eb484ea/inputs.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
model=Path('$MODEL_DIRECTORY/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
o=root/'rust-star/.work/oracle-v5-source'
r={'model_sha256':sha(model),'model_bytes':model.stat().st_size,'prompt_sha256':sha(root/'speed-bench/promessi_sposi.txt'),'oracle_executable_sha256':sha(o/'ds4-bench'),'cli_sha256':sha(o/'ds4'),'oracle_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=o,text=True).strip()}
for k,v in r.items():assert v==i[k],k
assert not subprocess.check_output(['git','status','--porcelain'],cwd=o,text=True).strip()
ref=root/'rust-star/reports/128k-c0-0eb484ea/reference-131072'
manifest=json.loads((ref/'manifest.json').read_text())
for n,info in manifest['artifacts'].items():assert sha(ref/n)==info['sha256'],n
r['reference_manifest_sha256']=sha(ref/'manifest.json')
r['verified_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();r['passed']=True
(w/'preflight.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
