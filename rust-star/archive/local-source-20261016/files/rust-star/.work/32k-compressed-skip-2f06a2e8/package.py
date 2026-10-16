from pathlib import Path
import hashlib,json,gzip
w=Path(__file__).resolve().parent;root=w.parents[2];d=root/'rust-star/reports/32k-compressed-skip-9d35953f'
assert json.loads((w/'final-pins.json').read_text())['passed']
d.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest();redactions={}
replacements=[(str(root),'<repo>'),('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf','<model>'),('/Users/deathcodevision','<user>')]
def copy(src,name):
 raw=src.read_bytes();s=raw.decode()
 for a,b in replacements:s=s.replace(a,b)
 out=s.encode();target=d/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(out)
 redactions[name]=dict(original_sha256=sha(raw),stored_sha256=sha(out),changed=raw!=out)
main=['plan.json','preflight.json','preflight.py','toolchain.json','hardware.json','candidate-build.json','freeze.py','source-audit.json','source-audit.py','implementation.patch','tests.log','build.log','shader-compile.json','shader-compile.stderr','shader-host-build.log','compile-shader.m','assembled.metal','validation-acceptance.json','validation-progress.json','validation.log','validate.py','host-observations.jsonl','audit.json','final-pins.json','final-pins.py','native-pipeline-helper.txt','native-predicate.txt','original-dispatch.txt','candidate-dispatch.txt']
for n in main:copy(w/n,n)
for name in ['candidate-32k-output','baseline-32k-profiled','candidate-32k-profiled']:
 for ext in ['.json','.log']+(['-components.json'] if 'profiled' in name else []):copy(w/(name+ext),name+ext)
copy(root/'rust-star/runtime/src/compressed_prefill.metal','candidate-shader.metal')
copy(root/'rust-star/runtime/build.rs','candidate-build.rs')
copy(root/'rust-star/reports/compressed-skip-runs-fbbea533/candidate-kernel.metal','isolated-candidate.metal')
copy(root/'rust-star/.work/32k-attention-breakdown-ce94f3fb/candidate-build.json','baseline-build.json')
(d/'candidate-host.m.gz').write_bytes(gzip.compress((root/'rust-star/runtime/src/metal_shim.m').read_bytes(),mtime=0))
for n in json.loads((w/'source-audit.json').read_text())['assembly_inputs']:copy(root/n,'source-inputs/'+n)
copy(w/'audit.py','audit-original.py')
s=(w/'audit.py').read_text().replace("sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()",'''def sha(p):
 r=json.loads((w/'redactions.json').read_text()).get(str(p.relative_to(w)))
 stored=hashlib.sha256(p.read_bytes()).hexdigest()
 if r:
  assert stored==r['stored_sha256']
  return r['original_sha256']
 return stored''')
(d/'audit.py').write_text(s)
copy(w/'verify.py','verify.py')
(d/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
print(d)
