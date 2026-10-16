from pathlib import Path
import hashlib,json
w=Path(__file__).resolve().parent;root=w.parents[2];d=root/'rust-star/reports/32k-attention-1711856e'
assert json.loads((w/'completion-pins.json').read_text())['passed']
d.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest();redactions={}
replacements=[(str(root),'<repo>'),('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf','<model>'),('/Users/deathcodevision','<user>')]
def copy(src,name):
 raw=src.read_bytes();s=raw.decode()
 for a,b in replacements:s=s.replace(a,b)
 out=s.encode();target=d/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(out)
 redactions[name]=dict(original_sha256=sha(raw),stored_sha256=sha(out),changed=raw!=out)
main=['plan.json','preflight.json','preflight.py','toolchain.json','candidate-build.json','freeze.py','source-audit.json','source-audit.py','implementation.patch','tests.log','build.log','validation-acceptance.json','validation-progress.json','validation.log','validate.py','host-observations.jsonl','attention-32k-profiled.log','attention-32k-profiled.json','attention-32k-profiled-components.json','audit.json','final-pins.json','final-pins.py','completion-pins.json','indexed-baseline.m','indexed-diagnostic.m','compressed-baseline.m','compressed-diagnostic.m']
for n in main:copy(w/n,n)
base=['plan.json','arguments.h','source-audit.json','baseline-wrapper.metal','candidate-wrapper.metal','combined.metal','probe.m','build.log','freeze.json','run.py','probe.jsonl','run.log','host-observations.jsonl','status.json','audit.json']
for role,folder,extra in [('candidate','compressed-grid-probe',['prepare.py','final-verification.json']),('null','compressed-grid-probe/aa-control',[])]:
 for n in base+extra:copy(w/folder/n,folder+'/'+n)
 # Audit raw output is unaffected by path redaction; keep original independent checker.
 copy(w/folder/'audit.py',folder+'/audit.py')
source=json.loads((w/'compressed-grid-probe/source-audit.json').read_text())
for n in source['sources']:copy(root/n,'compressed-grid-probe/source-inputs/'+n)
# The main audit authenticates stored files before using source-run digests.
s=(w/'audit.py').read_text();s=s.replace("sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()",'''def sha(p):
 r=json.loads((w/'redactions.json').read_text()).get(str(p.relative_to(w)))
 stored=hashlib.sha256(p.read_bytes()).hexdigest()
 if r:
  assert stored==r['stored_sha256']
  return r['original_sha256']
 return stored''')
(d/'audit.py').write_text(s)
(d/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
print(d)
