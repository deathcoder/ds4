from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;root=w.parents[2];d=root/'rust-star/reports/attention-output-half-input-038dded4'
assert json.loads((w/'completion-pins.json').read_text())['passed'];d.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest();red={}
def copy(src,n):
 raw=src.read_bytes();s=raw.decode().replace(str(root),'<repo>').replace('/Users/deathcodevision','<user>');out=s.encode();target=d/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(out);red[n]=dict(original_sha256=sha(raw),stored_sha256=sha(out),changed=raw!=out)
main=['plan.json','source-audit.json','prepare.py','arguments.h','candidate-instantiation.metal','combined.metal','production-encoding.m','production-arguments.m','probe.m','build.log','edge-audit.py','edge-freeze.json','edge-execution-freeze.json','run-edges.py','edge-audit.json','toolchain.txt','hardware.json','run.py','audit.py','audit.json','freeze.json','completion-pins.json','verify.py']
for n in main:copy(w/n,n)
for folder,names in [('edges',['probe.jsonl','status.json','run.log']),('session-0',['probe.jsonl','status.json','run.log','preflight.json','final-pins.json','host-observations.jsonl']),('session-1',['probe.jsonl','status.json','run.log','preflight.json','final-pins.json','host-observations.jsonl'])]:
 for n in names:copy(w/folder/n,folder+'/'+n)
for n in json.loads((w/'source-audit.json').read_text())['sources']:copy(root/n,'source-inputs/'+n)
copy(root/'rust-star/reports/compressed-aa-protocol-a6bf9e24/audit.json','parent-protocol-audit.json')
(d/'binary-identity.json').write_text(json.dumps(dict(sha256=sha((w/'probe').read_bytes()),bytes=(w/'probe').stat().st_size,omitted=True),indent=2)+'\n')
(d/'redactions.json').write_text(json.dumps(red,indent=2)+'\n')
print(d)
