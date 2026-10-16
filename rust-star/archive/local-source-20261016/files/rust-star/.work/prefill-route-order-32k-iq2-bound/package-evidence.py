from pathlib import Path
import hashlib,json,subprocess
w=Path(__file__).resolve().parent; root=w.parents[2]; dest=root/'rust-star/reports/32k-route-order-45520435'; failed=w.with_name('prefill-route-order-32k-653f791c')
assert (w/'audit.json').exists() and (w/'final-pins.json').exists()
dest.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
redactions={}
replacements=[(str(root),'<repo>'),('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf','<model>'),('/Users/deathcodevision','<user>')]
def copy(src,name):
 raw=src.read_bytes(); content=raw.decode()
 for old,new in replacements:content=content.replace(old,new)
 stored=content.encode(); target=dest/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(stored)
 redactions[name]=dict(original_sha256=sha(raw),stored_sha256=sha(stored),changed=raw!=stored)
main=['plan.json','preflight.json','candidate-build.json','source-audit.json','tests.log','build.log','reused-host-tests.json','shader-compile.json','shader-compile.stderr','shader-compile-build.log','compile-shader.m','assembled.metal','baseline-kernel.metal','candidate-shader.metal','implementation.patch','host-observations.jsonl','validation-progress.json','validation-acceptance.json','validation.log','audit.json','final-pins.json','validate.py','final-pins.py']
for name in json.loads((w/'plan.json').read_text())['live_sequence']:
 main.extend([name+'.log',name+'.json'])
main+=['baseline-32k-profiled-components.json','candidate-32k-profiled-components.json']
for name in main:copy(w/name,name)
for name in ['failure.json','plan.json','candidate-build.json','source-audit.json','preflight.json','build.log','tests.log','failed-candidate-shader.metal','candidate-32k-output.log','validation.log','validation-progress.json','host-observations.jsonl','implementation.patch']:copy(failed/name,'rejected/'+name)
(dest/'redactions.json').write_text(json.dumps(redactions,indent=2)+'\n')
script=(w/'audit.py').read_text().replace("sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()",'''def sha(p):
    stored = hashlib.sha256(p.read_bytes()).hexdigest()
    redactions = read('redactions.json')
    record = redactions.get(str(p.relative_to(w)))
    if record:
        assert stored == record['stored_sha256']
        return record['original_sha256']
    return stored''')
(dest/'audit.py').write_text(script)
(dest/'verify.py').write_text((w/'portable-verify.py').read_text())
print(dest)
