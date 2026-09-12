from pathlib import Path
import json,hashlib,subprocess
w=Path(__file__).resolve().parent;root=w.parents[3];p=root/'rust-star/runtime/src/metal_shim.m';s=p.read_text();edits=json.loads((w/'source-edits.json').read_text());extra=[]
for line in [6221,6240,6643,6662]:
 lines=s.splitlines(keepends=True);assert 'threadsPerThreadgroup:MTLSizeMake(rows,1,1)' in lines[line-1];before=''.join(lines[line-7:line]);after=before.replace('threadsPerThreadgroup:MTLSizeMake(rows,1,1)','threadsPerThreadgroup:MTLSizeMake(MIN(rows,256u),1,1)');assert s.count(before)==1;s=s.replace(before,after);extra.append(dict(before=before,after=after))
edits['rust-star/runtime/src/metal_shim.m']+=extra;p.write_text(s);(w/'source-edits.json').write_text(json.dumps(edits,indent=2)+'\n')
base=json.loads((w.parent/'plan.json').read_text())['accepted_pins']['sources']
for n,ops in edits.items():
 t=(root/n).read_text()
 for op in reversed(ops):assert t.count(op['after'])==1;t=t.replace(op['after'],op['before'])
 assert hashlib.sha256(t.encode()).hexdigest()==base[n]
print('Four legal router dispatches; complete three-file inverse proof passes')
