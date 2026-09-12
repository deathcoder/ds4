from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;root=w.parents[2];prior=root/'rust-star/.work/routed-half-prefill-c7adfb22'
edits=json.loads((w/'protocol-source-edits.json').read_text());proof=json.loads((w/'protocol-derivation.json').read_text())
for n in ['audit.py','audit-edges.py']:
 s=(prior/n).read_text();assert hashlib.sha256(s.encode()).hexdigest()==proof['files'][n]['before_sha256']
 for e in edits[n]:assert s.count(e['before'])==e['count'];s=s.replace(e['before'],e['after'])
 assert s==(w/n).read_text() and hashlib.sha256(s.encode()).hexdigest()==proof['files'][n]['after_sha256']
(w/'audit-derivation.json').write_text(json.dumps({n:proof['files'][n] for n in ['audit.py','audit-edges.py']},indent=2)+'\n')
