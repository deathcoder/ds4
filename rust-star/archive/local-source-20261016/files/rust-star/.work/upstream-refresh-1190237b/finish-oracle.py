from pathlib import Path
import json,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[2];m=w/'oracle-v5-acceptance/manifest.json'
while True:
 a=json.loads(m.read_text());status=a['status']
 if status=='complete':break
 if status in ('failed','interrupted'):raise SystemExit('Oracle capture '+status)
 time.sleep(30)
for args in [ ['python3',str(root/'rust-star/verify_oracle_bundle.py'),str(m.parent),'--json',str(w/'oracle-v5-verification.json')],['python3','-u',str(w/'conformance-v5.py')],['python3',str(w/'import-v5.py')]]:
 print('START',args,flush=True);subprocess.run(args,cwd=root,check=True)
print('ORACLE trace and versioned fixtures complete',flush=True)
