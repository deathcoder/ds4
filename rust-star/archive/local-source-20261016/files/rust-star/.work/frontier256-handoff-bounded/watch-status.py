from pathlib import Path
import json,datetime
w=Path(__file__).resolve().parent
log=(w/'validation.log').read_text();lines=log.splitlines();phase=next((x[6:] for x in reversed(lines) if x.startswith('START ')),None)
records=json.loads((w/'validation-progress.json').read_text()) if (w/'validation-progress.json').exists() else []
h=json.loads((w/'host-observations.jsonl').read_text().splitlines()[-1]);m=json.loads((w/'memory-observations.jsonl').read_text().splitlines()[-1])
p=w/'checks'/(phase+'.log');gate_log=p.read_text() if p.exists() else '';detail=gate_log.splitlines()[-1] if gate_log else None
import re
status={'gate':phase,'passed':sum(x['exit']==0 for x in records),'terminal':(w/'validation-acceptance.json').exists() or 'Traceback' in log}
progress=re.findall(r'prefill through (\d+) tokens',gate_log)
if progress:status['prefill_K']=int(progress[-1])//1024
count=gate_log.count('all 129280 logits exact, token ')
if count:status['exact_decode_rows']=count
if 'Traceback' in log:status['failure']=log[-2000:]
print(json.dumps(status))
