from pathlib import Path
import json,datetime
w=Path(__file__).resolve().parent
log=(w/'validation.log').read_text();lines=log.splitlines();phase=next((x[6:] for x in reversed(lines) if x.startswith('START ')),None)
records=json.loads((w/'validation-progress.json').read_text()) if (w/'validation-progress.json').exists() else []
h=json.loads((w/'host-observations.jsonl').read_text().splitlines()[-1]);m=json.loads((w/'memory-observations.jsonl').read_text().splitlines()[-1])
p=w/'checks'/(phase+'.log');gate_log=p.read_text() if p.exists() else '';detail=gate_log.splitlines()[-1] if gate_log else None
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':sum(x['exit']==0 for x in records),'active':phase,'detail':detail,'exact_decode_rows_so_far':gate_log.count('all 129280 logits exact, token '),'model_processes':h['model_processes'],'swap':h['swap'],'compressor_GiB':round(m['pages']['Pages occupied by compressor']*m['page_size_bytes']/2**30,3),'thermal':h['thermal'],'terminal':(w/'validation-acceptance.json').exists() or 'Traceback' in log,'failed':'Traceback' in log}))
