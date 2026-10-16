from pathlib import Path
import datetime,json,re,subprocess,time
w=Path(__file__).resolve().parent
out=w/'memory-observations.jsonl'
assert not out.exists()
while not (w/'observation-stop').exists():
 p=subprocess.run(['vm_stat'],capture_output=True,text=True)
 fields={k:int(v) for k,v in re.findall(r'^([^\n:]+):\s+(\d+)\.',p.stdout,re.M)}
 size=re.search(r'page size of (\d+) bytes',p.stdout)
 log=(w/'validation.log').read_text() if (w/'validation.log').exists() else ''
 phase=log.splitlines()[-1:] or ['post-rejection baseline before corrected validation']
 record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase[0],'command':['vm_stat'],'exit':p.returncode,'page_size_bytes':int(size[1]) if size else None,'pages':fields,'raw_stdout':p.stdout,'raw_stderr':p.stderr,'scope':'system-wide; post-rejection baseline before corrected native validation; not fresh boot; no per-process attribution'}
 with out.open('a') as f:f.write(json.dumps(record)+'\n')
 time.sleep(45)
