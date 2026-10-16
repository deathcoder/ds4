from pathlib import Path
import datetime,json,os,subprocess,time
w=Path(__file__).resolve().parent
while True:
 rows=subprocess.check_output(['ps','-axo','pid=,pcpu=,comm='],text=True).splitlines();top=[]
 for row in rows:
  fields=row.split(None,2)
  if len(fields)==3:
   cpu=float(fields[1])
   if cpu>=5:top.append({'process':Path(fields[2]).name,'cpu_percent':cpu})
 record={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_processes':sorted(top,key=lambda x:-x['cpu_percent'])[:15]}
 with (w/'load-observations.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 commands=subprocess.check_output(['ps','-axo','command='],text=True).splitlines()
 if not any(x.endswith('run-milestone.py') and 'Python' in x for x in commands):break
 time.sleep(45)
