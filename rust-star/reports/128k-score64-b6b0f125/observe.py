from pathlib import Path
import subprocess,json,time,datetime
w=Path(__file__).resolve().parent
with (w/'host-observations.jsonl').open('x') as log:
 while not (w/'observation-stop').exists():
  record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  for label,cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('load',['uptime'])]:
   p=subprocess.run(cmd,capture_output=True,text=True);record[label]=p.stdout.strip();record[label+'_exit']=p.returncode
  rows=subprocess.check_output(['ps','-axo','pid=,rss=,pcpu=,comm='],text=True).splitlines();record['model_processes']=[r.strip() for r in rows if r.split()[-1].endswith(('/ds4-bench','/candidate-rust-star','/baseline-rust-star'))]
  log.write(json.dumps(record)+'\n');log.flush();time.sleep(45)
