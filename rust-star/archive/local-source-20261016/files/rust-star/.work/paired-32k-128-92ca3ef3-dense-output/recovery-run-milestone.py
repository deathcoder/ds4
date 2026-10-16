from pathlib import Path
import datetime,json,os,subprocess,time
w=Path(__file__).resolve().parent;root=w.parents[2];out=w/'results';env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
def capture(name):
 record={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 for key,args in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('uptime',['uptime'])]:
  p=subprocess.run(args,capture_output=True,text=True);record[key]=p.stdout.strip();record[key+'_exit']=p.returncode
 if name.startswith('host-after'):
  record['power_configuration']=subprocess.check_output(['pmset','-g','custom'],text=True).strip()
  displays=json.loads(subprocess.check_output(['system_profiler','SPDisplaysDataType','-json'],text=True))['SPDisplaysDataType']
  record['displays']=[{'chip':d.get('sppci_model'),'cores':d.get('sppci_cores'),'screens':[{'resolution':x.get('_spdisplays_resolution'),'online':x.get('spdisplays_online')} for x in d.get('spdisplays_ndrvs',[])]} for d in displays]
 (w/'inputs'/name).write_text(json.dumps(record,indent=2)+'\n');return record
def checkpoint():
 path=out/'state.json'
 if not path.exists():return 'starting'
 s=json.loads(path.read_text());last=s.get('attempts',[])[-1:] ;return json.dumps({'phase':s['phase'],'warmups':{k:[r['status'] for r in v] for k,v in s['warmups'].items()},'last_pair':[{k:v for k,v in a.items() if k in ['repetition','order','status']}|{'engines':{k:v['status'] for k,v in a['engines'].items()}} for a in last]},sort_keys=True)
capture('host-after-restart-before-pairs.json')
print('Waiting for documented pair-4 retry to complete',flush=True)
while True:
 state=json.loads((out/'state.json').read_text())
 if state['phase']=='paused' and state['attempts'][-1]['repetition']==4 and state['attempts'][-1]['attempt']==2 and state['attempts'][-1]['status']=='passed':break
 if state['attempts'][-1]['status']=='failed':raise SystemExit('Retry failed; preserve and diagnose without another retry')
 record=capture('host-latest.json');record['checkpoint']=json.loads(checkpoint())
 with (w/'host-observations.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 time.sleep(30)
print('Interrupted pair recovered; resume final pair and profile',flush=True)
with (w/'runner.log').open('w') as log:
 p=subprocess.Popen(['python3',str(root/'rust-star/run_paired_benchmark.py'),'run',str(w/'inputs/plan.json'),'--output',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
 previous=None
 while True:
  current=checkpoint()
  if current!=previous:
   print(current,flush=True);previous=current
  record=capture('host-latest.json');record['checkpoint']=json.loads(current) if current!='starting' else current
  with (w/'host-observations.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  try:p.wait(timeout=45);break
  except subprocess.TimeoutExpired:pass
capture('host-after-pairs.json');print('PAIRED RUN EXIT',p.returncode,checkpoint(),flush=True)
if p.returncode:raise SystemExit(p.returncode)
s=json.loads((out/'state.json').read_text());assert s['phase']=='complete'
print('START isolated 32K profile after paired block',flush=True)
model=json.loads((w/'inputs/plan.json').read_text())['candidate']['adapter_command'];model=model[model.index('--model')+1]
with (w/'profile32k.log').open('w') as log:
 r=subprocess.run([str(w/'bin/candidate/rust-star'),'engine-profile',model,'--context','32768','--gen-tokens','128','--json',str(w/'profile32k.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
capture('host-after-profile.json');print('PROFILE EXIT',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)

work=w.with_name('dense-output-milestone-9c1d5009')
for args in [['python3',str(work/'analyze-profile.py'),str(w/'profile32k.json')],['python3',str(w/'audit.py')]]:
 subprocess.run(args,cwd=root,check=True)
(work/'measurement-complete.json').write_text(json.dumps({'passed':True,'work':str(w),'external_event_retry':'pair 4 attempt 1 interrupted by host restart; attempt 2 reruns whole pair','remaining':'Verify report and resolve retention; STOP after current milestone and recap burn period.'},indent=2)+'\n')
subprocess.run(['python3',str(work/'finish-report.py'),str(os.getpid())],cwd=root,check=True)
