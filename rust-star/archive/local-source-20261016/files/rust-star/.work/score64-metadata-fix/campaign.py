from pathlib import Path
import subprocess,time,os,json,sys,datetime
w=Path(__file__).resolve().parent;root=w.parents[2]
assert json.loads((w/'validation-acceptance.json').read_text())['passed']
subprocess.run([sys.executable,str(w/'prepare-paired.py')],cwd=root,check=True)
p=Path(json.loads((w/'paired-location.json').read_text())['work']);env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
print('START PAIRED CAMPAIGN',p,flush=True)
with (p/'runner.log').open('x') as log:r=subprocess.run([sys.executable,str(root/'rust-star/run_paired_benchmark.py'),'run',str(p/'inputs/plan.json'),'--output',str(p/'results')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
record={'exit':r.returncode,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for key,args in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('power',['pmset','-g','custom']),('uptime',['uptime'])]:record[key]=subprocess.check_output(args,text=True).strip()
(p/'inputs/host-after-pairs.json').write_text(json.dumps(record,indent=2)+'\n')
print('PAIRED EXIT',r.returncode,flush=True)
assert r.returncode==0
assert json.loads((p/'results/state.json').read_text())['phase']=='complete'
print('ALL FIVE PAIRS COMPLETE',flush=True)
