"""Portable comparison verification, without model, GPU or compiler."""
from pathlib import Path
import json,hashlib,gzip,tempfile,subprocess,sys,shutil
base=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();manifest=read(base/'manifest.json')
with tempfile.TemporaryDirectory(prefix='rust-star-direct32k-') as t:
 w=Path(t)
 for n,a in manifest['artifacts'].items():
  for name in [n,a['stored']]:assert not Path(name).is_absolute() and '..' not in Path(name).parts
  raw=(base/a['stored']).read_bytes();assert sha(raw)==a['stored_sha256'],n
  data=gzip.decompress(raw) if a['compressed'] else raw;assert len(data)==a['bytes'] and sha(data)==a['sha256'],n
  p=w/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 expected=(w/'audit.json').read_bytes();(w/'audit.json').unlink()
 r=subprocess.run([sys.executable,str(w/'audit.py')],capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert (w/'audit.json').read_bytes()==expected
 r=subprocess.run([sys.executable,str(w/'replay-profile.py')],capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['byte_identical_regeneration']
 # Corruption controls exercise both raw parsers, even with corrected artifact hashes.
 for role,filename in [('A','benchmark.csv'),('B','engine-run.json')]:
  mroot=w/'engine'/('pair-1-'+role)/'measurement';p=mroot/filename;original=p.read_bytes();mp=mroot/'measurement.json';saved=mp.read_bytes();m=read(mp)
  if role=='A':
   import csv,io
   rows=list(csv.DictReader(io.StringIO(original.decode())));rows[0]['sampling_in_interval']='0'
   out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);p.write_text(out.getvalue())
  else:
   x=read(p);x['selection']['oracle_transcript_match']=False;p.write_text(json.dumps(x))
  a=m['artifacts']['csv' if role=='A' else 'engine_run'];a['sha256']=sha(p.read_bytes());a['bytes']=p.stat().st_size;mp.write_text(json.dumps(m))
  r=subprocess.run([sys.executable,str(w/'audit.py')],capture_output=True,text=True);assert r.returncode!=0
  assert ('sampling-inclusive' if role=='A' else 'oracle transcript') in r.stderr,r.stderr
  p.write_bytes(original);mp.write_bytes(saved)
 a=read(w/'audit.json');print(json.dumps(dict(passed=True,artifacts=len(manifest['artifacts']),fresh_engine_processes=a['runs'],timed_pairs=a['timed_pairs'],raw_corruption_controls_rejected=2,profile_intervals_replayed=4466,runtime_changed=False),indent=2))
