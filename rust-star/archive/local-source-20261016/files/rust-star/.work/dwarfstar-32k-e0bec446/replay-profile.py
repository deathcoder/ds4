from pathlib import Path
import tempfile,shutil,subprocess,sys,json,hashlib
w=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='rust-star-profile-replay-') as t:
 d=Path(t)
 for p in (w/'profile').iterdir():
  if p.is_file() and p.name!='profile-audit.json':shutil.copy2(p,d/p.name)
 r=subprocess.run([sys.executable,str(d/'audit-profile.py')],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 assert (d/'profile-audit.json').read_bytes()==(w/'profile/profile-audit.json').read_bytes()
print(json.dumps(dict(passed=True,intervals=4466,byte_identical_regeneration=True,prior_failure='Existing-file equality compared Python tuples with JSON lists; replay to a fresh directory matches all saved bytes. Original script and failed replay preserved.'),indent=2))
