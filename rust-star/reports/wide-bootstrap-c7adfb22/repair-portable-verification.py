from pathlib import Path
import json,hashlib,datetime,shutil
root=Path.cwd();w=root/'rust-star/reports/wide-bootstrap-c7adfb22';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();m=read(w/'artifacts.json')
def emit(n,raw,origin,redact=False):
 data=raw.replace(str(root).encode(),b'$REPO').replace(b'$USER',b'$USER') if redact else raw;p=w/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);m['files'][n]=dict(origin=origin,original_sha256=sha(raw),stored_sha256=sha(data),original_bytes=len(raw),stored_bytes=len(data),redacted=data!=raw)
for n in ['verify-report.py','portable-audit-profile.py','baseline/portable-audit-profile.py','README.md']:
 p=w/('verification-attempt-1/'+n);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(w/n,p);m['files'][str(p.relative_to(w))]=dict(m['files'][n],origin='archived first packaged version/'+n)
emit('verification-attempt-1/failure.log',(root/'rust-star/.work/wide-bootstrap-c7adfb22-verify-1.log').read_bytes(),'first portable verifier execution',True)
marker="\np=w/'profile-audit.json'";replacement="\nr=json.loads(json.dumps(r))\np=w/'profile-audit.json'"
for prefix in ['', 'baseline/']:
 n=prefix+'portable-audit-profile.py';s=(w/n).read_text();assert s.count(marker)==1;s=s.replace(marker,replacement);emit(n,s.encode(),'derived frozen profile audit: original-hash lookup and JSON tuple/list canonicalization')
n='verify-report.py';s=(w/n).read_text();old="assert s==(w/(prefix+'portable-audit-profile.py')).read_text();subprocess.run";new="s=s.replace(\"\\np=w/'profile-audit.json'\",\"\\nr=json.loads(json.dumps(r))\\np=w/'profile-audit.json'\");assert s==(w/(prefix+'portable-audit-profile.py')).read_text();subprocess.run";assert s.count(old)==1;s=s.replace(old,new);emit(n,s.encode(),'revised verifier: explicitly derive JSON canonicalization before unchanged full-profile comparison')
n='README.md';s=(w/n).read_text();s=s.replace('Both portable profile audits change only hash lookup.', 'Both portable profile audits change hash lookup and canonicalize the recomputed result through JSON before exact comparison, because Python tuples in the ranked tail list serialize as JSON arrays. No measured value or gate changes. The first failed verification and original packaged scripts are retained under verification-attempt-1/.');emit(n,s.encode(),'completed decision report with disclosed portable verifier correction')
info=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),first_failure='Original audit recomputed ranked_tail_components as tuples; stored JSON contains arrays. Direct Python equality fails although all serialized values match.',different_fields_in_both_profiles=['ranked_tail_components'],both_profiles_exact_after_json_roundtrip=True,changes='Only portable audit adapters and their exact derivation check; original frozen audits, source, raw measurements and decisions unchanged.')
emit('portable-verifier-correction.json',(json.dumps(info,indent=2)+'\n').encode(),'verified serialization diagnosis')
emit('repair-portable-verification.py',Path(__file__).read_bytes(),'repair script after first packaged verifier failure',True)
(w/'artifacts.json').write_text(json.dumps(m,indent=2)+'\n');print('Preserved first failure and corrected portable JSON comparison.')
