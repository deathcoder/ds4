from pathlib import Path
import hashlib,json
source=Path(__file__).resolve().parent
root=Path(json.loads((source/'report-location.json').read_text())['path'])
freeze=json.loads((root/'inputs/freeze.json').read_text())
artifacts={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact-hashes.json' and '__pycache__' not in p.parts}
(root/'artifact-hashes.json').write_text(json.dumps({'schema':'rust-star-report-artifacts-v1','original_private_plan_sha256':freeze['plan_sha256'],'note':'Derived files explicitly recorded in portability.json; raw evidence unchanged.','artifacts':artifacts},indent=2)+'\n')
print(len(artifacts),'hashed artifacts')
