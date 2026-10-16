from pathlib import Path
import sys,json,hashlib,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];sys.path.insert(0,str(root/'rust-star'))
from capture_oracle_v1 import collect_host_manifest
current=collect_host_manifest();prior=json.loads((root/'rust-star/reports/128k-score64-b6b0f125/inputs/host-before.json').read_text())
(w/'host-current.json').write_text(json.dumps(current,indent=2)+'\n')
for field in ['macos','machine','system_profiler_filtered']:assert current[field]==prior[field],field
for field in ['uname','xcodebuild','clang']:
 assert current['commands'][field]['exit_code']==0 and current['commands'][field]['output']==prior['commands'][field]['output'],field
for field in ['hw.memsize','hw.ncpu','hw.physicalcpu','hw.logicalcpu','machdep.cpu.brand_string','kern.osrelease','kern.osversion']:assert current['sysctl_allowlist'][field]==prior['sysctl_allowlist'][field],field
(w/'host-provenance.json').write_text(json.dumps({'passed':True,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Hardware, GPU core count, macOS build, uname, Xcode and clang match accepted 128K host. Collected during first oracle capture; continuous thermal/swap observations began before capture.','collector_sha256':hashlib.sha256((root/'rust-star/capture_oracle_v1.py').read_bytes()).hexdigest(),'current_host_sha256':hashlib.sha256((w/'host-current.json').read_bytes()).hexdigest(),'prior_host_sha256':hashlib.sha256((root/'rust-star/reports/128k-score64-b6b0f125/inputs/host-before.json').read_bytes()).hexdigest()},indent=2)+'\n');print('HOST PINS PASS')
