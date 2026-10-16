from pathlib import Path
import json,subprocess,sys,time
w=Path(__file__).resolve().parent
while True:
 log=(w/'campaign.log').read_text()
 if 'Traceback (most recent call last)' in log:raise RuntimeError('Campaign failed; preserve all evidence')
 if 'ALL FIVE PAIRS COMPLETE' in log:break
 time.sleep(20)
for name in ['audit-pairs.py','final-input-verification.py']:
 print('START',name,flush=True)
 subprocess.run([sys.executable,str(w/name)],check=True)
(w/'observation-stop').touch()
# Observer samples at 45-second intervals. Its source detects this marker.
while True:
 ps=subprocess.check_output(['ps','-axo','command'],text=True)
 if not any(line.endswith('measure128-writer-fix/observe.py') for line in ps.splitlines()):break
 time.sleep(5)
subprocess.run([sys.executable,str(w/'render-report.py')],check=True)
print('EVIDENCE READY FOR FINAL REVIEW',flush=True)
