from pathlib import Path
import json,os,subprocess,sys,time
w=Path(__file__).resolve().parent;r=w.parents[2];producer=int(sys.argv[1])
while not (w/'measurement-complete.json').exists():
 try:os.kill(producer,0)
 except ProcessLookupError:
  (w/'report-status.json').write_text(json.dumps({'status':'measurement_did_not_complete','remaining':'Inspect finish-measurements.log and paired raw state; preserve all evidence.'},indent=2)+'\n')
  raise SystemExit('Measurement producer exited before completion')
 time.sleep(30)
try:
 for script in ['export-report.py','render-report.py','hash-report.py']:
  print('START',script,flush=True);subprocess.run(['python3',str(w/script)],cwd=r,check=True)
 report=Path(json.loads((w/'report-location.json').read_text())['path'])
 subprocess.run(['python3',str(report/'verify.py')],cwd=r,check=True)
 (w/'report-status.json').write_text(json.dumps({'status':'verified','report':str(report),'remaining':'Review evidence and retention decision; update current journal/docs and commit. Whole project remains active.'},indent=2)+'\n')
 print('REPORT VERIFIED; final review/docs/commit pending',flush=True)
except subprocess.CalledProcessError as e:
 (w/'report-status.json').write_text(json.dumps({'status':'needs_assessment_or_report_repair','failed_command':e.cmd,'exit_code':e.returncode,'remaining':'Inspect paired outcome and finish-report.log. Do not weaken failed retention criteria or rerun measurements to obtain a favorable result.'},indent=2)+'\n')
 raise
