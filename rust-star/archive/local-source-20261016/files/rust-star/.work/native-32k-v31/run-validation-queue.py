import json, subprocess, time
from pathlib import Path
work=Path(__file__).resolve().parent
report=json.loads((work/'native32k-output-only.json').read_text())
assert report['complete_32k_c0_claim'] is True
records=[]
for name,command in json.loads((work/'validation-queue.json').read_text()):
    print(f'START {name}',flush=True)
    start=time.time()
    with (work/(name+'-queue.log')).open('w') as output:
        result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
    records.append(dict(phase=name,returncode=result.returncode,wall_seconds=time.time()-start))
    (work/'validation-queue-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'END {name} status={result.returncode}',flush=True)
    if result.returncode: raise SystemExit(result.returncode)
