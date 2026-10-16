"""Run only the three predeclared 32K checks; persist failures immediately."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import time

w = Path(__file__).resolve().parent
root = w.parents[2]
model = '/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
read = lambda p: json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
build, plan = read(w/'candidate-build.json'), read(w/'plan.json')
assert read(w/'preflight.json')['passed']
assert sha(w/'plan.json') == build['plan_sha256']
assert '318 passed; 0 failed' in (w/'tests.log').read_text()
assert sha(w/'tests.log') == build['tests_log_sha256']
env = {k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))}
queue = [('candidate-32k-output','candidate-rust-star',False),
         ('baseline-32k-profiled','baseline-rust-star',True),
         ('candidate-32k-profiled','candidate-rust-star',True)]
assert [name for name,_,_ in queue] == plan['live_sequence']
assert not (w/'validation-progress.json').exists()
records = []
def observe(log, phase):
    record = dict(utc=utc(), phase=phase, load=list(os.getloadavg()))
    for name, cmd in [('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat'])]:
        p = subprocess.run(cmd,capture_output=True,text=True)
        record[name] = p.stdout.strip(); record[name+'_exit'] = p.returncode
    log.write(json.dumps(record)+'\n'); log.flush()
with (w/'host-observations.jsonl').open('x') as host:
    observe(host,'before')
    for name, executable, profiled in queue:
        expected_sha = build['executable_sha256'] if executable.startswith('candidate') else plan['baseline_executable_sha256']
        assert sha(w/executable) == expected_sha
        for path, h in build['sources'].items():
            assert sha(root/path) == h, path
        processes = subprocess.check_output(['ps','-axo','pid,comm'],text=True).splitlines()
        assert not [r for r in processes if Path(r.strip().split(maxsplit=1)[-1]).name in
                    ['rust-star','candidate-rust-star','baseline-rust-star','ds4','ds4-bench','ds4-server']]
        args = ['/usr/bin/time','-l',str(w/executable),'long-prefill-chunk-probe',model,
                '--context','32768','--output-only','--json',str(w/(name+'.json'))]
        if profiled:
            args += ['--profile-json',str(w/(name+'-components.json'))]
        print('START',name,utc(),flush=True)
        observe(host,name+'/before')
        started = time.monotonic(); started_utc = utc()
        with (w/(name+'.log')).open('x') as log:
            p = subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            (w/'active-run.json').write_text(json.dumps(dict(name=name,pid=p.pid,started_utc=started_utc),indent=2)+'\n')
            next_observation = time.monotonic()+30
            while p.poll() is None:
                time.sleep(1)
                if time.monotonic() >= next_observation:
                    observe(host,name+'/active'); next_observation = time.monotonic()+30
        text = (w/(name+'.log')).read_text()
        rss = re.search(r'(\d+)\s+maximum resident set size',text)
        record = dict(name=name,exit_code=p.returncode,started_utc=started_utc,finished_utc=utc(),
                      elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=int(rss[1]) if rss else None,executable_sha256=expected_sha)
        records.append(record)
        (w/'validation-progress.json').write_text(json.dumps(records,indent=2)+'\n')
        observe(host,name+'/after')
        if p.returncode:
            (w/'failure.json').write_text(json.dumps(record,indent=2)+'\n')
            raise SystemExit(p.returncode)
        assert rss
        x = read(w/(name+'.json'))
        expected = dict(frontier_tokens=32768,checked_logits=129280,checked_decode_steps=128,
                        logits_checksum=2996662110144943749,decode_logits_checksum=1716816788919264169,
                        decode_tokens_checksum=11977321378026793466,selected_token=85166,
                        complete_32k_c0_claim=True,throughput_claim=False,internal_hc_collection=False)
        for key, value in expected.items():
            assert x[key] == value, (name,key,x[key],value)
        assert text.count('all 129280 logits exact, token ') == 128
        print('PASS',name,record['elapsed_seconds'],flush=True)
    observe(host,'complete')
(w/'validation-acceptance.json').write_text(json.dumps(dict(passed=True,runs=records,
    full_fp32_words_per_run=16677120,throughput_claim=False),indent=2)+'\n')
print('ALL THREE 32K C0 RUNS PASS',flush=True)
