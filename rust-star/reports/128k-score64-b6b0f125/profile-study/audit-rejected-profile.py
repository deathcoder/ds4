from pathlib import Path
import json,re,hashlib
w=Path(__file__).resolve().parent
status=json.loads((w/'baseline-profile-status.json').read_text())
assert status['exit']==1 and not (w/'baseline-profile.json').exists()
log=(w/'baseline-profile.log').read_text();assert 'chunk=57344 layer=41' in log
packet=re.search(r'raw_ticks=\[([^]]+)\]',log);assert packet
ticks=[int(v,16) for v in packet[1].split(',')];assert len(ticks)==18
assert all(0<=v<2**64-1 for v in ticks)
assert all(a<b for a,b in zip(ticks[::2],ticks[1::2]))
backward=[i for i in range(1,18) if ticks[i]<=ticks[i-1]];assert backward==[14]
assert ticks[10]<ticks[14]<ticks[15]<ticks[11]<ticks[12]<ticks[13]
control=[json.loads(line) for line in (w/'counter-order-unfenced.jsonl').read_text().splitlines()]
fenced=[json.loads(line) for line in (w/'counter-order-fenced.jsonl').read_text().splitlines()]
assert control[-1]==dict(fenced=False,runs=100,non_increasing_packets=100)
assert [r['run'] for r in control[:-1]]==list(range(100))
for r in control[:-1]:
 t=r['ticks'];assert len(t)==18 and all(0<=v<2**64-1 for v in t)
 assert all(a<b for a,b in zip(t[::2],t[1::2]))
 assert any(t[i]<=t[i-1] for i in range(1,18))
assert fenced==[dict(fenced=True,runs=100,non_increasing_packets=0)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=dict(passed=True,rejected_profile_exit=1,chunk_start=57344,layer=41,raw_ticks=ticks,backward_boundaries=backward,each_component_span_positive=True,shared_expert_span_contained_in_routed_down_span=True,unfenced_microprobe_packets_audited=100,fenced_microprobe_reported_ordered_packets=100,fenced_microprobe_raw_success_packets_emitted=False,interpretation='Independent encoder overlap is supported by positive component spans and the controlled fence experiment. The complete baseline profile was rejected; no attribution or performance claim is accepted from it.',artifacts={n:sha(w/n) for n in ['baseline-profile-status.json','baseline-profile.log','counter-order-probe.m','counter-order-unfenced.jsonl','counter-order-fenced.jsonl']})
(w/'rejected-profile-audit.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
