from pathlib import Path
import hashlib,json,subprocess,re,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(x) for x in (w/'probe-results.jsonl').read_text().splitlines()];assert rows[-1]['passed']
cases=[x for x in rows if 'copied_words' in x];assert [(x['rows'],x['cycle']) for x in cases]==[(131072,1),(131072,2),(262144,1),(262144,2)]
for x in cases:
 n=x['rows'];words=43*128*512
 for l in range(2,43):
  ratio=128 if l%2 else 4
  words+=(n//ratio)*512+2*(65536 if l%2 else 8192)
  if ratio==4:words+=(n//4)*128+2*2048
 assert x['copied_words']==words and x['untouched_tail_words']==928000 and x['copies']==229
 assert x['commands']==x['waits']==(43 if n>131072 else 1)
 for k in ['exact','per_command_source_release','residency_preserved','reset_passed']:assert x[k] is True
# Independently establish that extraction preserved the previous exact copy body.
old=subprocess.check_output(['git','show','0d5f50ff:rust-star/runtime/src/metal_shim.m'],cwd=root,text=True)
a=old.index('int rust_star_metal_adopt_prefill_decoder_state(');a=old.index('            id<MTLBuffer> source_raw',a)
b=old.index('\n        [blit endEncoding];\n        const double wall_start',a)
expected=old[a:b];assert expected.endswith('\n        }');expected=expected[:-len('\n        }')]
expected=expected.replace('[blit endEncoding];','').replace('if (layer < 2u) continue;','if (layer < 2u) return 1;').replace('if (ratio != 4u) continue;','if (ratio != 4u) return 1;').replace('copies++;','(*copies)++;').replace('copies += 3u;','*copies += 3u;')
current=(root/'rust-star/runtime/src/metal_shim.m').read_text();a=current.index('static int encode_prefill_layer_handoff(');a=current.index('    id<MTLBuffer> source_raw',a);b=current.index('\n    return 1;\n}',a)
assert re.sub(r'\s+','',current[a:b])==re.sub(r'\s+','',expected)
files=['handoff-probe.m','probe-build.log','probe-results.jsonl','probe-error.log','verify-handoff-probe.py']
result=dict(passed=True,scope=rows[-1]['scope'],cases=cases,all_copy_checks_and_ranges_preserved=True,production_shim_sha256=sha(root/'rust-star/runtime/src/metal_shim.m'),probe_executable_sha256=sha(w/'handoff-probe'),files={n:sha(w/n) for n in files},verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(w/'probe-verification.json').write_text(json.dumps(result,indent=2)+'\n');print('HANDOFF REGRESSION VERIFIED',sum(x['copied_words'] for x in cases),'copied words')
