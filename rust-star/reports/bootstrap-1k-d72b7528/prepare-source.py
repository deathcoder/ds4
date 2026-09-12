"""Preparation only: requires a closed successful current-compiler C0 milestone.
Run from the repository after the next experiment's plan has been reviewed/frozen.
"""
from pathlib import Path
import json,hashlib,subprocess,datetime
root=Path.cwd();w=root/'rust-star/.work/bootstrap-1k-current';read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();c0=root/'rust-star/.work/msrv-c0-5834fe56';assert read(c0/'milestone-completion.json')['passed'];profile=read(c0/'profile-audit.json');assert profile['passed'];share=profile['stages']['bootstrap_ingress']['gpu_ms']/sum(x['gpu_ms'] for x in profile['stages'].values());assert share>=.10
plan=read(w/'plan.json');assert plan['frozen_before_edits'];base=read(c0/'completion.json')['sources'];assert all(sha((root/n).read_bytes())==h for n,h in base.items());assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['baseline_commit']
old=read(root/'rust-star/.work/autonomous-prefill-a46dfe8b/wide-bootstrap-router-bound/source-edits.json');ops=json.loads(json.dumps(old))
for n,es in ops.items():
 for e in es:
  a=e['after']
  if n.endswith('metal.rs'):
   a=a.replace('native_wide','native_1k').replace('rows == 4096','rows == 1024').replace('tile_rows != 4096','tile_rows != 1024')
   a=a.replace('            && position_start.checked_add(rows) == Some(prefill_rows);','''            && prefill_rows >= 8192
            && prefill_rows % 4096 == 0
            && position_start >= prefill_rows - 4096
            && position_start % 1024 == 0
            && position_start.checked_add(rows).map_or(false, |end| end <= prefill_rows);''')
  elif n.endswith('long_prefill.rs'):
   a=a.replace('validated lower-context','lower-context').replace('uses a 4K','uses a 1K').replace('{ 4096 } else { 64 }','{ 1024 } else { 64 }')
  else:
   a=a.replace('native_wide_bootstrap','native_1k_bootstrap').replace('rows == 4096u','rows == 1024u')
   a=a.replace('        (uint64_t)position_start + rows == prefill_rows;','''        prefill_rows >= 8192u && prefill_rows%4096u == 0u &&
        position_start >= prefill_rows-4096u && position_start%1024u == 0u &&
        (uint64_t)position_start + rows <= prefill_rows;''')
   a=a.replace('context.prefillLayerChunkStart.unsignedIntValue != position_start','context.prefillLayerChunkStart.unsignedIntValue != prefill_rows-4096u').replace('wide bootstrap requires','1K bootstrap requires')
   a=a.replace('int32_t positions[4096]','int32_t positions[1024]').replace('int32_t compressed_positions[1024]','int32_t compressed_positions[256]').replace('A wide batch','A 1K batch')
  e['after']=a
for n,es in ops.items():
 p=root/n;s=p.read_text();before=s
 for e in es:
  assert s.count(e['before'])==1,(n,e['before'][:80]);s=s.replace(e['before'],e['after'])
 assert before!=s
 (w/('baseline-'+p.name)).write_text(before);p.write_text(s)
# Preserve the pre-format ledger and let the subsequent source audit account for
# Rust 1.74 formatting explicitly; never silently rewrite this ledger afterward.
(w/'source-edits.json').write_text(json.dumps(ops,indent=2)+'\n')
(w/'source-selection.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile_sha256=sha((c0/'profile-audit.json').read_bytes()),bootstrap_share=share,baseline_sources=base,files=list(ops)),indent=2)+'\n')
print('Applied scoped 1K bootstrap edits; formatting/build/source inverse still required')
