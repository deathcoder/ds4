from pathlib import Path
import json,datetime,re
w=Path(__file__).resolve().parent;root=w.parents[2];read=lambda p:json.loads(p.read_text())
b=read(w/'candidate-build.json');report='rust-star/reports/256k-c0-'+b['commit'][:8];r=root/report
assert read(r/'independent-audit.json')['passed'] and read(r/'final-model-verification.json')['passed']
a=read(r/'validation-acceptance.json');assert a['passed'] and len(a['checks'])==11
count=len(read(r/'artifact-hashes.json')['artifacts']);fresh=next(x for x in a['checks'] if x['name']=='c0-256k');retained=next(x for x in a['checks'] if x['name']=='retained-256k')
m=[json.loads(x) for x in (w/'memory-observations.jsonl').read_text().splitlines()]
compress=max(x['pages']['Pages occupied by compressor']*x['page_size_bytes']/2**30 for x in m)
host=[json.loads(x) for x in (w/'host-observations.jsonl').read_text().splitlines()]
swap=[]
for row in host:
 match=re.search(r'used = ([0-9.]+)([KMG])',row.get('swap',''))
 if match:swap.append(float(match.group(1))*{'K':1/1024,'M':1,'G':1024}[match.group(2)])
assert swap
swap_max=max(swap)/1024
date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
p=root/'RUST_STAR_PROJECT.md';s=p.read_text();old='''Internal HC remains accepted through 24K. The next bounded milestone is now
authorized: audit 256K capacity, establish repeatable oracle-v5 full outputs,
validate native 256K/128 C0 and retained-session reset/reuse, then report and
stop. Performance pairing and the coding-agent corpus remain future work.'''
new=f'''Internal HC remains accepted through 24K.

The 256K capacity/correctness milestone is complete at {b['commit'][:8]}.
`{report}/README.md` verifies {count} artifacts.
Two fresh oracle-v5 256K/128 traces agree on all words. One fresh native run
and two complete retained-process cycles match all 16,677,120 words per trace;
request reset and model/context/residency reuse pass at both 128K and 256K.
All eleven corrected native gates, 318 Rust tests, 94 Python tests, release
build, decode and full-size handoff GPU regressions and final input hashes pass.
Both rejected candidates remain preserved: c1047756 fails its first 256K decode
word; 0d5f50ff passes fresh 256K and the first retained cycle, then fails the
second handoff with Metal memory exhaustion. Corrections match oracle SIMD-group
selection and, above 128K, transfer one layer per completed handoff command while
preserving model residency. Imported shaders, prefill arithmetic and RoPE are
unchanged.
Fresh 256K peak RSS is {fresh['peak_rss_bytes']/2**30:.3f}GiB; the explicitly warmed
retained process peaks at {retained['peak_rss_bytes']/2**30:.3f}GiB. System memory compression
and swap use occur; process RSS is not total memory and this does not prove uncompressed
residency. No 256K performance claim is established. Existing timing commands
retain their 128K maximum. Stop here; performance pairing and the coding-agent
corpus remain future work.'''
assert old in s;s=s.replace(old,new,1)
old='''in `RUST_STAR_JOURNAL.md`. As of 2026-09-09, native prefill through all 43 layers
and the output head is C0 exact at 128K, with 128 exact full-logit decode steps
in two fresh runs.'''
new=f'''in `RUST_STAR_JOURNAL.md`. As of {date}, native prefill through all 43 layers
and the output head is C0 exact at 256K, with 128 exact full-logit decode steps
in one fresh run and two retained-process cycles.'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
p=root/'rust-star/README.md';s=p.read_text();anchor='The [128K correctness report]';idx=s.index(anchor)
entry=f'''The [256K correctness report](reports/256k-c0-{b['commit'][:8]}/README.md) validates
262,144 prompt tokens plus 128 decode steps against two identical oracle-v5
traces. One fresh native run and two retained-process cycles match every output
word; reset and residency reuse also pass at 128K. All eleven corrected gates,
318 Rust tests, 94 Python tests and the independent artifact audit pass. The
rejected decode and retained-handoff attempts, both corrections and their GPU
regressions are preserved. System
memory compression and swap use occur, so this establishes tested correctness/capacity,
not uncompressed residency or 256K throughput. The accepted 128K performance
result below remains separate.

'''
s=s[:idx]+entry+s[idx:];p.write_text(s)
p=root/'rust-star/runtime/README.md';s=p.read_text();old='''combined with profiling and does not produce a paired timing sample. The 256K
extension is under validation in the current capacity/correctness milestone;
existing performance commands retain their accepted 128K maximum.''';new=f'''combined with profiling and does not produce a paired timing sample. The
[256K correctness report](../reports/256k-c0-{b['commit'][:8]}/README.md) validates one
fresh and two complete retained 256K/128 traces, plus the corresponding 128K
controls. Both the first decode failure and second candidate's retained-handoff memory
failure are preserved. Corrected gathered
decode matches oracle-v5 SIMD-group selection (one through 2048 keys, two
through 4096, four above). Above 128K, handoff uses 43 completed commands,
releasing each layer history while preserving model residency; shorter contexts
retain one command. The new pipelines are cached lazily. All imported
Metal shaders and prefill arithmetic remain unchanged. System memory compression
and swap use occur; this does not prove uncompressed residency or establish 256K throughput.
Existing performance commands retain their accepted 128K maximum.'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();anchor='## 2026-09-10 — Freeze bounded handoff d6c14fa0 and start full revalidation';assert anchor in s
entry=f'''## {date} — Complete bounded 256K capacity and correctness milestone

- Accept runtime candidate {b['commit'][:8]}. Report `{report}/README.md`
  independently verifies {count} artifacts. Two unchanged oracle-v5 traces agree
  on all 129 rows; fresh native 256K and two complete retained cycles each match
  all 16,677,120 words. Retained reset/model/context/residency checks also pass
  at 128K. All eleven corrected native gates pass, including 192K prefill and
  the established shorter/internal/profile controls.
- Preserve rejected c1047756: eight gates and 256K prefill pass, then first
  decode word fails. Corrected decode follows oracle's SIMD-group policy above
  2048 keys. Independent separate-oracle GPU regression passes 294,912 words
  across nine shapes and detects the old dispatch above the boundary. Imported
  shaders and RoPE remain unchanged; no unexplained output drift is accepted.
- Preserve rejected 0d5f50ff: ten gates and the first retained 256K cycle pass,
  but the second-cycle handoff exhausts Metal memory after an exact prefill.
  Above 128K, one layer per completed command now retires source ownership and
  preserves model residency. Production copy/reset regression checks all words
  and untouched tails in two full-size cycles each at 128K and 256K; weak
  ownership, final cleanup and unchanged residency identity pass. Its small
  model allocation is distinct from the separately accepted real-model gates.
  A missing-reference runner setup error is preserved; no model gate started
  until unchanged reference copies were restored and rehashed.
- All 318 release Rust tests, 94 Python tests, release build, formatting/CLI,
  reference/source/lifecycle/profile/memory audit, final input hashes and the
  portable artifact verifier pass. New candidate binary SHA-256:
  {b['executable_sha256']}.
- Fresh 256K process peak RSS is {fresh['peak_rss_bytes']/2**30:.3f}GiB; explicitly warmed
  two-cycle retained 256K peak is {retained['peak_rss_bytes']/2**30:.3f}GiB. Corrected-sequence
  system compressor samples reach {compress:.3f}GiB and swap use reaches
  {swap_max:.3f}GiB. Full sampling starts before
  the first corrected gate from a post-rejection baseline, not a fresh boot;
  the old attempt's late-only coverage is preserved separately. Counters are
  system-wide. Neither process RSS nor capacity estimates establish total
  memory or uncompressed residency. No paired performance run was performed.
- Validation and both observers have exited. Source candidate stays frozen;
  this completion adds evidence/docs locally. Stop at the authorized milestone.
  No push, 256K performance pairs, larger frontier or coding-agent corpus.

'''
s=s.replace(anchor,entry+anchor,1)
a0=s.index('- ACTIVE: authorized 256K capacity/correctness milestone.',s.index('## Current State'));z=s.index('\n\n- COMPLETE:',a0)
s=s[:a0]+f'''- COMPLETE: authorized 256K capacity/correctness milestone. Candidate {b['commit'][:8]}
  passes all eleven corrected gates, 318 Rust/94 Python tests, both GPU
  regressions and independent verification of {count} report artifacts. Fresh
  256K plus two retained cycles are full-output exact; reset/residency reuse
  also passes at 128K. Both rejected candidates remain preserved. Report:
  `{report}/README.md`. Memory compression and swap use occur; no uncompressed-residency
  or 256K performance claim. Final inputs match, all processes have exited.
  Stop here; no push or next milestone is authorized.
'''+s[z:]
a0=s.index('## Immediate Next Actions\n');z=s.index('\n## Entries',a0)
s=s[:a0]+'''## Immediate Next Actions

1. Stop: the authorized 256K capacity/correctness milestone is complete locally.
2. Keep accepted and rejected artifacts immutable. No push, performance pairing,
   larger frontier or coding-agent corpus is authorized by this milestone.
3. A separately authorized next milestone can establish the 256K measurement
   protocol with explicit warm/residency and system-memory evidence.
'''+s[z:]
p.write_text(s)
print('Completion documentation updated for',b['commit'],count,'artifacts')
