from pathlib import Path
w=Path(__file__).resolve().parent
p=w/'render-report.py';s=p.read_text()
s=s.replace("reg=jsonlines(w/'regression/probe-results.jsonl')[1:-1]", "reg=jsonlines(w/'rejected-0d5f50ff/regression/probe-results.jsonl')[1:-1]\nsecond=w/'rejected-0d5f50ff';secondmem=jsonlines(second/'memory-observations.jsonl');secondobs=jsonlines(second/'host-observations.jsonl')\nsecondrows='\\n'.join(f\"| {c['name']} | {'pass' if c['exit']==0 else 'FAIL: second handoff memory'} | {c['seconds']:.3f} | {c['peak_rss_bytes']/2**30:.3f} |\" for c in read(second/'validation-progress.json'))\nhandoff=read(w/'regression/probe-verification.json');handoffrows='\\n'.join(f\"| {x['rows']} | {x['cycle']} | {x['copied_words']:,} | {x['untouched_tail_words']:,} | {x['commands']} |\" for x in handoff['cases'])")
s=s.replace('The first candidate failed at its first 256K decode word; that attempt and the corrective evidence are preserved below.', 'Two rejected candidates and both corrections are preserved: the first failed at its first 256K decode word; the second passed fresh 256K and its first retained cycle, then ran out of Metal memory at the second handoff.')
s=s.replace('This history includes one rejected full-model candidate followed by a source correction and complete revalidation.', 'This is the first of two rejected full-model candidates; each failure required a source correction before complete revalidation.')
s=s.replace('Regression source, results, hashes and compiler logs are in `regression/`.', 'Decode regression source, results, hashes and compiler logs are in `rejected-0d5f50ff/regression/`. The independent audit proves all shim code outside the new handoff remains byte-identical to that tested predecessor, and all shader hashes match.')
anchor='## Oracle references and provenance'
extra='''## Rejected retained handoff and resource correction

Candidate `0d5f50ff` passes ten gates, including fresh 256K and both retained 128K cycles. Its first retained 256K cycle matches all 16,677,120 words. The second cycle passes all prefill words, then fails in the GPU prefill-to-decode blit with `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`. It executes no second-cycle decode step. The required full prefill comparison precedes that handoff in the frozen source. The retained gate is rejected, with no completion JSON. Complete evidence is in `rejected-0d5f50ff/`.

| Second rejected candidate gate | Result | Diagnostic process seconds | Peak RSS GiB |
|---|---|---:|---:|
{secondrows}

Its system-memory record runs from {secondmem[0]['utc']} to {secondmem[-1]['utc']}. Sampled compressor storage reaches {compressor(secondmem):.3f} GiB, representing up to {stored(secondmem):.3f} GiB of pages. Swap use ranges from {swaprange(secondobs)} MiB; {thermal(secondobs)} unusual thermal-query observations occur. These are system-wide samples and cannot be attributed solely to the model.

The previous handoff encodes all 43 layers' source histories and destination state in one command. The second request also has the model residency set attached to its queue. Above 128K, the corrected handoff transfers one layer per command and waits before dropping that history's dictionary ownership and any executor-slot aliases. An iteration-local autorelease pool retires completed command references before the next layer. All 229 copies keep their original ranges, sizes, validation and destination keys. Model/context/residency remain attached and reused; no arithmetic or shaders change. The <=128K schedule stays at one command; longer handoffs require 43 commands and 43 waits. This bounds resources referenced by each handoff command; it does not establish a whole-process memory bound or a throughput improvement.

The production copy/reset regression uses all 43 full-size synthetic layer histories, with GPU-filled position/layer/role/cycle-specific finite FP32 patterns. Every transferred word and untouched destination tail matches in two cycles at each context. Before every new command, weak ownership checks establish that completed source buffers and slot aliases are released. Final prefill cleanup, copy/wait counts and model/residency identity through reset pass.

| Prompt tokens | Cycle | Exact copied words | Untouched tail words | Commands and waits |
|---|---:|---:|---:|---:|
{handoffrows}

These synthetic checks use a small attached model allocation and prove copy/ownership behavior. The separate real-model retained gates establish the tested full-model lifecycle. New handoff regression evidence is in `regression/`. A validation-runner setup error from omitted reference copies is also preserved: no model gate had started, the unchanged canonical references were restored and rehashed, and the same frozen candidate began its first eleven-gate run. It is not an additional model attempt.

'''
assert s.count(anchor)==1;s=s.replace(anchor,extra+anchor)
s=s.replace('regression, rejected attempt, source hashes', 'both regressions, both rejected attempts, source hashes')
p.write_text(s)
p=w/'update-completion-docs.py';s=p.read_text()
s=s.replace('build, nine-shape GPU regression and final input hashes pass. Rejected c1047756\npasses 256K prefill but fails its first decode word; its evidence is preserved.\nThe correction matches oracle SIMD-group selection above 2048 attention keys,\nwithout changing imported shaders, prefill arithmetic or RoPE constants.', '''build, decode and full-size handoff GPU regressions and final input hashes pass.
Both rejected candidates remain preserved: c1047756 fails its first 256K decode
word; 0d5f50ff passes fresh 256K and the first retained cycle, then fails the
second handoff with Metal memory exhaustion. Corrections match oracle SIMD-group
selection and, above 128K, transfer one layer per completed handoff command while
preserving model residency. Imported shaders, prefill arithmetic and RoPE are
unchanged.''')
s=s.replace('rejected first decode attempt and SIMD-group correction are preserved. System', 'rejected decode and retained-handoff attempts, both corrections and their GPU\nregressions are preserved. System')
s=s.replace("The first candidate's decode failure is preserved; corrected gathered", "Both the first decode failure and second candidate's retained-handoff memory\nfailure are preserved. Corrected gathered")
s=s.replace('through 4096, four above). The new pipelines are cached lazily. All imported', 'through 4096, four above). Above 128K, handoff uses 43 completed commands,\nreleasing each layer history while preserving model residency; shorter contexts\nretain one command. The new pipelines are cached lazily. All imported')
s=s.replace("anchor='## 2026-09-10 — Retained 128K reset and reuse pass; final 256K gate active'", "anchor='## 2026-09-10 — Bound long-context handoff resources; full-size copy regression passes'")
s=s.replace('  shaders and RoPE remain unchanged; no unexplained output drift is accepted.', '''  shaders and RoPE remain unchanged; no unexplained output drift is accepted.
- Preserve rejected 0d5f50ff: ten gates and the first retained 256K cycle pass,
  but the second-cycle handoff exhausts Metal memory after an exact prefill.
  Above 128K, one layer per completed command now retires source ownership and
  preserves model residency. Production copy/reset regression checks all words
  and untouched tails in two full-size cycles each at 128K and 256K; weak
  ownership, final cleanup and unchanged residency identity pass. Its small
  model allocation is distinct from the separately accepted real-model gates.
  A missing-reference runner setup error is preserved; no model gate started
  until unchanged reference copies were restored and rehashed.''')
s=s.replace('passes all eleven corrected gates, 318 Rust/94 Python tests, nine-shape GPU\n  regression and independent verification', 'passes all eleven corrected gates, 318 Rust/94 Python tests, both GPU\n  regressions and independent verification')
s=s.replace('also passes at 128K. Rejected c1047756 remains preserved. Report:', 'also passes at 128K. Both rejected candidates remain preserved. Report:')
p.write_text(s)
p=w/'verify-report.py';s=p.read_text().replace('rejected predecessor and separate oracle GPU regression','two rejected predecessors, separate oracle GPU regression and full-size handoff/reset regression');p.write_text(s)
p=w/'source-audit.md';s=p.read_text().replace('the complete corrective patch touches only decode pipeline selection and dispatch.', 'that first corrective patch touched only decode pipeline selection and dispatch.');p.write_text(s)
