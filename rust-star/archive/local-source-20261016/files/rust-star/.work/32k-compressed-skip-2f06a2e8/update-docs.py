from pathlib import Path
import json
w=Path(__file__).resolve().parent;root=w.parents[2]
assert json.loads((w/'final-pins.json').read_text())['passed']
a=json.loads((w/'audit.json').read_text());assert a['selection']['criteria_pass']
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();entry='''## 2026-09-11 — Native run-skip passes full 32K C0 and real-component selection

- Runtime 9d35953f is retained as a correctness-validated candidate. Normal
  candidate, baseline profile and candidate profile all pass full 32K/128 C0:
  50,031,360 FP32 words in total. Fresh process elapsed times are 312.064s,
  296.182s and 276.177s; these are harness observations, not engine speed data.
- Both profiles pass all 287 attention packets (1,883 component intervals)
  and 287 tail packets (2,583 intervals), complete prefix coverage, strict
  counter ordering and exact parent attribution. All 320 release host tests,
  format/build checks and both production Metal specializations pass.
- Compressed-attention time is 29.100284s baseline / 27.131982s candidate:
  raw ratio 0.932361 (6.76% lower), normalized aggregate 0.950530 (4.95% lower),
  and median matched normalized ratio 0.947557. All three prospective criteria
  pass; 126/140 normalized packets improve, MAD 0.023656. The unchanged
  control sum is itself 1.91% lower, and normalized chunk aggregates at starts
  4096/12288 slightly regress by 0.89%/0.24%; do not hide this variation.
- The 1.968302s raw component saving is only 0.828% of baseline recorded GPU
  stage time (237.619650s). The pair is profiled/perturbed, fixed baseline-first
  order, and cannot establish whole-prefill gain or DwarfStar parity. Keep the
  candidate for a later bounded engine decision; do not launch a broad campaign.
- Portable report rust-star/reports/32k-compressed-skip-9d35953f/README.md includes
  raw outputs/profiles, frozen plans and build/input pins, full host restoration,
  exact tested-kernel renaming, source assembly, constants and a strict verifier.
  Full model/prompt/oracle/baseline/candidate and all 41 source pins pass final
  checks. All GPU work is finished; swap stayed 33.25MB with no recorded thermal
  warning. No push or 256K resumption. Only 32K is validated for this new kernel.

''';i=s.index('## 2026-09-11');s=s[:i]+entry+s[i:]
aidx=s.index('- ACTIVE: narrow run-skip integration and three predeclared 32K C0 runs.');end=s.index('\n\n',aidx)
s=s[:aidx]+'''- COMPLETE: native compressed run-skip 9d35953f passes all three full 32K/128
  C0 checks and prospective component selection. Report
  rust-star/reports/32k-compressed-skip-9d35953f/README.md. Retain the candidate;
  raw component time is 6.76% lower / 4.95% lower after control normalization,
  with heterogeneous chunks. Only 0.828% of recorded GPU stage time is saved;
  no accepted engine gain, higher-frontier gate or DwarfStar parity. All GPU
  work is finished; no additional benchmark is queued.'''+s[end:]
aidx=s.index('## Immediate Next Actions\n');end=s.index('\n## Entries',aidx)
s=s[:aidx]+'''## Immediate Next Actions

1. Treat 9d35953f as a full-output 32K/128 C0-validated candidate, with a
   promising real-component diagnostic rather than accepted engine speed.
   Preserve the 1711856e comparator and both immutable executables/reports.
2. Before additional engine timing, weigh the roughly two-second component
   saving against the full-run cost and declare a small unprofiled paired
   protocol and decision criterion. No automatic expanded timing campaign.
   A separate profile-led lead is attention output, the largest measured tail
   component at 35.918s in this baseline; inspect its actual projections and
   memory/dispatch cost before selecting or claiming another optimization.
3. Keep the head-first grid rejected, prior routing evidence scoped correctly,
   and all old failed/interrupted data intact. Do not infer longer-context
   acceptance from this 32K-only check. Keep 256K paused; no push or automatic
   frontier work.
'''+s[end:];p.write_text(s)
p=root/'RUST_STAR_PROJECT.md';s=p.read_text();needle='Current implementation boundaries, evidence, and immediate actions are maintained'
addition='''Native integration of that candidate is now validated at runtime commit
`9d35953f`. `rust-star/reports/32k-compressed-skip-9d35953f/README.md` records
three exact full-output 32K/128 runs (50,031,360 FP32 words), strict complete
component profiles and the original prospective selection. Compressed-attention
attributed time is 29.100284s to 27.131982s: 6.76% lower raw, 4.95% lower after
control normalization, with 126/140 matched normalized packets improving.
The effect is heterogeneous; two chunk aggregates slightly regress. Retain the
Rust-owned native-only kernel as a correctness-validated candidate, preserving
the original prefix/decode/legacy paths and all imported kernels. The roughly
1.97s raw saving is only 0.828% of recorded GPU stage time; fixed-order perturbed
profiles do not establish whole-prefill gain, DwarfStar parity or acceptance at
higher frontiers. All 41 runtime/source and full input pins pass final checks;
all GPU work is finished. Any engine timing needs a fresh bounded plan, and
256K stays paused. Attention output remains a larger measured component for a
separate profile-led investigation; no new optimization there is selected.

''';assert needle in s;s=s.replace(needle,addition+needle,1);p.write_text(s)
print('Journal and charter updated to completed bounded validation')
