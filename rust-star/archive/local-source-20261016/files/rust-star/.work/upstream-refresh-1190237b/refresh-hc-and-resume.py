from pathlib import Path
import json,subprocess,time,shutil
w=Path(__file__).resolve().parent;root=w.parents[2]
while not (w/'hc-v5-acceptance.json').exists():
 if 'Traceback (most recent call last)' in (w/'capture-hc-v5.log').read_text():raise SystemExit('HC capture failed; preserve evidence')
 time.sleep(30)
assert json.loads((w/'hc-v5-acceptance.json').read_text())['passed']
subprocess.run(['python3',str(w/'import-hc-v5.py')],cwd=root,check=True)
old=w/'old-hc-attempt-9a519a36';old.mkdir()
for p in list(w.glob('batch-*'))+[w/'build-and-validate.log',w/'finish-milestone.log']:
 if p.exists():shutil.move(str(p),old/p.name)
p=root/'RUST_STAR_JOURNAL.md';s=p.read_text();pos=s.index('## 2026-09-07');s=s[:pos]+'''## 2026-09-07 — Repeated v5 internal targets accepted through24K

- Two fresh pinned-reference captures at each12K/16K/20K/24K agree for all43
  complete4096-row HC tensors and full frontier logits.16K layer34 exactly
  reproduces Rust;12K remains identical to historical targets. Separate new
  `long-prefill-chunks-v5` inventories and provenance preserve old v3 files.
- Runtime later-chunk expected checksums now come from those independently
  captured v5 tensors. No arithmetic or selection code changes in this step.
- Original9a519a36 successful full-output run and stopped historical-HC run
  archived under `old-hc-attempt-9a519a36/` in refresh W. Rebuild and complete
  six-gate queue will run again against the new frozen candidate before timing.

'''+s[pos:]
start=s.index('- Active authorized milestone after upstream review:');end=s.index('- Completed milestone:',start);s=s[:start]+'''- Active upstream-refresh milestone: pinned c0a6119f with oracle-v5 ccbd16d.
  Reference official gates, four2K/32K repetitions,8K control, tokenizer proof
  and two full129-row traces pass. Rust selector port passes full32K C0.
  Historical16K layer34 mismatch reproduced directly by v5; two full HC
  inventories at each12K/16K/20K/24K now accepted as separate v5 targets.
  Rebuilding candidate with these targets, then repeating all six live gates.
  Paired baseline and portable report remain pending. Evidence and sequential
  scripts: `rust-star/.work/upstream-refresh-1190237b/`.
'''+s[end:]
s=s.replace('   differs, and direct v5 capture reproduces Rust. Finish repeated HC captures,\n   version accepted checkpoints, rebuild and repeat all live gates before timing.', '   differs, and repeated direct v5 captures reproduce Rust. New internal v5\n   targets through24K accepted; rebuild and repeat all live gates before timing.')
p.write_text(s)
subprocess.run(['git','add','rust-star/runtime/src/long_prefill_checksums.rs','rust-star/fixtures/long-prefill-chunks-v5','RUST_STAR_JOURNAL.md'],cwd=root,check=True)
subprocess.run(['git','diff','--cached','--stat'],cwd=root,check=True)
subprocess.run(['git','commit','-m','Version repeated oracle-v5 internal prefill checkpoints'],cwd=root,check=True)
print('START new frozen build and complete live gates',flush=True)
with (w/'build-and-validate.log').open('w') as log:subprocess.run(['python3','-u',str(w/'build-and-validate.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
print('All new candidate gates pass; continue frozen paired baseline',flush=True)
with (w/'finish-milestone.log').open('w') as log:subprocess.run(['python3','-u',str(w/'finish-milestone.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
