from pathlib import Path
import json,datetime
w=Path(__file__).resolve().parent;root=w.parents[2]
r=root/'rust-star/reports/128k-measure-ef37e566'
assert r.exists() and json.loads((r/'independent-audit.json').read_text())['passed']
def replace(path,old,new):
 p=root/path;s=p.read_text();assert old in s,(path,old);p.write_text(s.replace(old,new,1))
replace('rust-star/ENGINE_MEASUREMENT_FORMAT.md',
 '`rust-star-retained-engine-run-v1`. It executes two exact 2K/128 cycles in one',
 '`rust-star-retained-engine-run-v1`. It executes two exact cycles at the selected\n2K, 32K or 128K frontier, each with 128 decode steps, in one')
replace('rust-star/MEASUREMENT_ADAPTER.md',
 'The native producer supports the validated 2K and 32K frontiers. It runs',
 'The native producer supports the validated 2K, 32K and 128K frontiers, each\nwith exactly 128 decode steps. The [128K report](reports/128k-measure-ef37e566/README.md)\nrecords full-logit correctness, retained lifecycle and five paired measurements. It runs')
replace('rust-star/runtime/README.md',
 '`--context 32768 --gen-tokens 128` selects the native long-prefill path for\n`engine-measure` and `engine-retained-measure`. Establish C0 with both the\nnormal and `--output-only` long-prefill gates before interpreting measurements.\nThe 32K path includes prefix rebuilding, history growth, command execution,',
 '`--context 32768 --gen-tokens 128` or `--context 131072 --gen-tokens 128`\nselects the native long-prefill path for `engine-measure` and\n`engine-retained-measure`. Establish C0 with the appropriate independent\nfull-output gate before interpreting measurements: the built-in 32K controls\nand the external-reference 128K probe. Both paths include prefix rebuilding,\nhistory growth, command execution,')
replace('rust-star/runtime/README.md',
 'frontier; existing built-in fixture and engine measurement commands retain their\n32K contract.',
 'frontier; built-in diagnostic fixtures retain their earlier contract. The\nsubsequent measurement milestone also admits the pinned 128K prompt and complete\n129-selection transcript into fresh and retained engine measurement.')
replace('rust-star/runtime/README.md',
 'milestone makes no new throughput or memory-reduction claim. The next milestone\nis 128K measurement and retained-engine protocol support with paired benchmarks.',
 'milestone makes no new throughput or memory-reduction claim. The subsequent\n[128K measurement report](../reports/128k-measure-ef37e566/README.md) records\n317 Rust tests, 94 Python tests, six live gates and five alternating pairs\nagainst unchanged oracle-v5. Timed runs omit full-logit collection and check\nthe exact transcript after timing. The two-cycle retained diagnostic reuses\nmodel mapping/context/residency while rebuilding request state; it is paired\nineligible. Profile controls were exercised at 32K; this report does not\nestablish a validated 128K profile or arbitrary-prompt serving support.')
replace('rust-star/README.md',
 'reference probe is diagnostic; existing benchmark contracts remain 32K and the\nRust-versus-DwarfStar performance baseline below is unchanged.',
 'reference probe is diagnostic. The subsequent\n[128K measurement report](reports/128k-measure-ef37e566/README.md) extends fresh\nand retained timed execution to the pinned 128K/128 transcript. All 317 Rust\ntests, 94 Python tests, six live gates and five alternating oracle-v5 pairs\npass. It adds a 128K performance frontier; the historical 32K comparison below\nremains unchanged.')
replace('rust-star/README.md','The [current native-tail report]', 'The [historical native-tail report]')
print('Contract and runtime documentation updated; journal/charter and result interpretation remain for review.')
