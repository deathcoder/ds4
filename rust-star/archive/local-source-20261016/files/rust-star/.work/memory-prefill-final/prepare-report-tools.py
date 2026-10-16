from pathlib import Path
w=Path(__file__).resolve().parent;old=w.with_name('counter-bootstrap-94565861')
s=(old/'package-report.py').read_text().replace('32k-bootstrap-912fae1e', '32k-memory-de6d2c11')
a=s.index("tree(root/'rust-star/reports/32k-dense-output-92ca3ef3'");b=s.index("for directory in ['inputs','results']",a)
s=s[:a]+'''reference=root/'rust-star/reports/32k-dense-output-92ca3ef3/correctness'
clone(reference/'conformance-v5.json',r/'oracle-evidence/conformance-v5.json')
for folder in ['fixtures/prefill-frontier-32768-v5','fixtures/prefill-decode-32768-v5','long-prefill-chunks-v5']:
 tree(reference/folder,r/'oracle-evidence'/folder)
accepted=root/'rust-star/reports/32k-bootstrap-912fae1e'
for name in ['artifact-hashes.json','batch-build.json','independent-audit.json','retention-decision.json']:
 clone(accepted/name,r/'accepted-evidence'/name)
phase=w.with_name('memory-prefill-291eb611')
for name in ['batch-build.json','batch-validation.json','batch-tests-fixed.log','batch-release.log','batch-12k.json','batch-12k.log']:
 clone(phase/name,r/'phase1'/name)
'''+s[b:]
s=s.replace("'counter-probe.json','counter-probe-final.json','counter-probe-final-provenance.json','counter-probe-final-build.log','counter-probe-final-run.log'", "'capacity-probe.json','capacity-provenance.json','capacity-build.log','capacity-probe-run.log','capacity-analysis.json'")
s=s.replace("clone(root/'rust-star/diagnostics/tail_counter_probe.m',r/'tail_counter_probe.m')", "clone(w/'capacity_probe.m',r/'capacity_probe.m')")
s=s.replace("'94565861','912fae1e'", "'912fae1e','de6d2c11'")
(w/'package-report.py').write_text(s)
s=(old/'verify-report.py').read_text().replace('complete portable bootstrap study','portable memory study')
s=s.replace("run(r/'reference-evidence/verify.py');run(r/'accepted-profile-study/verify.py')", "assert read(r/'phase1/batch-validation.json')[0]['exit']==0")
a=s.index("assert base['executable_sha256']==freeze['oracle_sha256']");b=s.index('for n,d in b[',a)
s=s[:a]+"assert base['executable_sha256']==freeze['oracle_sha256']==read(r/'accepted-evidence/batch-build.json')['executable_sha256']\n"+s[b:]
s=s.replace("old=r/'reference-evidence/correctness'", "old=r/'oracle-evidence'")
a=s.index("assert decision['prefill_acceptance_passed']");b=s.index("if (r/'profile32k.json').exists()",a)
s=s[:a]+"""assert decision['milestone_acceptance_passed']==audit['milestone_acceptance_passed']
assert decision['memory_changes_retained'] and audit['milestone_acceptance_passed']
assert decision['oracle_v5_unchanged'] and decision['stop_after_milestone']
assert manifest['accepted_report_manifest_sha256']==sha(r/'accepted-evidence/artifact-hashes.json')
"""+s[b:]
a=s.index("probe=read(r/'counter-probe-final.json')")
s=s[:a]+"""probe=read(r/'capacity-probe.json');prov=read(r/'capacity-provenance.json')
assert sha(r/'capacity_probe.m')==prov['source_sha256'] and sha(r/'capacity-probe.json')==prov['output_sha256']
assert probe['passed'] and probe['context_tokens']==131072 and probe['chunk_rows']==4096
assert sum(x['bytes'] for x in probe['buffers'])==probe['total_requested_bytes']
assert max(x['bytes'] for x in probe['buffers'])==probe['largest_requested_bytes']<=probe['max_buffer_length']
assert read(r/'capacity-analysis.json')['full_128k_inference_verified'] is False
print(f'PASS: {len(actual)} artifacts; paired memory/performance, full 32K C0, allocation preflight')
"""
(w/'verify-report.py').write_text(s)
