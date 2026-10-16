from pathlib import Path
w=Path(__file__).resolve().parent;old=w.with_name('counter-bootstrap-94565861')
s=(old/'prepare-comparison.py').read_text().replace("paired-rust-bootstrap-", "paired-rust-memory-").replace("'94565861'", "'912fae1e'").replace("'94565861^{tree}'", "'912fae1e^{tree}'")
s=s.replace("initial_bootstrap_last_completed_layer=2 if role=='candidate' else 42,initial_bootstrap_output_head=role!='candidate'", "initial_bootstrap_last_completed_layer=2,initial_bootstrap_output_head=False")
s=s.replace("'accepted_restoration_proof_sha256':sha(root/'rust-star/reports/32k-dense-output-92ca3ef3/restored-runtime-check.json')", "'accepted_report_manifest_sha256':sha(root/'rust-star/reports/32k-bootstrap-912fae1e/artifact-manifest.json')")
s=s.replace("'prefill':'Require lower median paired candidate/baseline latency and improvement in at least four of five pairs. Report all per-pair ratios and dispersion.'", "'prefill':'Require median paired candidate/baseline latency <= 1.03. Report every pair and dispersion; no favorable retries.', 'memory':'Require lower median peak RSS and lower RSS in at least four of five pairs. Report all ratios and absolute GiB.'")
s=s.replace("'decode':'Report paired throughput/first-step distributions; do not accept a clear material decode regression masked by prefill improvement.'", "'decode':'Require median paired steady decode throughput >= 0.97; report first-step latency and every pair.'")
(w/'prepare-comparison.py').write_text(s)
s=(old/'audit-comparison.py').read_text().replace("assert freeze['oracle_commit'].startswith('94565861') and freeze['candidate_commit'].startswith('912fae1e')", "assert freeze['oracle_commit'].startswith('912fae1e') and freeze['candidate_commit'].startswith('de6d2c11')")
s=s.replace("(2 if role=='candidate' else 42)", '2').replace("cfg['initial_bootstrap_output_head']==(role=='oracle')", "cfg['initial_bootstrap_output_head'] is False")
s=s.replace("statistics.median(lat)<1 and sum(x<1 for x in lat)>=4", "statistics.median(lat)<=1.03")
s=s.replace("(p/'independent-audit.json').write_text", """memory=[x['candidate']['peak_rss_mb']/x['oracle']['peak_rss_mb'] for x in pairs]
result['paired_peak_rss_ratio']=dist(memory)
result['memory_acceptance_passed']=statistics.median(memory)<1 and sum(x<1 for x in memory)>=4
result['decode_acceptance_passed']=result['paired_decode_throughput_ratio']['median']>=0.97
result['milestone_acceptance_passed']=all(result[k] for k in ['prefill_acceptance_passed','memory_acceptance_passed','decode_acceptance_passed'])
(p/'independent-audit.json').write_text""")
(w/'audit-comparison.py').write_text(s)
s=(old/'follow-through.py').read_text();a=s.index(" source=root/'rust-star/diagnostics/tail_counter_probe.m'");b=s.index(" run([sys.executable,str(w/'prepare-comparison.py')]",a)
s=s[:a]+""" source=w/'capacity_probe.m';exe=w/'capacity-probe';output=w/'capacity-probe.json'
 run([str(exe),str(output)],'capacity-probe-run')
 data=read(output);assert data['passed'] and data['context_tokens']==131072
 write(w/'capacity-provenance.json',dict(source_sha256=sha(source),executable_sha256=sha(exe),output_sha256=sha(output),compiler=subprocess.check_output(['clang','--version'],text=True),build_command=['clang','-fobjc-arc','-O2','-framework','Foundation','-framework','Metal','capacity_probe.m','-o','capacity-probe'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
"""+s[b:]
s=s.replace("  checks=read(w/'batch-validation.json')", "  if not (w/'batch-validation.json').exists():\n   time.sleep(45);continue\n  checks=read(w/'batch-validation.json')")
(w/'follow-through.py').write_text(s)
