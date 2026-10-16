"""Render preserved negative evidence only after the improvement predicate fails.
Does not change runtime, acceptance criteria, or any measured artifact.
"""
from pathlib import Path
import json
w=Path(__file__).resolve().parent
source=(w/'render-report.py').read_text()
check="assert s['candidate']['prefill_ms']['median']<old['candidate']['prefill_ms']['median'] and latency<old_latency,'Optimization did not improve baseline; retain evidence and reassess'"
replacement="assert not (s['candidate']['prefill_ms']['median']<old['candidate']['prefill_ms']['median'] and latency<old_latency), 'This renderer is only for a failed improvement predicate'"
assert source.count(check)==1
source=source.replace(check,replacement,1)
source=source.replace('the exact latency improvement cannot be attributed solely', 'the observed latency difference cannot be attributed solely')
source=source.replace('Reference-normalized prefill improvement across campaigns:', 'Reference-normalized prefill factor across campaigns:')
namespace={'__file__':str(w/'render-report.py'),'__name__':'__negative_report__'}
exec(compile(source,str(w/'render-report.py'),'exec'),namespace)
p=namespace['root']/'README.md';text=p.read_text()
first,rest=text.split('\n',1)
notice="\n**Outcome: improvement criterion not met.** The candidate does not improve both absolute median prefill and the reference-normalized paired latency ratio against the accepted native-tail baseline. Preserve these exact-gated results as a negative/inconclusive optimization experiment. They do not establish a new accepted speed improvement. Runtime retention/reversion and current documentation must be resolved explicitly; no measurements have been discarded or retried.\n"
p.write_text(first+' — criterion not met\n'+notice+rest)
print('Rendered failed-criterion evidence; rehash/verify and explicit runtime disposition still required.')
