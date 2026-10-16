from pathlib import Path
import json,re
w=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
b=read(w/'candidate-build.json');a=read(w/'validation-acceptance.json');cap=read(w/'capacity-analysis.json');probe=read(w/'capacity-probe.json')
assert a['passed'] and read(w/'independent-audit.json')['passed'] and read(w/'final-model-verification.json')['passed'] and (w/'observation-stop').exists()
def jsonlines(p):return [json.loads(x) for x in p.read_text().splitlines()]
obs=jsonlines(w/'host-observations.jsonl');memory=jsonlines(w/'memory-observations.jsonl');old=w/'rejected-c1047756';oldmem=jsonlines(old/'memory-observations.jsonl');oldobs=jsonlines(old/'host-observations.jsonl')
def swaprange(xs):
 values=[]
 for x in xs:
  m=re.search(r'used = ([0-9.]+)([KMG])',x.get('swap',''))
  if m:values.append(float(m[1])*{'K':1/1024,'M':1,'G':1024}[m[2]])
 assert values;return f'{min(values):.2f}–{max(values):.2f}'
def compressor(xs):return max(x['pages']['Pages occupied by compressor']*x['page_size_bytes']/2**30 for x in xs)
def stored(xs):return max(x['pages']['Pages stored in compressor']*x['page_size_bytes']/2**30 for x in xs)
def thermal(xs):return sum(x.get('thermal_exit')!=0 or any(line and not line.startswith('Note: No ') for line in x.get('thermal','').splitlines()) for x in xs)
rows='\n'.join(f"| {c['name']} | pass | {c['seconds']:.3f} | {c['peak_rss_bytes']/2**30:.3f} |" for c in a['checks'])
oldrows='\n'.join(f"| {c['name']} | {'pass' if c['exit']==0 else 'FAIL: first decode word'} | {c['seconds']:.3f} | {c['peak_rss_bytes']/2**30:.3f} |" for c in read(old/'validation-progress.json'))
oracles='\n'.join(f"| {c['context']} | {c['repetition']} | {129 if c['context']==262144 else 1} | {c['seconds']:.3f} | {c['process_peak_bytes']/2**30:.3f} |" for c in read(w/'capture-progress.json'))
history='\n'.join(f"| {r['tokens']} | {r['total_history_bytes']/2**30:.3f} | {r['even_indexer_scores_full_chunk_bytes']/2**30:.3f} | {r['topk_scratch_upper_bytes']/2**30:.3f} |" for r in cap['history_and_selector_budgets'])
memrows=[]
for phase in dict.fromkeys(x['phase'] for x in memory):
 xs=[x for x in memory if x['phase']==phase]
 memrows.append(f'| {phase} | {len(xs)} | {compressor(xs):.3f} | {stored(xs):.3f} |')
memtable='\n'.join(memrows)
fresh=next(x for x in a['checks'] if x['name']=='c0-256k');retained=next(x for x in a['checks'] if x['name']=='retained-256k')
reg=jsonlines(w/'regression/probe-results.jsonl')[1:-1]
regrows='\n'.join(f"| {x['keys']} | {x['nsg']} | {x['words']} | {x['old_nsg1_different_words']} |" for x in reg)
text=f'''# Rust Star: 256K capacity and correctness

Candidate `{b['commit'][:8]}` passes full-output C0 at 262,144 prompt tokens plus 128 decode steps against pinned oracle-v5. Two fresh oracle traces agree on every FP32 word. Rust completes one fresh request and two complete requests in a second process with request reset and model/context/residency reuse: three exact native traces across two processes. The eleven corrected candidate gates pass. The first candidate failed at its first 256K decode word; that attempt and the corrective evidence are preserved below.

This completes the bounded capacity/correctness milestone on the tested M1 Ultra with 128 GiB. System memory compression and swap use occur, so these results do not prove uncompressed residency or establish 256K throughput. No performance pairs were run.

## Corrected native validation

All 318 release Rust tests, 94 Python toolkit tests, release build, formatting and CLI checks pass. Each complete trace checks 16,677,120 FP32 words: one prefill row and 128 decode rows, each with 129,280 vocabulary entries. The 192K gate checks prefill only. Internal HC acceptance remains capped at 24K; complete 32K output/decode and strict profile controls remain mandatory. The profile retains 287 ordered tail packets and 2,583 component intervals.

| Gate | Result | Diagnostic process seconds | Peak RSS GiB |
|---|---|---:|---:|
{rows}

Both retained contexts execute two complete C0 cycles after warming the mapped model once. The runtime resets request state between cycles and requires the same model mapping and Metal context. Actual residency metadata must show initial establishment followed by reuse. Every decode position/token and both full-output receipts are preserved. This validates the tested request lifecycle; arbitrary-prompt serving and cross-request KV prefix reuse are outside this check.

## Memory observations and limits

The fresh 256K process peaks at {fresh['peak_rss_bytes']/2**30:.3f} GiB RSS, while the explicitly warmed, two-cycle retained 256K process peaks at {retained['peak_rss_bytes']/2**30:.3f} GiB. These are different residency/warmup paths. Process RSS is not total unified-memory usage and a low fresh-process RSS does not demonstrate a memory optimization.

Corrected-attempt system memory sampling starts at {memory[0]['utc']}, before the first native gate, and ends at {memory[-1]['utc']}, after final input verification. This baseline follows the rejected attempt and compilation; it is not a fresh boot. Counters cover the whole system and cannot be attributed solely to the model. The table shows sampled maxima, not continuously measured peaks; the compressor bytes and represented uncompressed pages can peak in different samples. Phases follow the runner's most recent log line, including any interval between gates.

| Recorded phase | Samples | Maximum compressor GiB | Maximum represented pages GiB |
|---|---:|---:|---:|
{memtable}

There are {len(obs)} corrected-attempt host observations, {thermal(obs)} unusual thermal-query observations, and swap use from {swaprange(obs)} MiB. All model processes run sequentially. Host-side development and auditing overlap these conformance diagnostics; their durations are not eligible performance samples.

The context-sized FullKv and compressed histories remain absolute-addressed; Q, HC, norm and split activations remain chunk-bounded. Modeled selector scratch is an upper bound and not every maximum necessarily coexists.

| Prompt tokens | Modeled history GiB | Full-chunk scores GiB | Top-k scratch upper bound GiB |
|---|---:|---:|---:|
{history}

The reused allocation/fill preflight touches {probe['total_requested_bytes']/2**30:.3f} GiB of simultaneous histories and selected transients, excluding model storage, other scratch and driver overhead. The earlier measured 128K peak plus modeled history/selector growth gives a planning estimate of {cap['planning_estimate_128k_peak_plus_history_and_selector_delta_bytes']/2**30:.3f} GiB. Neither is a runtime bound or proof of full residency. Recommended device working set is {probe['recommended_max_working_set_size']/2**30:.3f} GiB; maximum buffer length is {probe['max_buffer_length']/2**30:.3f} GiB; physical memory is {probe['physical_memory_bytes']/2**30:.0f} GiB. The dispatch correction does not change these allocation formulas. The original preflight source pins and artifact provenance remain explicit.

## Rejected candidate and decode correction

Candidate `c1047756` passes eight shorter gates and all 129,280 prefill logits at 256K, then fails at decode position 262144, logit 0: actual `0xc16ca0ca`, expected `0xc16ea2a8`. Reaching this decode comparison establishes that the mandatory full prefill comparison returned successfully in the frozen source; the final result JSON is not written because the function exits on the decode error. Its retained gates never start. The failure, eight successful outputs, process logs, frozen source/build hashes, and partial memory observations are in `rejected-c1047756/`. This history includes one rejected full-model candidate followed by a source correction and complete revalidation.

| Rejected candidate gate | Result | Diagnostic process seconds | Peak RSS GiB |
|---|---|---:|---:|
{oldrows}

Oracle-v5 gathered decode uses 32 split-K workgroups and 32 keys per SIMD group, selecting one SIMD group through 2048 keys, two through 4096, and four above that. Native decode fixed one group. Odd layers at the tested 256K endpoint have 128 raw plus 2048 compressed keys, crossing the first boundary. The correction selects matching pipeline constants, shared-memory length (`3328 × nsg` bytes) and dispatch dimensions. The additional pipelines compile lazily and remain cached. Imported Metal kernels, prefill arithmetic and RoPE constants are unchanged.

The synthetic regression includes the actual native shim factory and separately compiles the unmodified oracle-v5 FlashAttention and reduction shaders. Every output word is finite and bitwise identical across nine shapes (294,912 words). The old one-group dispatch is a negative control that differs above 2048 keys. The 4097-key case checks the policy's next branch; it is not an additional full-model context claim.

| Keys | Correct SIMD groups | Exact candidate words | Old dispatch differing words |
|---|---:|---:|---:|
{regrows}

Regression source, results, hashes and compiler logs are in `regression/`. The initial shared-library check and the standalone wrapper's missing-MIN-macro compilation error are preserved; the wrapper was corrected before the separate oracle check passed. The native and oracle vector-kernel source prefixes agree; their historically different reduction source is compiled separately in this check. No production shader changed.

The rejected attempt's timed system-memory observations cover only {oldmem[0]['utc']} through {oldmem[-1]['utc']}, beginning late in fresh 256K validation. Sampled compressor storage reaches {compressor(oldmem):.3f} GiB representing up to {stored(oldmem):.3f} GiB of pages. An earlier ad hoc snapshot has bounded rather than exact timing. Its full host record includes swap use from {swaprange(oldobs)} MiB and {thermal(oldobs)} unusual thermal-query observations. Those counters are system-wide and have different coverage from the corrected attempt.

## Oracle references and provenance

The four accepted captures were produced before the rejected native attempt and reused unchanged after rehashing for the correction. Two 192K prefill-only captures establish an intermediate control; two 256K captures preserve all 129 full rows. The accepted 128K reference remains unchanged. Every row is finite, its lowest-ID greedy argmax is independently recomputed, and no selected token is EOS. Both repetitions' complete binary vectors and per-row hashes are packaged. Original bulk JSON dumps remain in the local capture directory.

| Prompt tokens | Repeat | Full rows | Diagnostic process seconds | Peak RSS GiB |
|---|---|---:|---:|---:|
{oracles}

Oracle source: `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`. Oracle executable SHA-256: `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`. Model SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`. Model, prompt, oracle source/executable and CLI hashes match before and after corrected validation. Hardware, macOS build, Xcode and clang were reverified before the corrected gate sequence against the accepted 128K environment. `reused-artifact-provenance.json` binds each reused capture/capacity artifact to its original hash.

## Implementation, verification and scope

External-reference parsing and native boundary/chunk/output/adoption guards now admit aligned frontiers through 256K. The 128K and 256K endpoints require complete 129-row references. Token bounds, finite values, greedy selection, the immutable 32K prompt prefix, completed layer-42 history and all 43 layer states remain checked. Decoder capacity is prompt plus 128. `--retained` adds two full-output diagnostic cycles and is mutually exclusive with profiling. Existing engine timing commands and their 2K/32K/128K fixtures remain unchanged.

`source-audit.md`, `candidate-source/` and `implementation.patch` preserve reasoning and exact source state. Candidate executable SHA-256: `{b['executable_sha256']}`. Run `python3 verify.py` without a model to verify the artifact inventory, repeated reference rows/argmaxes, native C0 receipts, every decode position/token, retained lifecycle, regression, rejected attempt, source hashes, profile intervals and memory receipts. Compiled binaries and model weights are excluded. Redacted local paths retain original/stored digests in `redactions.json`.

The accepted 128K performance result remains separate. A 256K performance comparison and the coding-agent workload corpus remain future work. Stop after this local milestone; no push is part of this task.
'''
(w/'report-README.md').write_text(text)
