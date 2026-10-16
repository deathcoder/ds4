from pathlib import Path
import json,re
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());a=read(w/'validation-acceptance.json');b=read(w/'candidate-build.json');cap=read(w/'capacity-analysis.json');probe=read(w/'capacity-probe.json');captures=read(w/'capture-progress.json');obs=[json.loads(x) for x in (w/'host-observations.jsonl').read_text().splitlines()]
assert a['passed'] and read(w/'independent-audit.json')['passed'] and read(w/'final-model-verification.json')['passed']
rows='\n'.join(f"| {c['name']} | pass | {c['seconds']:.3f} | {c['peak_rss_bytes']/2**30:.3f} |" for c in a['checks'])
oracles='\n'.join(f"| {c['context']} | {c['repetition']} | {129 if c['context']==262144 else 1} | {c['seconds']:.3f} | {c['process_peak_bytes']/2**30:.3f} |" for c in captures)
history='\n'.join(f"| {r['tokens']} | {r['total_history_bytes']/2**30:.3f} | {r['even_indexer_scores_full_chunk_bytes']/2**30:.3f} | {r['topk_scratch_upper_bytes']/2**30:.3f} |" for r in cap['history_and_selector_budgets'])
thermal=[x for x in obs if x.get('thermal_exit')!=0 or any(line and not line.startswith('Note: No ') for line in x.get('thermal','').splitlines())]
swaps=[float(m[1]) for x in obs if (m:=re.search(r'used = ([0-9.]+)M',x.get('swap','')))];fresh=next(c for c in a['checks'] if c['name']=='c0-256k');retained=next(c for c in a['checks'] if c['name']=='retained-256k')
memory=[json.loads(line) for line in (w/'memory-observations.jsonl').read_text().splitlines()]
initial_memory=read(w/'memory-initial-snapshot.json')
initial_occupied=initial_memory['pages']['Pages occupied by compressor']*16384/2**30
initial_stored=initial_memory['pages']['Pages stored in compressor']*16384/2**30
memory_rows=[f'| Fresh 256K (initial ad hoc snapshot) | 1 | {initial_occupied:.3f} | {initial_stored:.3f} |']
for phase,label in [('START c0-256k','Fresh 256K (late samples only)'),('START retained-128k','Retained 128K'),('START retained-256k','Retained 256K')]:
 samples=[x for x in memory if x['phase']==phase];assert samples
 occupied=max(x['pages']['Pages occupied by compressor']*x['page_size_bytes']/2**30 for x in samples)
 stored=max(x['pages']['Pages stored in compressor']*x['page_size_bytes']/2**30 for x in samples)
 memory_rows.append(f'| {label} | {len(samples)} | {occupied:.3f} | {stored:.3f} |')
memory_table='\n'.join(memory_rows)
text=f'''# Rust Star: 256K capacity and correctness

Candidate `{b['commit'][:8]}` passes full-output C0 at 262,144 prompt tokens plus 128 decode steps against unchanged oracle-v5. Two fresh oracle traces agree on every FP32 word. Rust passes one fresh request and two full-output requests in a second, retained process: three exact native traces across two processes. Request reset and model/context/residency reuse pass. This completes the bounded capacity/correctness milestone; it does not establish 256K throughput or a Rust-versus-DwarfStar speedup.

## Memory evidence

Fresh 256K full-output validation peaks at {fresh['peak_rss_bytes']/2**30:.3f} GiB process RSS. The two-cycle retained diagnostic, which explicitly warms model pages once, peaks at {retained['peak_rss_bytes']/2**30:.3f} GiB for the complete process. That is a process high-water mark, not a separate peak for each cycle. The fresh diagnostic does not explicitly warm all mapped model pages, so these two process peaks are not directly comparable. RSS is the measured process statistic here; total device/driver memory is not independently measured. Both are diagnostic observations and do not substitute for the fresh warmed timing protocol. Current prefill FullKv history still grows with context; no memory reduction or sliding-window storage change is claimed.

An initial ad hoc `vm_stat` snapshot during the late fresh 256K run records {initial_occupied:.3f} GiB occupied by the system compressor, representing {initial_stored:.3f} GiB of uncompressed pages. Its exact capture timestamp was not recorded; the artifact preserves the bounding observation times. A separate timed `vm_stat` observer began at {memory[0]['utc']} and continued through the remaining runs. These are system-wide sampled counters, not process-attributed allocations, and no earlier baseline was collected. Compressor storage and its uncompressed page equivalent are shown separately; neither should be added to process RSS. The successful capacity result includes the host's memory-compression behavior and does not establish a fully resident, uncompressed working set.

| Active gate | Samples | Maximum compressor storage GiB | Maximum stored-page equivalent GiB |
|---|---:|---:|---:|
{memory_table}

The source budget below separates retained history from selected attention scratch. Q/HC/normalization/split activations stay bounded to 4K chunks. Top-k scratch is an upper bound derived from its work width; not every maximum necessarily coexists.

| Prompt tokens | Modeled retained history GiB | Full-chunk scores GiB | Top-k scratch upper bound GiB |
|---|---:|---:|---:|
{history}

The allocation/fill preflight touches {probe['total_requested_bytes']/2**30:.3f} GiB of simultaneous modeled history and selected transients. It excludes the model, other scratch and driver overhead. The earlier 128K maximum plus modeled history/selector growth gives a planning estimate of {cap['planning_estimate_128k_peak_plus_history_and_selector_delta_bytes']/2**30:.3f} GiB. Neither calculation is a runtime upper bound or proof of full residency. Device recommended working set is {probe['recommended_max_working_set_size']/2**30:.3f} GiB, maximum buffer length {probe['max_buffer_length']/2**30:.3f} GiB and physical memory {probe['physical_memory_bytes']/2**30:.0f} GiB. Actual observations above determine this milestone's capacity evidence.

## Repeated oracle outputs

The same pinned corpus extends the accepted 128K prompt. Two 192K prefill-only captures establish an intermediate control. Two 256K captures each preserve all 129 full-vocabulary rows: prefill and 128 decode steps. Every row is finite, its lowest-ID greedy argmax is independently recomputed, and no selected token is EOS. Both repetitions' full FP32 vectors and per-row hashes remain in the package; bulk JSON dumps remain in the local work directory.

| Prompt tokens | Repeat | Full rows | Diagnostic process seconds | Peak RSS GiB |
|---|---|---:|---:|---:|
{oracles}

Oracle source is `ccbd16d36895ef9fe8edf7f793014988ff1ab7f4`, executable SHA-256 `a4056d7297bd0a6de8e46fac294e7913d9bc8c96465e5d68f1b6ef6471e49aa0`. Model SHA-256 is `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`. Final model, prompt, oracle source/executable and CLI hashes match their preflight pins. Hardware, macOS build, Xcode and clang match the accepted 128K environment; the detailed host manifest was collected during the first capture, while thermal/swap observation began before capture.

## Native controls and retained lifecycle

All 318 release Rust tests, 94 Python toolkit tests, release build, formatting and CLI checks pass. Eleven sequential live gates pass with no failed attempts or retries. Each complete output trace checks 16,677,120 FP32 words. The intermediate 192K gate checks prefill only and makes no decode claim. Internal HC acceptance remains limited to 24K; fresh complete 32K output/decode checks stay mandatory. The strict 32K profile retains 287 ordered tail packets and 2,583 component intervals.

| Gate | Result | Diagnostic process seconds | Peak RSS GiB |
|---|---|---:|---:|
{rows}

Both retained contexts require all 129 reference rows for each of two cycles. The implementation warms mapped model pages once, runs complete prefill and full-logit decode, resets request state, and runs again in the same model mapping and Metal context. Actual residency metadata must show establishment for the first cycle and reuse for the second. Both cycles' complete C0 receipts and every decode position/token log are retained. This proves the tested repeated-request lifecycle, not arbitrary-prompt serving or cross-request KV prefix reuse.

## Implementation and scope

External reference parsing and the native boundary/chunk/output/adoption guards now admit aligned frontiers through 256K. The 128K and 256K endpoints require complete 129-row references. Token bounds, finite outputs, greedy selection, immutable 32K prompt prefix, complete layer-42 history and all 43 layer states remain checked. Decoder capacity is prompt plus 128.

`long-prefill-reference-probe ... --retained` adds two full-output diagnostic cycles. Normal engine timing commands and their existing 2K/32K/128K fixtures remain unchanged. The retained 8×64 indexer score kernel, all imported Metal kernels, arithmetic order and RoPE training constants are unchanged. `source-audit.md`, `candidate-source/` and `implementation.patch` preserve the allocation/offset reasoning and exact source state.

There are {len(obs)} host observations, with {len(thermal)} unusual thermal-query observations and swap use from {min(swaps):.2f} to {max(swaps):.2f} MiB. This was an interactive host, and host-side development/build work overlapped the ineligible oracle diagnostics. Model jobs ran one at a time. These process durations and memory observations describe this environment; no paired throughput comparison was run.

Candidate executable SHA-256: `{b['executable_sha256']}`. All references and native outputs are independently audited before final input verification. Run `python3 verify.py` without the model to verify the complete artifact inventory, repeated row hashes/argmaxes, native C0 receipts, full decode transcripts, retained lifecycle, source hashes, profile intervals and process memory receipts. Compiled binaries and the model are excluded. Redacted local paths retain original/stored digests in `redactions.json`.

Stop at this milestone. A 256K paired performance comparison and the coding-agent workload corpus remain future work.
'''
(w/'report-README.md').write_text(text)
