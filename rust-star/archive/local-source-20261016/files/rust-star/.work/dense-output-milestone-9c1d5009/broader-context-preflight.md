Read-only preflight while candidate92ca3ef3 is frozen (2026-09-08).
No implementation or additional model process was started for this future work.

The runtime remains a 2K/32K exact benchmark executor. A broader-context milestone
must establish new evidence before raising guards:
- long_prefill.rs run_long_prefill_with_context restricts frontiers12288..32768,
  loads the fixed32768 token fixture, and validates/decode-handoffs at32768/32896.
- metal_shim.m rust_star_metal_begin_long_prefill_chunk rejects start>=32768.
- metal.rs measurement shape/JSON validators only accept2048/32768 and128decode;
  expectedtranscripts and mismatchdiagnostics currently branch on32768.
- capture_oracle_v5.py delegates to the existing general capture helper with
  pinned v5 source/tree and four conformance repetitions. Preserve that oracle;
  new accepted frontiers must bind repeated full outputs, prompt IDs, executable,
  model and configuration. Existing fixtures must remain unchanged.

Proposed next broader-context sequence after measured optimization is resolved:
1. Audit long-prefill capacity growth and scratch/index limits; measure pinned
   oracle128K repeatability/capacity with one model process at a time.
2. Separate prompt/expected-output input from chunk geometry and decode handoff.
   Generalize only to evidence-backed frontiers; keep existing exact controls.
3. Validate full-prefill and128decode outputs, then retained reset/reuse, then
   paired performance. Larger internal HC captures require their own repeated
   oracle acceptance, and cannot be inferred from current24K internal fixtures.

Memory questions to resolve in source before implementation: whether FullKv
history grows with all past rows or only the attention window, and how indexed
attention scratch scales with compressed rows. Do not claim128K/256K/1M capacity
from the current32K peak memory or from arithmetic estimates alone.

Validated future capture CLI plan (dry-run only, no model read/build/execution):
python3 rust-star/capture_oracle_v5.py --model MODEL --contexts 2048,32768,131072
  --conformance-repetitions 4 --skip-performance --dry-run
Saved broader-context-oracle-dry-run.json.
The v5 capture helper requires both 2K/32K controls and all correctness gates;
its full-logit scope is post-prefill only. It does NOT establish 128 subsequent
full decode rows. Extend the separately audited conformance-v5.py closed-loop
trace producer for new 128K decode evidence, preserving its identity and exact
prompt/token sequence. Do not describe a frontier-only capture as full decode C0.

Read-only capacity issue identified in bind_long_prefill_layer:
The executor currently allocates transient buffers using absolute end position,
not the current 4096-token chunk. FullQ is end * 32768 * sizeof(float) for both
odd/even slots; even slots additionally allocate FullQNorm (1024), FullNorm
(4096), InputHc (16384) and AttnSplit (24) columns. This is not KV history.

| Context | One FullQ buffer GiB | Even-slot listed transient buffers GiB |
|---|---:|---:|
| 32768 | 4.000 | 6.628 |
| 131072 | 16.000 | 26.512 |
| 262144 | 32.000 | 53.023 |
| 524288 | 64.000 | 106.047 |
| 1048576 | 128.000 | 212.094 |

These are source allocation sizes, not observed RSS or total requirements.
Aliased prior slot buffers and compressor/KV/model storage may coexist. Before
raising the 32K guard, query the target device maxBufferLength, audit all absolute
index/offset uses and checked byte arithmetic, then either bound transient storage
to the live chunk or reject unsupported capacity before allocating. A naive guard
change can request a 32 GiB FullQ buffer at 256K and 128 GiB at 1M. The existing
32K result does not establish broader-frontier allocation safety. No model run
was started for this calculation.

Concrete offset dependency: the odd continuation Flash path binds
prefillLayer3FullQ at first_position * q_dim * sizeof(float), and checks the
buffer against chunk_end * q_dim. Reducing only the allocation to 4096 rows
would therefore be incorrect. A bounded query layout needs an explicit base
position (or equivalent checked relative offsets) through producer, consumer and
diagnostic paths. The layer-2 batched path already has q_batch_start-relative
addressing in its attention helper, a useful local precedent to inspect.
