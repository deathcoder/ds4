# Rust Star Paired Benchmark Protocol

Protocol: `rust-star-paired-benchmark-v1`.

This protocol defines how DwarfStar and Rust Star throughput claims are made on
the same M1 Ultra. It separates correctness, closed-loop user-visible
performance, and diagnostic microbenchmarks so a faster internal kernel cannot
be presented as a faster coding-agent decoder without measuring the whole path.

## Eligibility

A default-path Rust Star result is eligible for a headline comparison only when:

1. the oracle version, model SHA-256, prompt SHA-256, source commits, build
   flags, backend configuration, and runtime flags are recorded;
2. the candidate passes the applicable C0 artifacts against that oracle;
3. correctness instrumentation and tensor dumping are disabled during timed
   runs; and
4. both engines use the same physical Mac, power mode, model file, prompt token
   IDs, context frontier, generation length, sampling rule, and EOS behavior.

An experimental C1/C2/C3 path may be timed, but its result must carry that
classification in the title, table, and raw record. It is never merged into the
default C0 distribution.

## Workloads

### Primary: batch-one closed-loop decode

- Backend: Metal.
- Concurrency: one live sequence.
- Sampling: greedy argmax with DwarfStar-compatible EOS exclusion.
- Generation length: 128 committed tokens unless a protocol revision says
  otherwise.
- Weight mode: resident imatrix Q2 with warmed weights.
- Primary operating context: 256K tokens.
- Development contexts: 2K and 32K.
- Characterization contexts: 128K, 512K, and 1M when capacity permits.

The primary metric is `gen_steady_tps`: committed tokens per second after the
first generated token, including model execution and sampling. Also report:

- `gen_tps` for the complete generation interval;
- `gen_first_ms` for time to first generated token;
- `prefill_tps` and total prefill time;
- committed and steady token counts;
- KV-cache bytes, process peak memory, and allocation/capacity failures; and
- wall time for the complete process invocation.

The headline is not draft-token speed, accepted-proposal speed, raw GPU kernel
speed, or tokens produced before rollback.

### Secondary: low-concurrency throughput

Concurrency 2--4 is a separate workload for future agentic use. It must report
both aggregate committed tokens/s and per-session latency/fairness. It must not
replace the batch-one result.

### Diagnostic workloads

Kernel, layer, replay, fixed-token, command-submission, and prefix-cache tests
are diagnostic. They must state exactly which work is excluded. Their purpose
is root-cause analysis, not the headline comparison.

## Prompt and Context Construction

Until a coding-agent corpus is versioned, use the oracle capture prompt:

- base: `speed-bench/promessi_sposi.txt` from the pinned oracle source;
- expansion: repeat the base bytes with two newline bytes between copies;
- target allocation: eight source bytes per requested context token;
- tokenizer and resulting token IDs: oracle-compatible and checked before the
  timed comparison.

Each timed process starts from no KV state and prefills to exactly one selected
frontier. Prefix-cache or persisted-session results belong in a separate
agent-loop workload.

## Pairing and Run Order

`A` is the pinned DwarfStar oracle and `B` is the candidate Rust Star commit.

1. Close unrelated memory-heavy work and record the privacy-filtered host state.
2. Run correctness/conformance gates before the timed block.
3. Execute one untimed warm-up for each engine at the smallest selected context.
4. Treat one A/B execution at the same context as a pair.
5. Alternate engine order by repetition: `A,B` then `B,A`.
6. Alternate context order by repetition: ascending then descending.
7. Launch every engine/context measurement in a fresh process.
8. Do not run full-logit dumping or profiler capture in a timed pair.

For development, collect at least five complete pairs at 2K, 32K, and 256K.
For a publishable primary claim, predeclare and collect nine complete pairs at
256K. Full-frontier characterization uses at least five pairs per feasible
context because 512K/1M runs are substantially more expensive.

Do not stop early because a favorable ratio appeared.

## Thermal and System Controls

Record before and after the timed block:

- macOS version/build, hardware identity without serial/UUID, and memory size;
- compiler and Metal toolchain versions;
- power mode, `pmset -g therm`, uptime, and relevant allowlisted environment;
- whether displays, background model processes, or other material load changed;
- model residency/warm-up policy and any memory-pressure/capacity event.

If a run is interrupted by a documented external event, retain it, mark the
pair invalid with the reason, and rerun the complete pair. Never remove a result
only because it is slow or inconsistent.

## Aggregation

Preserve every raw row. For each engine, context, and metric report:

- repetition count;
- median;
- median absolute deviation (MAD);
- minimum and maximum; and
- all raw observations or a checksummed raw artifact.

For the paired speedup, compute `candidate / oracle` within every valid pair,
then report the median ratio and its MAD/min/max. Do not report only the ratio
of two independently selected best runs or only the ratio of aggregate medians.

No universal target improvement is required. Report the observed distribution
and practical effect rather than converting noise into a pass/fail target.

## Failure and Capacity Semantics

- A correctness failure invalidates the corresponding default-path performance
  claim; it is not a zero-throughput sample.
- An allocation or memory-pressure failure is a capacity result and must retain
  the last successful context and diagnostics.
- Thermal throttling is part of sustained behavior when both engines face the
  same paired schedule. A one-sided thermal event invalidates that pair, not the
  complete experiment.
- Model-load time is excluded from steady decode throughput but remains recorded
  as an operational metric.

## Agent-Loop Extension

The later coding-agent benchmark will add versioned tasks and measure total wall
time from request availability to accepted final answer, including tokenization,
prefill, decoding, structured-output constraints, tool boundaries, cache reuse,
and harness transitions. Isolated decode remains reported beside it so a
harness gain cannot conceal an inference regression.

Any change to these rules creates a new protocol version; old results retain
their original label.
