# Rust Star Project Charter

Status: active research project. This file records stable scope, contracts, and
decisions. `RUST_STAR_JOURNAL.md` records changing state and chronological work.
If the two disagree, resolve the discrepancy explicitly and update both files.

## Mission

Build a DeepSeek V4 Flash inference runtime for one specific Apple Silicon
machine, learn the engine deeply enough to produce original optimizations, and
measure how far a narrow runtime can push local coding-agent throughput beyond
DwarfStar on the same hardware and model.

The research question is broader than "is Rust faster than C?" Rust and Metal
are implementation tools. The contribution is to test whether removing
unneeded generality, specializing scheduling and memory policy for one machine,
and eventually co-designing inference with an agent harness creates meaningful
end-to-end gains.

This is an ongoing project. There is no arbitrary minimum percentage improvement
that determines success. Negative results are useful when they are reproducible
and explain where time or bandwidth goes.

## Fixed Initial Target

- Machine: Mac Studio with M1 Ultra, 48 GPU cores, and 128 GB unified memory.
- Platform: macOS and Metal on that machine only.
- Model family/checkpoint: DeepSeek-V4-Flash-0731 only.
- Initial model format: DwarfStar's imatrix Q2 GGUF for the 128 GB resident path.
- Reference implementation: a versioned snapshot of upstream
  `antirez/ds4`, not this fork.
- Primary workload: batch-one decode for an interactive coding agent.
- Initial product surface: a minimal decoder and benchmark executable, not an
  HTTP server or complete agent framework.

The exact GGUF path and SHA-256, macOS build, Xcode/Metal toolchain, compiler
flags, and runtime configuration belong in the first oracle manifest. Until
that manifest is captured on the target Mac, the oracle is not fully pinned.

## Deliberate Initial Non-Goals

The first resident Metal path does not need to support:

- CUDA, ROCm, Linux, or non-Apple hardware;
- distributed inference;
- CPU inference as a production path;
- DeepSeek V4 Pro, GLM, or arbitrary model families;
- arbitrary GGUF layouts or every quantization;
- SSD expert streaming;
- an OpenAI/Anthropic-compatible server;
- speculative decoding; or
- portability across Apple Silicon generations.

Narrow scope is permission to remove abstractions, branches, formats, and
runtime checks that do not serve the fixed target. It is not permission to make
benchmark-only shortcuts or silently weaken correctness.

## Correctness Contract

The default runtime starts with a strict C0 contract: for the supported
configuration it must produce bit-identical full pre-softmax FP32 logits to the
pinned DwarfStar oracle. Greedy token agreement, top-k agreement, or similar
perplexity is insufficient for C0.

Correctness levels are named so benchmark results cannot blur different claims:

| Level | Contract | Eligible for the default path? |
| --- | --- | --- |
| C0 | Every full FP32 logit tensor is bit-identical to the pinned oracle. | Yes; initially required. |
| C1 | The probability distribution is identical under a precisely specified normalization, but raw logits are not bit-identical. | No, unless the project deliberately revises the default contract. |
| C2 | Numerical divergence is bounded by declared tensor/distribution metrics. | Experimental only. |
| C3 | Only behavioral or task-quality gates are met. | Experimental only. |

"Byte exact" and "same distribution" are not synonyms. Adding a constant to
all logits preserves a softmax distribution but violates C0. Likewise, matching
the sampled or greedy sequence does not prove either contract.

DwarfStar is built with fast-math optimizations. C0 therefore means identical
to a pinned executable, source revision, toolchain, flags, hardware, model, and
runtime configuration. It is not a claim of universal IEEE-754 reproducibility.

Potentially C0-safe optimization areas include:

- tensor or storage layout changes that preserve values and evaluation order;
- memory residency and prefetch policy;
- command-buffer scheduling;
- buffer reuse and allocation removal;
- eliminating redundant copies;
- dispatch changes that preserve arithmetic order;
- prefix/KV reuse; and
- overlap of operations proven independent.

The following require evidence and must not be presumed C0-safe:

- kernel fusion that changes arithmetic order;
- alternative parallel reductions;
- changes to FMA contraction or compiler math mode;
- approximate activations or arithmetic;
- lossy weights or KV-cache formats; and
- speculative or batched verification paths.

## Oracle Policy

The oracle moves through explicit immutable versions rather than silently
tracking upstream:

```text
oracle-vN = upstream commit
          + model artifact SHA-256
          + hardware identity
          + macOS and Metal/Xcode versions
          + compiler and flags
          + complete inference configuration
          + golden artifact schema/version
```

`current-oracle` may advance after an upstream update is evaluated. Advancing it
requires:

1. compare old and proposed oracle full logits on the conformance corpus;
2. identify whether differences are numerical, behavioral, or performance-only;
3. validate relevant correctness and speed regressions;
4. preserve historical measurements under the old oracle version; and
5. record the decision in `RUST_STAR_JOURNAL.md`.

The initial source candidate is upstream commit
`b0309611041655f4e45671cfd9c9886aff161406`. It becomes `oracle-v1` only after
the remaining manifest fields and golden outputs are captured on the target
machine.

Conformance and performance are separate run modes:

- Conformance mode captures or streams hashes and, on mismatch, full tensors at
  selected kernel, layer, and final-logit boundaries.
- Performance mode removes comparison, capture, tracing, and synchronization
  that are not part of normal inference.

A useful differential harness should locate the first mismatch, not merely say
that the last token differs.

`rust-star/ARTIFACT_FORMAT.md` is the canonical initial bundle and full-logit
contract. `rust-star/verify_oracle_bundle.py` verifies returned captures, while
`rust-star/compare_logits.py` performs bit-pattern C0 comparison and reports
non-C0 drift diagnostics without weakening the exact classification.

## Performance Contract and Benchmarks

The primary metric is committed decode tokens per second at batch one. The
timed interval must include all work needed to produce committed tokens,
including sampling. Warm-up, prompt, generation length, EOS handling, thermal
state, power mode, background load, model file, and context state must be held
constant and recorded.

Secondary metrics remain in research scope because they affect coding-agent
wall time:

- prefill tokens per second;
- time to first token;
- peak/resident memory and high-water mark;
- whole agent-loop or task wall time; and
- aggregate decode throughput at low concurrency, initially 2--4 live sessions.

For any future speculative path, committed tokens per second must include draft,
verification, rollback, and sampling time. Accepted-draft throughput alone is
not a valid comparison.

Use paired runs on the same machine and report distributions, not a favorable
single run. At minimum record median, dispersion, number of repetitions, and
the exact commit/oracle/configuration. Correctness instrumentation must be off
for headline performance runs.

Suggested context frontiers are 2K, 32K, 128K, 256K, 512K, and 1M tokens. The
common operating target is 256K, while correct operation up to the model's 1M
limit is desirable when memory permits. There is no separate semantic
"capacity mode" initially; these are benchmark operating points.

Moving from 256K to 1M is not merely four times more stored tokens. The model's
compressed/indexed attention structures grow and can move the bottleneck from
MoE weight traffic toward attention/indexing. DwarfStar estimates roughly 26 GB
for a 1M context, including about 22 GB for the compressed indexer. Combined
with an approximately 81 GB Q2 model, graph scratch space, and runtime buffers,
this approaches the practical working-set limit of a 128 GB machine. Capacity,
allocation safety, prefill, and decode must therefore be measured separately.

## Quantization Plan

Start with the imatrix Q2 model because it is the intended resident quant for a
128 GB machine. Later evaluate DwarfStar's Q2/Q4 hybrid, which stores the last
six expert layers at Q4.

An observed question worth testing is why Q4 can sometimes show decode
throughput comparable to Q2 despite moving more bytes. Possible causes include
Q2 unpack/codebook work, occupancy, instruction pressure, or a different active
bottleneck. This is a hypothesis, not a conclusion.

Full resident Q4 is not an initial target because DwarfStar recommends at least
256 GB. SSD-backed or other capacity experiments may be considered later, but
must not distort the first resident-runtime design.

## Implementation Direction

The working sequence is:

1. Create a minimal Rust host runtime, decoder, and benchmark.
2. Support the single Q2-imatrix resident Metal configuration.
3. Create the versioned oracle manifest and differential artifact format.
4. Reach C0 full-logit equivalence before claiming optimizations.
5. Initially reuse proven DwarfStar Metal kernels where practical, replacing or
   specializing one kernel at a time behind differential tests.
6. Profile before each optimization and keep only measured improvements.
7. Add the Q2/Q4 hybrid after the Q2 path is understood.
8. Add agent-native facilities only after the inference core is stable.

Rust does not make GPU kernels faster by itself. Metal kernels and unified-memory
traffic dominate device work. Rust is chosen as a viable way to build a narrow
host scheduler and eventually manage concurrent session/KV ownership safely.
Existing DwarfStar instructions forbid introducing C++ into its C production
path, which is another reason to keep this research runtime structurally clear.

The external `yijunyu/ds4-rs-metal` project is a useful comparison, not proof
that Rust is intrinsically faster. Its reported M1 Ultra results show a prefill
gain but approximate decode parity, with some differences attributed to
scheduling and attention defaults. Reproduce comparable configurations before
drawing conclusions.

## Agent/Engine Co-Design Research

The first executable is an inference benchmark, but the broader research scope
includes an agent-native harness. Candidate exact-safe gains include:

- keeping state in token IDs without unnecessary detokenize/re-tokenize cycles;
- multiple live sessions sharing immutable prefixes;
- copy-on-write or DAG-backed KV branching;
- explicit lifecycle and buffer pools instead of request-level allocation;
- structured-decoding state integrated without IPC; and
- measuring whole coding-agent loops rather than only isolated model calls.

These ideas need profiles. Tokenization, IPC, and parsing may be negligible next
to decode, while prefix/KV reuse may materially reduce repeated prefill. The
project must measure rather than assume which layer matters.

## Speculative Decoding Boundary

Speculative decoding is not on the initial critical path. A separate agent has
already investigated DSpark on branch `codex/dspark-observability-0`; avoid
duplicating that work. Its recorded paired HumanEval result was 0.8923x the
ordinary baseline geometrically, with only 2 of 32 tasks faster. Target
verification consumed most runtime, and the experiment estimated that target
time needed roughly a 13.4% reduction merely to reach parity.

Treat this as evidence for deprioritization, not a permanent claim that
speculation cannot work. Port a speculative mechanism only after it demonstrates
correctness and a relevant end-to-end win on this target.

## Experimental Feature Policy

An experimental flag may relax C0, but its name and documentation must state:

- the operation, tensor, or cache whose semantics change;
- the resulting correctness level (C1, C2, or C3);
- the comparison oracle and corpus;
- numerical metrics and worst cases, not only averages;
- task-quality results relevant to coding-agent use;
- context and quantization coverage;
- throughput, latency, and memory change; and
- whether outputs are reproducible.

Experimental results must be reported separately from the exact baseline. No
approximation may silently become the default.

## Durable Work Protocol

At the start of every session:

1. read this file, `RUST_STAR_JOURNAL.md`, and
   `RUST_STAR_MANUAL_TASKS.md`;
2. inspect the current branch, commit, working tree, and remotes;
3. confirm the active oracle and benchmark configuration; and
4. resume from the journal's current state and next actions.

During and at the end of every meaningful session, update the journal with:

- date and objective;
- branch and relevant commit IDs;
- files or mechanisms changed;
- commands, target machine, model, and configuration used;
- measured results or explicit statement that no benchmark was run;
- decisions, failed hypotheses, and blockers; and
- the next reproducible action.

Do not rewrite inconvenient history. Correct an old entry with a new entry and
update the mutable current-state summary. Commit journal updates with the work
they describe whenever practical.

## Open Items

- Capture the complete `oracle-v1` manifest on the M1 Ultra.
- Extend the stable full-logit artifact schema to kernel, layer, and every
  decode-step boundaries as those hooks are implemented.
- Execute `rust-star/BENCHMARK_PROTOCOL.md` and capture the initial paired
  DwarfStar numbers; the v1 protocol itself is now defined.
- Decide the minimal Rust/Objective-C/Metal interop layer after a measured host
  dispatch prototype.
- Reproduce the Q2 versus Q2/Q4 throughput observation under controlled runs.
- Define the first coding-agent workload corpus before agent co-design claims.
