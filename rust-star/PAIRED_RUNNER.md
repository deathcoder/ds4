# Checkpointed Paired Benchmark Runner

Schemas: `rust-star-paired-plan-v1` and `rust-star-paired-state-v1`.

`run_paired_benchmark.py` executes the schedule in
`BENCHMARK_PROTOCOL.md`. It invokes both engines through
`ENGINE_MEASUREMENT_FORMAT.md`, checkpoints after every engine process, and
produces `rust-star-paired-raw-v1` plus its validated summary only when every
predeclared pair has a final valid attempt.

## Plan

Plans are local JSON files. They bind correctness/host manifests, contexts,
repetitions, engine identities, explicit build/runtime configurations, and
adapter commands. A plan may contain private local paths, so do not add or share
it without review. Results store its SHA-256 rather than copying it.

Each `adapter_command` is a JSON string array, never a shell command. It must
contain `{context}`, `{gen_tokens}`, and `{output}` exactly once as complete
arguments. A DwarfStar entry can call `measure_dwarfstar.py`; the Rust Star
entry will follow the same output contract once its decoder exists.

Minimal structural example:

```json
{
  "schema": "rust-star-paired-plan-v1",
  "protocol": "rust-star-paired-benchmark-v1",
  "correctness_class": "C0",
  "host_manifest_sha256": "<64 lowercase hex>",
  "correctness_manifest_sha256": "<64 lowercase hex>",
  "configuration": {
    "contexts": [2048, 32768, 262144],
    "repetitions": 5,
    "gen_tokens": 128,
    "sampling": "greedy argmax excluding EOS",
    "primary_metric": "gen_steady_tps"
  },
  "oracle": {
    "identity": "<engine identity object from PAIRED_RESULT_FORMAT.md>",
    "working_directory": "/path/to/oracle/source",
    "adapter_command": [
      "python3", "/path/to/measure_dwarfstar.py",
      "--executable", "/path/to/oracle/ds4-bench",
      "--model", "/path/to/model.gguf",
      "--prompt", "/path/to/prompt.txt",
      "--context", "{context}",
      "--gen-tokens", "{gen_tokens}",
      "--output", "{output}"
    ]
  },
  "candidate": {
    "identity": "<Rust Star engine identity object>",
    "working_directory": "/path/to/rust-star/source",
    "adapter_command": [
      "python3", "/path/to/measure_ruststar.py",
      "--executable", "/path/to/rust-star",
      "--model", "/path/to/model.gguf",
      "--context", "{context}",
      "--gen-tokens", "{gen_tokens}",
      "--output", "{output}"
    ]
  }
}
```

The quoted identity placeholders above must be replaced by JSON objects, not
strings. The runner rejects changed plan bytes after creating a checkpoint.

## Run and resume

```sh
python3 rust-star/run_paired_benchmark.py run paired-plan.json \
  --output /path/to/paired-results
```

Run only one new timed pair when observing thermal/system state manually:

```sh
python3 rust-star/run_paired_benchmark.py run paired-plan.json \
  --output /path/to/paired-results \
  --max-new-pairs 1
```

The same command resumes from `state.json`. The runner performs one untimed
smallest-context warm-up per engine, alternates A/B versus B/A by repetition,
and alternates ascending versus descending contexts. Every adapter invocation
is a fresh process group. Finalization re-hashes and revalidates the underlying
measurement artifacts, so editing cached metrics in `state.json` cannot create
a publishable result.

Inspect progress without running an engine:

```sh
python3 rust-star/run_paired_benchmark.py status paired-plan.json \
  --output /path/to/paired-results
```

## Failures and retries

A timed adapter failure blocks the schedule and remains in `state.json`. Do not
retry a capacity, correctness, or reproducible engine failure merely to obtain a
complete table. If a pair is invalid only because of a documented external
event, invalidate and rerun the complete pair explicitly:

```sh
python3 rust-star/run_paired_benchmark.py retry paired-plan.json \
  --output /path/to/paired-results \
  --context 262144 \
  --repetition 3 \
  --reason "unrelated backup process started during this pair"
```

The original attempt and evidence remain. If already-finalized output is
invalidated, the old raw/summary files move under `superseded/revision-NN/`
before the replacement run.

After a hard interruption, the latest pair remains `running` and the runner
blocks. Confirm that no adapter or engine process remains before using an
explicit retry reason. A warm-up failure normally indicates a bad plan or
engine setup; correct the plan and start a new output directory.

Exit status is 0 for successful, paused, complete, and status operations; 2 for
invalid inputs/contracts; and 3 when execution is blocked by an adapter or pair
failure.
