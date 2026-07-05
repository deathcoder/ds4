# DSpark Development Journal

This file is durable working memory for the DSpark-on-ds4 effort. Read it when
resuming after context compaction, switching agents, or feeling unsure why a
particular DSpark change exists.

## Current Phase

Branch: `codex/dspark-observability-0`

Phase 0.6 is still intentionally narrow: `--dspark FILE` validates an official
DSpark drafter GGUF, binds every tensor needed by a future runtime path, checks
the expected DeepSeek V4 Flash DSpark shapes, and exposes a diagnostic
`--dspark-probe` bridge for target-layer hidden-state capture plus
`main_proj/main_norm`. It does not enable DSpark speculative decoding.

The design choice was to keep this separate from legacy `--mtp`. We do not
guess dynamically whether `--mtp` points at legacy MTP or DSpark. The first
hook is explicit and conservative:

- `--mtp FILE` remains the legacy MTP addon path.
- `--dspark FILE` means "validate this as a DSpark drafter sidecar".
- Supplying both is an error.
- DSpark validation happens before the `--inspect` early return, so
  `./ds4 --inspect --model base.gguf --dspark ds4-dspark.gguf` should validate
  the sidecar.
- `--dspark-probe` is a development diagnostic. It requires `--dspark`, runs a
  prompt through graph layer slices ending at target layers `40,41,42`,
  collapses the captured HC states to 4096-wide candidate context vectors, runs
  DSpark `main_proj/main_norm`, prints vector stats, and exits without
  generating or drafting tokens.
- No benchmarks should be run automatically by Codex. The user wants tok/s
  benchmarks to be run manually on an otherwise idle machine.

## Sources And Contract

Primary references used for the current phase:

- ds4 discussion: <https://github.com/antirez/ds4/issues/468>
- DeepSpec repo: <https://github.com/deepseek-ai/DeepSpec>
- Converted DSpark GGUF sidecar for validation:
  <https://huggingface.co/buckets/lobanov/ds4/tree/ds4flash-dspark.gguf>
  - Direct download:
    <https://huggingface.co/buckets/lobanov/ds4/resolve/ds4flash-dspark.gguf?download=true>
  - Size shown by Hugging Face on 2026-07-05: 11.5 GB.
  - Expected byte size from HTTP headers: `11489939840`.
  - Local target path: `gguf/ds4flash-dspark.gguf`.
  - Resume command:
    ```sh
    curl -L --fail -C - \
      -o /Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf \
      'https://huggingface.co/buckets/lobanov/ds4/resolve/ds4flash-dspark.gguf?download=true'
    ```
- A shallow DeepSpec checkout was used under `/tmp/DeepSpec`; do not assume it
  still exists.

DSpark contract we encoded for DeepSeek-V4-Flash DSpark:

- Three draft stages: `mtp.0`, `mtp.1`, `mtp.2`.
- Block size: `5`.
- Noise token id: `128799`.
- Markov rank: `256`.
- Target base layer ids: `40, 41, 42`.
- Stage 0 global inputs: `mtp.0.main_proj.weight`,
  `mtp.0.main_norm.weight`.
- Stage 2 output/head tensors: `mtp.2.norm.weight`,
  `mtp.2.hc_head_base.weight`, `mtp.2.hc_head_fn.weight`,
  `mtp.2.hc_head_scale.weight`,
  `mtp.2.markov_head.markov_w1.weight`,
  `mtp.2.markov_head.markov_w2.weight`,
  `mtp.2.confidence_head.proj.weight`.
- Every stage requires the DS4 layer-style suffixes:
  `hc_attn_fn.weight`, `hc_attn_scale.weight`, `hc_attn_base.weight`,
  `attn_norm.weight`, `attn_q_a.weight`, `attn_q_a_norm.weight`,
  `attn_q_b.weight`, `attn_kv.weight`, `attn_kv_a_norm.weight`,
  `attn_sinks.weight`, `attn_output_a.weight`, `attn_output_b.weight`,
  `hc_ffn_fn.weight`, `hc_ffn_scale.weight`, `hc_ffn_base.weight`,
  `ffn_norm.weight`, `ffn_gate_inp.weight`, `exp_probs_b.bias`,
  `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`,
  `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`,
  `ffn_down_shexp.weight`.

Important nuance: metadata keys under `deepseek4.dspark.*` are treated as
optional. If absent, ds4 uses the defaults above. If present, values must match.
This avoids rejecting current GGUFs that only communicate DSpark-ness through
tensor names.

Phase 0.5 shape/binding facts from the local
`gguf/ds4flash-dspark.gguf` sidecar:

- `exp_probs_b.bias` is present in all three stages and is now required. The
  earlier Phase 0 note that treated it as optional was wrong.
- The sidecar has 81 tensors: 9 global/output tensors plus 24 per-stage tensors
  across 3 stages.
- `mtp.0.main_proj.weight` is Q8_0 `[12288,4096]`, i.e.
  `[3*N_EMBD,N_EMBD]`.
- Each DSpark stage has the same DS4 layer-style attention/FFN shape as the
  legacy one-block MTP validator expects.
- `mtp.2.markov_head.markov_w1.weight` and `markov_w2.weight` are validated by
  dimensions only as `[markov_rank,N_VOCAB]` = `[256,129280]`.
- `mtp.2.confidence_head.proj.weight` is validated by dimensions only as 1D
  `[N_EMBD + markov_rank]` = `[4352]`.
- The current phase still has no runtime execution, no GPU-heavy generation
  path, no expert stats, and no benchmark claims.
- The first probe collapse method is intentionally marked as provisional:
  target layer `40` is collapsed with layer `41`'s attention HC-pre weights,
  target layer `41` with layer `42`'s attention HC-pre weights, and target layer
  `42` with the base output HC head. This gives a concrete 4096-wide bridge for
  `main_proj`; it still needs equivalence validation against official
  DeepSpec/HF hidden-state captures before becoming runtime contract.

## Files Changed So Far

- `ds4.h`
  - Added `dspark_path` to `ds4_engine_options`.
  - Added public `ds4_dspark_config`.
  - Added validation helper prototypes:
    `ds4_dspark_config_init_defaults` and
    `ds4_dspark_tensor_names_validate`.

- `ds4.c`
  - Added DSpark default config and tensor-name validation helpers.
  - Added model-backed DSpark validation for required tensors and optional
    metadata.
  - Added `ds4_dspark_weights` and `dspark_weights_bind` for the official
    DSpark sidecar tensor layout.
  - Added DSpark layout validation for global inputs, three per-stage
    transformer blocks, final HC head tensors, Markov heads, and the confidence
    head.
  - Added `ds4_engine_dspark_probe`, a graph-backend-only diagnostic that uses
    layer-slice evaluation to capture target-layer HC states, performs the
    provisional HC-to-plain collapse, and runs DSpark `main_proj/main_norm`.
  - DSpark Markov and confidence tensors are dimension-validated only for now;
    runtime math is not implemented yet.
  - Added `dspark_model`, `dspark_config`, and `dspark_ready` to `ds4_engine`.
  - `ds4_engine_open` now rejects `--mtp` plus `--dspark`, opens the DSpark
    sidecar, validates/binds it, and logs that runtime is not enabled yet.
  - `ds4_engine_close` closes the DSpark model if it was opened.
  - The DSpark path does not set `mtp_ready`, bind legacy MTP draft weights, or
    affect generation.

- Frontend parsers
  - `ds4_cli.c`, `ds4_server.c`, `ds4_eval.c`, and `ds4_agent.c` all accept
    `--dspark FILE`.
  - `ds4_cli.c` accepts `--dspark-probe` as a one-shot diagnostic; the server,
    eval, and agent frontends do not expose this probe.
  - `ds4-agent` was easy to miss because shared help mentioned the flag once
    `ds4_help.c` was updated. When adding runtime flags, always check all
    frontends.

- `ds4_help.c`
  - Runtime full help now documents `--dspark FILE`.
  - Diagnostics help now documents `--dspark-probe`.

- `README.md`
  - Added a short validation/probe note next to the existing MTP text.

- `tests/ds4_test.c`
  - Added `--dspark-validation`, a model-free test for defaults, rejection of
    legacy/ordinary MTP-looking names, and acceptance of a synthetic full DSpark
    tensor-name list.
  - Added `--dspark-shape-binding`, a model-free sanity test for the documented
    DeepSeek V4 Flash DSpark shape arithmetic. The real GGUF inspect path is
    what validates those dimensions against `ds4.c`'s private model-shape
    constants.

- `implementation_plan.md`
  - Imported from the user/Claude handoff. The plan direction was validated:
    Phase 0.5 should be shape and binding validation only, not runtime work.
  - Adjustment made during completion: keep Markov/confidence validation
    dimension-only, and do not widen the public API just to expose private
    shape constants to the model-free test.

## Verification Already Run

Build/check commands run after Phase 0/0.5 implementation:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation
./ds4_test --dspark-shape-binding
./ds4 --help runtime | rg -- '--dspark'
./ds4-server --help runtime | rg -- '--dspark'
./ds4-eval --help runtime | rg -- '--dspark'
./ds4-agent --help runtime | rg -- '--dspark'
git diff --check
```

No tok/s benchmark was run.

Real DSpark sidecar validation:

```sh
./ds4 --inspect \
  --model /Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf
```

Result on 2026-07-05: validation passed.

```text
ds4: DSpark drafter validated: /Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf (block=5 target_layers=40,41,42 main_proj=[12288,4096] stages=3; runtime not enabled yet)
```

Phase 0.5 completion checks run on 2026-07-05:

```sh
make -B ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
make -B ds4
./ds4 --inspect \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf
make ds4 ds4-server ds4-eval ds4-agent ds4_test
```

The inspect output confirmed:

```text
ds4: DSpark drafter validated: gguf/ds4flash-dspark.gguf (block=5 target_layers=40,41,42 main_proj=[12288,4096] stages=3; runtime not enabled yet)
```

Phase 0.6 probe checks run on 2026-07-05:

```sh
make ds4 ds4-server ds4-eval ds4-agent ds4_test
./ds4_test --dspark-validation --dspark-shape-binding
./ds4 --help diagnostics | rg -- '--dspark|--dspark-probe'
./ds4 --inspect \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark gguf/ds4flash-dspark.gguf \
  --dspark-probe --nothink -p "Hello"
git diff --check
```

The real probe completed on the downloaded base+DSpark GGUFs. It captured
target layers `40,41,42`, collapsed them with the provisional HC-to-plain
bridge, ran `main_proj/main_norm`, and exited before generation/speculation.
Observed last-token vector stats were finite:

```text
layer 40 hc rms=3.39173 plain rms=3.93296
layer 41 hc rms=3.54454 plain rms=4.50141
layer 42 hc rms=5.36406 plain rms=7.51114
main_proj rms=26.5668
main_norm rms=0.0882359
```

Downloaded sidecar byte size:

```text
/Users/deathcodevision/dev/ds4/gguf/ds4flash-dspark.gguf
11489939840 bytes
```

## Resume Checklist

If continuing from a compacted context, start here:

1. Check current state:

   ```sh
   git status --short --branch
   git diff --stat
   ```

2. Re-scan all engine-option consumers before adding or renaming a runtime
   option:

   ```sh
   rg -n -- "ds4_engine_options|ds4_engine_open\\(|--mtp|ds4_help_print"
   ```

3. If testing a real DSpark sidecar, start with inspect/validation only:

   ```sh
   ./ds4 --inspect --model /path/to/base.gguf --dspark /path/to/ds4-dspark.gguf
   ```

   If validation fails, record the exact missing tensor/metadata key here before
   changing the contract. Adjust only with evidence from an actual GGUF,
   converter code, DeepSpec, or the ds4 issue discussion.

4. If adding runtime execution, do not reuse the legacy MTP path blindly.
   DSpark input wiring differs:

   - It needs captured target-layer hidden states from base layers `40,41,42`.
   - It uses `main_proj/main_norm` over those states.
   - It uses noise-token block inputs for the parallel block.
   - It shares base embeddings and the base LM head; they are not expected in
     the sidecar.
   - It has Markov and confidence heads on the last stage.

5. Keep benchmark discipline:

   - Do not run tok/s benchmarks automatically.
   - Avoid hot-path logging by default.
   - If adding timing/debug logs, gate them behind an explicit option or env var.
   - For performance comparisons, the user should run no-log baseline first on
     an idle machine, then optional instrumentation runs.

## Open Questions

- How do we validate the provisional HC-to-plain collapse against official
  DeepSpec/HF hidden-state captures before using it as runtime contract?
- What is the exact first minimal DSpark runtime experiment after collapse
  validation: CPU-only correctness probe, Metal graph wiring, or an isolated
  sidecar forward helper?
- Where should DSpark runtime stats live so they are useful for development but
  cannot perturb benchmark measurements unless explicitly enabled?
- Should future CLI UX stay as `--dspark`, or eventually become a broader
  `--draft dspark` once runtime is real?
