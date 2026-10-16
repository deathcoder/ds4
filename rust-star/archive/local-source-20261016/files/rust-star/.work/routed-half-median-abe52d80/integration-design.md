# Conditional integration design — no runtime changes authorized by this screen yet

Execute only after this new protocol passes both contemporary neutrality and all
candidate selection criteria, its complete result is packaged/verified, and the
local milestone is closed. A fresh integration/full-C0/whole-engine plan must be
frozen first; these notes are not that executable plan.

- Baseline source is the accepted Rust 1.74 compatibility implementation at
  24c5d03e (subsequent commits so far are evidence/docs). The separate pinned
  Rust 1.74 optimized executable is under
  `.work/msrv-174-c7adfb22/target/release/rust-star`, recorded by
  `.work/msrv-174-array-borrows/completion.json`. It has host compatibility
  evidence, but no fresh full-model C0 or engine timing. Freeze its source,
  toolchain and executable, then run a normal full 32K/128 baseline C0 before
  interpreting a candidate comparison. Do not silently compare compiler versions.
- Keep all original shaders. Add a narrowly named production half-input shader
  derived from the exact standalone candidate by entry/type-name substitution
  only, with complete source inversion. Add it to both build.rs source tracking
  and the existing MoE library assembly after routed_prefill.metal.
- In the continuation tail use precisely the existing native_attention_half
  predicate (nonzero native chunk at its start), not the broader native_chunk
  predicate. Keep initial prefix, legacy, bootstrap and decode schedules intact.
- Reuse heads_half_buffer after its last attention-low reader. The fixed 4K
  buffer is 256 MiB and the normalized FFN conversion requires 32 MiB. Add an
  explicit capacity check; keep ffn_norm_buffer unchanged for the shared expert.
  No new transient allocation or ownership/release boundary is needed.
- Add the half-input pipeline lazily for the selected scope. Avoid compiling the
  unused original native gate/up pipeline for that same scope; preserve its
  existing initialization for remaining callers. Keep the 128-thread group,
  16 KiB threadgroup memory and 32-by-work-cap dispatch grid unchanged.
- Encode conversion after the routed-map/profile boundary 4 and before gate/up,
  so the gate/up profile interval includes conversion cost. Halve exactly the
  four RHS byte strides, as in the verified probe; all other ABI fields and
  arithmetic remain identical. Account for the additional dispatch without
  silently changing unrelated historical metadata.
- Required later evidence: same-compiler source/executable/input pins, all host
  contracts/build checks, fresh full baseline and candidate 32K/128 C0, strict
  counter attribution if profiling is used, and a separately prospectively
  frozen unprofiled whole-engine block. Preserve every failure and both warmups;
  do not infer an engine gain, DwarfStar parity or higher-frontier correctness
  from a standalone kernel result. The old 256K campaign remains paused.
