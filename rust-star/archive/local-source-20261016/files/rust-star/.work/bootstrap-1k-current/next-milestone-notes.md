# Conditional next milestone, not started

Wait for the complete current engine decision. Do not select a follow-up from
partial paired statistics, change this study, or start another GPU workload.

If this candidate passes every gate, a fresh direct comparison against pinned
DwarfStar is a useful next milestone. This study compares two Rust executables
and does not update reference parity.

If this candidate fails the working-set gate, first investigate bootstrap
allocation lifetimes rather than merely choosing another tile width. Source
review shows many row-sized buffers allocated on each boundary call in
metal_shim.m (the allocation block beginning around lines5055–5386), while
prefillBoundaryCommands accumulates submitted commands and waits at the chunk
endpoint (around lines7037–7064). These source facts do not establish their
actual contribution to macOS peak memory footprint or a specific allocator/
GPU-resource lifetime mechanism.

A bounded follow-up could measure live scratch and submitted/completed-command
counts in a separate diagnostic run, then test a narrowly scoped scratch-reuse
or bounded-in-flight policy. Distinguish CPU-initialized token/position/mask
buffers from GPU-produced scratch. Reuse needs explicit ownership/dependency
proof; do not assume that retaining a command object or queue ordering alone
proves buffer lifetime or C0 safety. Preserve the ordinary prefix/legacy/decode
and higher-frontier paths. Any implementation needs fresh full C0 and its own
prospective whole-engine/memory criteria.

This note records an unimplemented follow-up. It neither explains away a failed
criterion nor establishes that another scheduling policy will be faster.
