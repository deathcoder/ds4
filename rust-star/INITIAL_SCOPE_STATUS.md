# Rust Star initial-scope status

Updated 2026-09-12. The charter remains authoritative. This checklist supports
continuous milestone execution; an accepted optimization does not close the
project, and no arbitrary percentage gain defines research success.

| Requirement or milestone | Current evidence | Remaining work |
|---|---|---|
| Fixed-target Q2 resident decoder and benchmark | Native 43-layer runtime and greedy decoder exist on the M1 Ultra | Keep supported-mode boundaries and user-facing invocation accurate as code evolves |
| Immutable oracle and exact FP32 contract | Oracle-v5 and full-output differential fixtures are established | Preserve provenance; diagnose every new mismatch before performance acceptance |
| Current lower-context performance | Retained 4ced6ee8 passes full 32K/128 C0 and five Rust-versus-Rust pairs; 4.329% lower paired prefill latency | Continue profile-driven whole-engine decisions and refresh the direct pinned-DwarfStar comparison |
| Active optimization | Rust 1.74 current build passes fresh normal/profiled 32K/128 C0; bootstrap ingress 16.233% of recorded GPU stage time | Run a separate 1K bootstrap working-set study with exact source/internal/full-C0 gates and fixed engine/RSS/footprint criteria; preserve both routed-half and 4K failures. |
| Larger-frontier correctness on current source | Earlier versions have full 128K evidence; current attention-output changes have fresh full-model evidence at 32K only | Revalidate source-dependent frontiers under a new prospective plan after addressing the lower-context performance priority |
| Primary 256K operating point | Earlier attempt is preserved as interrupted, not accepted | Keep the interrupted campaign paused; establish a justified new correctness/capacity/performance milestone rather than automatically resuming it |
| Capacity characterization beyond 256K | Desired when memory permits, not guaranteed by model theoretical capacity | Plan bounded allocation/capacity evidence and preserve infeasible-frontier results |
| Reproducible project handoff | Journals, source/executable/input pins and portable reports are maintained | Keep complete build/run/verification instructions, limitations and remaining blockers consistent with final tested source |
| Remote host-contract CI | Sep12 read-only refresh still lists 134 runs, latest Aug29 failure with no job steps; manual-only workflow matches locally | Pinned Rust 1.74 local compatibility now passes; account restriction and a successful manual workflow remain unverified. Local checks do not imply remote success |

Q2/Q4 comparison and agent/engine co-design remain later charter research phases.
A server, speculative decoding, unrelated hardware/model support, and publication
are not silently added to the initial executable scope. Any unresolved initial
requirement must remain visible; research goals without a finite acceptance
condition must not be declared complete by inventing one.

The current instruction is to continue from one complete milestone to the next.
A rejected candidate closes its hypothesis and restores the accepted path; it
is not a reason to stop useful project work. The active journal and frozen work
plans identify the exact next action and any running process.
