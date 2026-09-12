# Rust Star initial-scope status

Updated 2026-09-12. The charter remains authoritative. This checklist supports
work through one milestone followed by an update; an accepted optimization does not close the
project, and no arbitrary percentage gain defines research success.

| Requirement or milestone | Current evidence | Remaining work |
|---|---|---|
| Fixed-target Q2 resident decoder and benchmark | Native 43-layer runtime and greedy decoder exist on the M1 Ultra | Keep supported-mode boundaries and user-facing invocation accurate as code evolves |
| Immutable oracle and exact FP32 contract | Oracle-v5 and full-output differential fixtures are established | Preserve provenance; diagnose every new mismatch before performance acceptance |
| Current lower-context performance | Retained 4ced6ee8 passes full 32K/128 C0 and five Rust-versus-Rust pairs; 4.329% lower paired prefill latency | Continue profile-driven whole-engine decisions and refresh the direct pinned-DwarfStar comparison |
| Latest optimization | 1K bootstrap passes full C0 and every timing/decode/RSS gate; paired prefill latency -11.234%, but separate peak memory footprint +44.722% fails the <=15% gate. Accepted source restored. | Lifetime diagnostic is complete; the prior optimization stays rejected. |
| Bootstrap lifetime diagnostic | Both fresh 32K/128 runs preserve 33,354,240 exact FP32 words. Four 1K scratch sets overlap; earlier sets release after completion and every chunk pool drains tracked scratch to zero. Verified 60-artifact report; accepted runtime unchanged. | Next milestone: test one unfinished eligible 1K tile at a time with checked completion, full C0 and unchanged engine/footprint gates. Not started; OS footprint cause remains unresolved. |
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

The user authorized one bounded 1K bootstrap milestone after ending continuous
execution. That milestone is complete: the candidate was rejected on memory
footprint, accepted source is restored, and the portable report verifies.
Report the outcome and next milestone, then stop. No follow-up implementation
or inference run has started.
