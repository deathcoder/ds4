# Closure checklist for the active 1K milestone

Active engine exec session:18051. Do not edit runtime/frozen scripts or run
another GPU workload/heavy build/model hash while it is active.

1. Drain session18051 only after engine-execution.log and engine/audit.json
   show the complete12 runs. Do not infer selection from partial runs.
2. Run audit-phases.py to audit the existing passive timing fields.
3. Run full-input-pins.py input-after.json with GPU idle; preserve its log.
4. Read all engine gates. If every gate passes, retain current candidate sources
   and install frozen candidate-rust-star at .work/runtime-target/release/rust-star.
   If any gate fails, restore the three source files from baseline-metal.rs,
   baseline-long_prefill.rs, baseline-metal_shim.m. All42 source hashes must
   equal plan.source_reference.sources afterward. Default performance executable
   remains2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4
   on rejection; experimental .work binaries stay preserved.
5. Write final-runtime-state.json(passed,sources,default_executable_sha256) and
   decision.json(closed,retained,reason,utc). Do not claim project completion.
6. Prepared result-independent helpers are currently OUTSIDE this directory:
   .work/bootstrap-1k-package.py
   .work/bootstrap-1k-write-report.py
   .work/bootstrap-1k-verify-report.py
   .work/bootstrap-1k-auditor-regression.py
   Copy appropriate helpers into this directory for preservation. The first two
   use repository cwd and explicit work paths; verifier uses its report parent.
7. Package to rust-star/reports/bootstrap-1k-d72b7528, write README, then run
   the report verifier into a unique .work verification log. It requires Python
   and installed Rust1.74 rustfmt, no GPU/model. Preserve/fix any verifier failure
   transparently; do not edit frozen original study scripts or raw evidence.
8. Update journal current state, charter history, initial-scope status. Run
   whitespace checks; preserve raw log/patch whitespace if necessary and record
   exact exclusions. Stage only intended runtime(if retained), docs and report.
9. Commit locally, write milestone-completion.json with commit/report/verifier
   hashes, then final update with outcome and next milestone. STOP here; user
   ended continuous token-burn mode. No push/release/256K or next implementation.

Accepted baseline source42 files is in msrv-c0-5834fe56/completion.json;
accepted current-compiler executable SHA is
d8c3919ed98281b36d1f9446c243d94655f416ebaa1a144778c5893ffaea161e.
Current candidate SHA is
31898eb72663f7ef2bac134569430c53a54232a49125a610e3a7a27adc996a5f.
All320 tests, router7840words/96guards, ring1567cases,12K internal, and both
full32K/128 C0 runs(33,354,240FP32 words) already pass. Profile4466intervals pass.
Diagnostic raw0.184402 and normalized0.195533 pass. No inference failures sofar.

Do not remove original untracked files rust-star-oracle-capture.txt,
rust-star-runtime-check.txt or rust-star/runtime/.work/. The package should
include only this experiment and pinned reference/adapter artifacts.
