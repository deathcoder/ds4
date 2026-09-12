# Restore declared Rust 1.74 compatibility

The runtime declared Rust 1.74 and both CI jobs installed 1.74.0, but the accepted source used APIs unavailable on that compiler. The corrected source passes all **320 Rust host tests**, the native optimized build, **94 Python artifact tests**, pinned formatting and the synthetic writer/reader check on this Mac. The minimum compiler declaration remains 1.74.

## Change

Only `rust-star/runtime/src/metal.rs` changes. Two `Option::is_none_or` calls become equivalent `map_or(true, predicate)` calls. A private safe `ArrayBorrows` adapter replaces 239 `each_ref` and 39 `each_mut` calls. It uses `std::array::from_fn` and `iter_mut` to borrow elements in index order without cloning or moving them. The helper uses no unsafe code or heap allocation. All other 41 pinned sources, including the Metal shim and shaders, remain byte-identical to accepted `4ced6ee8`.

The current-source snapshot and complete patch are included. Reversing the helper insertion and method substitutions, then running pinned rustfmt, recovers the preceding source exactly. Reversing the two Option substitutions recovers the complete accepted metal.rs snapshot in the preceding [wide-bootstrap report](../wide-bootstrap-c7adfb22/README.md).

## Preserved failures and validation

1. The first Rust 1.74 check of accepted source exits 101: two missing Option methods and 157 unstable array-method errors.
2. The initial Option-only repair passes formatting, but the host-test build exits 101 with 278 unstable array-method errors, including test-only callers. No tests ran in that failed build; later commands did not execute.
3. The corrected array-borrow implementation passes pinned formatting, all 320 host tests, optimized native build, all 94 Python artifact tests and the five-word synthetic writer/reader fixture. Every command, exit code, full log and source freeze is retained.

`Option::is_none_or` was introduced in Rust 1.82, after the declared minimum. [Official Rust API documentation](https://doc.rust-lang.org/std/option/enum.Option.html#method.is_none_or). The actual pinned compiler errors establish the array-method incompatibility directly.

The isolated Cargo target is reused only after the concrete source repair. The accepted performance executable remains unchanged, SHA-256 `2bf77edb9886f4efa5d011cba963aaef7e2bf140db1b3a09c6193e76bdc77af4`. The separately built Rust 1.74 executable is `d8c3919ed98281b36d1f9446c243d94655f416ebaa1a144778c5893ffaea161e`; binaries are omitted from the report.

## Verify

From the checkout, run:

```sh
python3 rust-star/reports/msrv-174-179e0c24/verify-report.py
```

The verifier requires Python 3 and installed Rust 1.74.0 with rustfmt. It verifies every artifact hash, the prior accepted source snapshots, the entire formatted source reversal, all preserved compiler failures, each successful command record and the synthetic artifact contract. It performs no model inference or GPU timing. Private workspace paths are redacted with separate original/stored hashes. The previous wide-bootstrap report must remain beside this report.

This is local macOS compiler/host/artifact compatibility evidence. The five-word fixture is synthetic; it does not establish model C0. No new full-model correctness, performance result, Linux execution or remote CI success is claimed. No workflow was dispatched and no publication occurred. The next planned milestone is the routed-input half experiment under its prospectively fixed local gates, followed by full C0 and whole-engine evidence only if selected.
