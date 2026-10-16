# Bounded 1K scheduling scope

The candidate inherits the exact previously rejected 1K Rust source and Metal
shim, pinned by prior-1k-build.json. Its two Rust files are byte-identical to
that candidate. Removing the single recorded scheduling insertion from the new
shim recovers the prior shim exactly. All other 39 source hashes match the
accepted runtime, including every Metal shader and fixture consumer.

The insertion is inside batch_production_tiles and after command commit and
addition to prefillBoundaryCommands. It runs only when native_1k_bootstrap is
true and the current tile is nonfinal. That existing predicate requires 1024
rows, output collection disabled, complete layer1 plus layer2 compressors,
consume-state mode, a supported positive native continuation position and an
8K..32K aligned chunk frontier. The existing context check also requires the
active native chunk start to match the frontier. The higher-level Rust caller
selects 1K only for supported overall frontiers up to32K.

The checked helper waits for completion and returns a Metal command error to
the caller. The final tile retains its existing wait and full command error
audit. The batch array retains first/last commands for the existing timing
calculation; its GPU span can include gaps and remains distinct from summed
kernel execution time. No timer or dispatch-count definition changes.

A successful nonfinal boundary call therefore finishes its GPU work before its
function autorelease pool drains and before Rust invokes the next tile. This
avoids allocating or CPU-initializing the next tile's scratch while this tile
is unfinished. No buffer aliases, explicit pools, callbacks or new queues are
introduced. Persistent KV/compressor state is unchanged.

The trace must independently verify the observed object-lifetime bound. The
source argument and a passing trace do not prove allocator reuse, a specific
physical-footprint cause, or acceptable engine performance. Full C0 and all
prospective engine/memory gates remain mandatory.
