# Explicit initialized scratch reuse: ownership and scope proof

The new shim is an exactly reversible edit of the preserved bounded 1K shim.
The two Rust files are byte-identical to that prior candidate. All other39
runtime/shader source files must hash exactly as the accepted runtime at build
and execution. The prior width/ring/router changes retain their existing proofs.

Only native_1k_bootstrap creates an owner. Its inherited predicate requires
1024 rows, no output collection, complete layer1 and layer2 compressors,
consume-state mode, an aligned native4K chunk ending at8K..32K. The inherited
context guard and overall Rust selection remain unchanged. Nonselected calls
pass nil to the helper, which executes the same shared length/bytes constructor.

The source transformer enumerates and routes exactly the105 declared scratch
names. It expands scratch-only L1/L2 macro invocations; persistent invocations
and their conditional branches remain original. The16 context/transition aliases
are excluded. Inspection of context assignments in the boundary finds only the
excluded persistent buffers, scalar state and command objects escaping the
function. Scratch is referenced by completed encoders/commands and the dedicated
pool. It is never stored in a persistent activation/model cache or returned as
an MTLBuffer. Host outputs are copied synchronously after command completion;
those copies occur only on nonpooled collection paths.

Each chunk gets a new owner, fixed frontier and next-position cursor. It owns
one distinct new MTLBuffer per allowlisted name, never a heap subrange or a
cross-label alias. A checkout rejects unknown names, duplicates, zero/excess
length, capacity changes, excessive count/total, and an unfinished prior command.
The total cannot exceed105 logical buffers and2,493,173,233 bytes. Capacities
remain equal throughout each four-tile chunk; there is no growth/replacement.
The source retains the existing checked wait on every tile. In addition, entry
waits on the previous command before clearing the checkout set. Thus every CPU
initialization happens after prior GPU execution and completion handlers finish.
The command is assigned after commit, while all current local buffers remain
owned. The final command and batch errors retain their original checks.

For all102 length constructors, the helper clears every requested byte before
returning, including first checkout. Apple's newBufferWithLength contract is
zeroed storage. For the three byte constructors (tokens and two position arrays),
memcpy copies the complete requested range before returning, matching the original
copy constructor. Existing sentinel/raw-cache/mask/group-id initialization follows
unchanged. This establishes the same initial buffer state even for kernels that
leave holes; it does not assume full kernel writes. No shader, offset, CPU value,
encoding order, resource storage mode or arithmetic changes are introduced.

An Objective-C finally block clears the context owner on every in-scope failure
return/exception and every successful final tile. Successful nonfinal calls keep
the pool and advance its expected position. A nonnative call with an outstanding
owner fails and clears it. Existing release_prefill_layer_buffers also clears
this object property through its prefillLayer prefix; request reset calls that
function. Context destruction releases its strong property. Pre-try dimension/
null validation does not allocate or write scratch. A rejected malformed call
therefore cannot reuse an outstanding owner; later reset/destruction still clears
it. The normal Rust call sequence is serialized, as before; this does not add
thread-safe concurrent use of a context.

The actual GPU synthetic proof embeds the exact production class/helper. Across
seven four-tile chunks it validates distinct labels, identical per-label reuse,
full zero/copy contents, GPU readback, guard bytes, GPU poison and next-reset
contents, unknown/duplicate/changed-size/unfinished/zero/oversized rejection,
and the same finally cleanup condition. Tiny odd-length tails use CPU poison
and checking; aligned prefixes also use actual GPU fills/copies. This is a
bounded helper proof, not an end-to-end model error-injection campaign.

Pool-aware inference instrumentation emits numeric buffer/owner identities and
initialization metadata after checkout, never retaining either owner. The weak
ledger retains only metadata between calls. The production observer inverse is
exact. The prospective audit checks all28 tiles/2940 checkouts, fixed identities
within chunks, capacity, completed prior command, no prior-chunk scratch at new
entry, at most one unfinished command and context-owner removal after each final
tile. Weak final liveness is recorded separately; immediate Metal destruction is
not required. Logical length bounds are not claims about physical OS footprint.
Only full C0 and every unchanged engine gate can justify retaining this candidate.

Primary contracts verified on2026-09-12:
- https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28length%3Aoptions%3A%29?language=objc
- https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytes%3Alength%3Aoptions%3A%29?language=objc
- https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29?language=objc
