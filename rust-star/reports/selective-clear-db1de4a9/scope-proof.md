# Selective scratch initialization scope

The mutation proof passes before any runtime edit: full-copy destinations begin
with opposing poisons or zeros, receive a full blit, then a read/modify/write
XOR operation and all-byte verification. Prefix/suffix guards remain unchanged.
All 24 cases and 2,147,549,248 payload bytes pass. The verifier detects all12
shortened copies and all12 deliberately omitted mutations. The earlier
scratch-rss report remains incomplete under its original prospective protocol;
this new proof does not rewrite that historical verdict.

Relative to the rejected explicit pool, the sole behavioral edit is the helper
clear condition. Only q_cur_buffer, attention_back_buffer, next_q_cur_buffer
and next_attention_back_buffer with length134217728 and source=NULL in an
active pool skip CPU memset. Nonpool constructors and source memcpy remain
unchanged. Every other initialization, shader and dispatch stays identical.
The helper embedded in pool-proof.m is byte-for-byte production code. Its
selected-size test proves reused poison survives omitted clears and explicit
source input still copies every byte; existing small-size tests cover all105
labels with complete initialization, guards, invalid checkout and owner cleanup.

Source references in source-proof.json show the first content access to each
selected destination is a full q_bytes blit with zero offsets. For native rows
1024 and q_dim32768, q_bytes equals allocation length134217728. Producers end
before copies and copies end before consumers, including in-place RoPE kernels.
Layer-1 work is required by the native guard. Encoder failures return without
submitting a consumer. No earlier content read or alias exists in the reviewed
reference inventory. Therefore destination prior bytes are irrelevant to all
subsequent unchanged operations; source buffers and other101 initializers stay
unchanged. The inherited checked prior-command completion still precedes every
checkout. Shape or ownership failures still return errors. Chunk owner,105
unique capacities,2,493,173,233-byte maximum and final/error cleanup are unchanged.

The observed executable adds CPU metadata only, including an elided flag for
every checkout. Reversing recorded observer edits restores production exactly.
Its auditor prospectively expects the four elisions per tile,105 stable distinct
identities, checked prior status4 (NSUIntegerMax only before the first command),
exact per-frontier capacity and owner cleanup. Full C0 and an unchanged fixed
engine protocol remain required. Proofs alone do not retain this optimization.
