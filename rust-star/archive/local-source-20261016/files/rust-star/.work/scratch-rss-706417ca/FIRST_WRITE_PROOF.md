# Full-copy destination proof

This proof applies only to the native 1024-row branch in the frozen rejected
scratch-pool source, SHA-256 `a41d4f3cfd9323ae8ed8702c3c10ef8ba700ba9aa3c3ab51eaa391541d98cf59`.
The accepted runtime has not been edited. See `source-proof.json` for every
object reference and `source-proof.py` for the reproducible lexical checks.

| Destination | Source | Full-copy source line | Bytes |
|---|---|---:|---:|
| `q_cur_buffer` | `q_raw_buffer` | 6221 | 134,217,728 |
| `attention_back_buffer` | `attention_output_buffer` | 6334 | 134,217,728 |
| `next_q_cur_buffer` | `next_q_raw_buffer` | 6643 | 134,217,728 |
| `next_attention_back_buffer` | `next_attention_output_buffer` | 6756 | 134,217,728 |

Each allocation and copy uses `q_bytes = rows*q_dim*sizeof(float)` with rows
1024 and q_dim 32768: 134,217,728 bytes. Every copy begins at source and
destination offset zero and covers the entire logical destination. Total:
536,870,912 bytes. Allocation success checks are not content reads.

All destination references were reviewed. Before the copy there is no content
read, kernel binding, copy-source use or alias assignment. The producer compute
encoder ends before the blit; the blit encoder ends before the first kernel
binding that reads or mutates the destination. Layer-1 copies are inside
complete_layer1, which the native 1K branch requires. Encoder failures return
without submitting a consumer. Preserve these paths and the existing checked
prior-command completion before scratch reuse. Keep distinct source/destination
identities, exact lengths and final/error owner cleanup.

After a complete copy, every destination byte equals the corresponding source
byte independently of the destination's previous contents. Therefore removing
only these destination clears cannot change the bytes supplied to subsequent
unchanged consumers. This reasoning does not require a new claim that the
source producers fully write their buffers: preserve those source initializers
and all other 101 initializations. Do not generalize to other sizes or paths.
The source proof is bounded review plus reproducible reference checks, not a
formal proof of every Metal or engine behavior.

The GPU check verifies two opposing poisons against a zeroed destination,
all copied bytes and prefix/suffix guards at sizes 4, 4096 and 134217728,
with destination reuse across two seeds. All 24 cases, 2,147,549,248 payload
bytes and 24,576 guard bytes pass. All 12 shortened-copy controls are detected.
However, the planned in-place mutation after copying is absent. Re-poisoning
before the next copy does not satisfy that condition. The full prospective
protocol is incomplete; `decision.json` overrides the original audit's positive
selection. The missing test belongs before any next integration. No model/C0
claim is made by this synthetic check.
