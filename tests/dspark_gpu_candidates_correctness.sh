#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ds4_bin=${DS4_BIN:-"$root/ds4"}
base_model=${DS4_TEST_MODEL:-"$root/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"}
dspark_model=${DS4_TEST_DSPARK_MODEL:-"$root/gguf/ds4flash-dspark.gguf"}
mode=${DS4_TEST_DSPARK_MODE:-observer}
fast_verify_observer=${DS4_TEST_DSPARK_FAST_VERIFY_OBSERVER:-0}
fast_verify_runtime=${DS4_TEST_DSPARK_FAST_VERIFY_RUNTIME:-0}
serial_ffn_runtime=${DS4_TEST_DSPARK_SERIAL_FFN_RUNTIME:-0}
serial_attn_pre_runtime=${DS4_TEST_DSPARK_SERIAL_ATTN_PRE_RUNTIME:-0}
# Compatibility switch retained for Phase 0.68 commands; the path is now default.
attn_pre_runtime=${DS4_TEST_DSPARK_ATTN_PRE_RUNTIME:-0}
attn_suffix_runtime=${DS4_TEST_DSPARK_ATTN_SUFFIX_RUNTIME:-0}
runtime_stats=${DS4_TEST_DSPARK_RUNTIME_STATS:-0}
metal_drafter=${DS4_TEST_DSPARK_METAL_DRAFTER:-0}
exact_attn_row_views=${DS4_TEST_DSPARK_EXACT_ATTN_ROW_VIEWS:-0}
exact_attn_out_nr4=${DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR4:-default}
exact_attn_out_nr8=${DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR8:-0}
acceptance_audit=${DS4_TEST_DSPARK_ACCEPTANCE_AUDIT:-0}
acceptance_trace=${DS4_TEST_DSPARK_ACCEPTANCE_TRACE:-0}
confidence_threshold=${DS4_TEST_DSPARK_CONFIDENCE_THRESHOLD:-}
default_confidence_threshold=0.455
confidence_expect_zero=${DS4_TEST_DSPARK_CONFIDENCE_EXPECT_ZERO:-0}
compressor_pair_nr4=${DS4_TEST_DSPARK_COMPRESSOR_PAIR_NR4:-0}
indexed_attn_rb16_promotion=${DS4_TEST_DSPARK_INDEXED_ATTN_RB16_PROMOTION:-\
${DS4_TEST_DSPARK_INDEXED_ATTN_RB16_DIRECT:-0}}
dense_mixed_direct=${DS4_TEST_DSPARK_DENSE_MIXED_DIRECT:-0}
dense_mixed_gathered_legacy=${DS4_TEST_DSPARK_DENSE_MIXED_GATHERED_LEGACY:-0}
exact_q8_rows=${DS4_TEST_DSPARK_EXACT_Q8_ROWS:-0}
serial_q8_rows_runtime=${DS4_TEST_DSPARK_SERIAL_Q8_ROWS_RUNTIME:-0}
attn_inv_rope_fused=${DS4_TEST_DSPARK_ATTN_INV_ROPE_FUSED:-0}
exact_prefix_checkpoint=${DS4_TEST_DSPARK_EXACT_PREFIX_CHECKPOINT:-default}
exact_q2_down_batch=${DS4_TEST_DSPARK_EXACT_Q2_DOWN_BATCH:-0}
exact_shared_q8_rows=${DS4_TEST_DSPARK_EXACT_SHARED_Q8_ROWS:-0}
ffn_batch_observer_layer=${DS4_DSPARK_EXACT_FFN_BATCH_OBSERVER_LAYER:-}
attn_pre_observer_layer=${DS4_DSPARK_EXACT_ATTN_PRE_BATCH_OBSERVER_LAYER:-}
attn_suffix_observer_layer=${DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH_OBSERVER_LAYER:-}
unset DS4_DSPARK_EXACT_FFN_BATCH
unset DS4_DSPARK_EXACT_ATTN_PRE_BATCH
unset DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH
unset DS4_DSPARK_EXACT_Q8_ROWS
unset DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED
unset DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED_TRACE
unset DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED_OBSERVER_LAYER
unset DS4_DSPARK_EXACT_PREFIX_CHECKPOINT
unset DS4_DSPARK_EXACT_Q2_DOWN_BATCH
unset DS4_DSPARK_EXACT_Q2_DOWN_BATCH_TRACE
unset DS4_DSPARK_EXACT_SHARED_Q8_ROWS
unset DS4_DSPARK_EXACT_SHARED_Q8_ROWS_TRACE
unset DS4_DSPARK_ACCEPTANCE_AUDIT
unset DS4_DSPARK_ACCEPTANCE_TRACE
unset DS4_DSPARK_CONFIDENCE_THRESHOLD
unset DS4_DSPARK_METAL_DRAFTER
unset DS4_DSPARK_EXACT_ATTN_ROW_VIEWS
unset DS4_DSPARK_EXACT_ATTN_OUT_NR4
unset DS4_DSPARK_EXACT_ATTN_OUT_NR4_TRACE
unset DS4_DSPARK_EXACT_ATTN_OUT_NR8
unset DS4_DSPARK_EXACT_ATTN_OUT_NR8_TRACE
unset DS4_METAL_COMPRESSOR_PAIR_NR4
unset DS4_METAL_INDEXED_ATTN_RB16_DIRECT
unset DS4_METAL_INDEXED_ATTN_RB16_LEGACY
unset DS4_METAL_INDEXED_ATTN_RB16_DIRECT_TRACE
unset DS4_METAL_DENSE_MIXED_DIRECT
unset DS4_METAL_DENSE_MIXED_GATHERED_LEGACY
unset DS4_METAL_DENSE_MIXED_DIRECT_TRACE
unset DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD

if [[ $fast_verify_observer == 1 && $fast_verify_runtime == 1 ]]; then
    printf 'fast verifier observer and runtime modes are mutually exclusive\n' >&2
    exit 2
fi
if [[ $serial_ffn_runtime == 1 && -n $ffn_batch_observer_layer ]]; then
    printf 'serial FFN runtime and selected-layer observer are mutually exclusive\n' >&2
    exit 2
fi
observer_count=0
[[ -n $ffn_batch_observer_layer ]] && ((observer_count += 1))
[[ -n $attn_pre_observer_layer ]] && ((observer_count += 1))
[[ -n $attn_suffix_observer_layer ]] && ((observer_count += 1))
if [[ $observer_count -gt 1 ]]; then
    printf 'selected-layer observers are mutually exclusive\n' >&2
    exit 2
fi
if [[ $attn_suffix_runtime == 1 && $observer_count -ne 0 ]]; then
    printf 'attention-suffix runtime and selected-layer observers are mutually exclusive\n' >&2
    exit 2
fi
if [[ $attn_pre_runtime == 1 && -n $attn_pre_observer_layer ]]; then
    printf 'attention-pre runtime and observer are mutually exclusive\n' >&2
    exit 2
fi
if [[ $serial_attn_pre_runtime == 1 && -n $attn_pre_observer_layer ]]; then
    printf 'serial attention-pre runtime and observer are mutually exclusive\n' >&2
    exit 2
fi
if [[ $serial_attn_pre_runtime == 1 && $attn_pre_runtime == 1 ]]; then
    printf 'serial and required attention-pre runtime are mutually exclusive\n' >&2
    exit 2
fi
if [[ ($attn_pre_runtime == 1 || $serial_attn_pre_runtime == 1) &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'attention-pre controls require exact runtime verification\n' >&2
    exit 2
fi
if [[ $attn_suffix_runtime == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_attn_pre_runtime == 1) ]]; then
    printf 'attention-suffix runtime requires exact verification with attention-pre batching\n' >&2
    exit 2
fi
if [[ $runtime_stats == 1 && $mode != runtime ]]; then
    printf 'runtime stats require DS4_TEST_DSPARK_MODE=runtime\n' >&2
    exit 2
fi
if [[ $metal_drafter != 0 && $metal_drafter != 1 ]]; then
    printf 'DS4_TEST_DSPARK_METAL_DRAFTER must be 0 or 1\n' >&2
    exit 2
fi
if [[ $metal_drafter == 1 && $mode != runtime ]]; then
    printf 'Metal drafter requires DS4_TEST_DSPARK_MODE=runtime\n' >&2
    exit 2
fi
if [[ $exact_attn_row_views != 0 && $exact_attn_row_views != 1 ]]; then
    printf 'DS4_TEST_DSPARK_EXACT_ATTN_ROW_VIEWS must be 0 or 1\n' >&2
    exit 2
fi
if [[ $exact_attn_row_views == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_attn_pre_runtime == 1) ]]; then
    printf 'exact attention row views require exact attention-pre runtime\n' >&2
    exit 2
fi
case "$exact_attn_out_nr4" in
    default|0|1) ;;
    *)
        printf 'DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR4 must be default, 0, or 1\n' >&2
        exit 2
        ;;
esac
if [[ $exact_attn_out_nr4 != default &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_attn_pre_runtime == 1) ]]; then
    printf 'exact attention-output NR control requires exact attention-pre runtime\n' >&2
    exit 2
fi
if [[ $exact_attn_out_nr8 != 0 && $exact_attn_out_nr8 != 1 ]]; then
    printf 'DS4_TEST_DSPARK_EXACT_ATTN_OUT_NR8 must be 0 or 1\n' >&2
    exit 2
fi
if [[ $exact_attn_out_nr8 == 1 && $exact_attn_out_nr4 != default ]]; then
    printf 'exact attention-output NR8 and explicit NR4 controls are mutually exclusive\n' >&2
    exit 2
fi
if [[ $exact_attn_out_nr8 == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_attn_pre_runtime == 1) ]]; then
    printf 'exact attention-output NR8 requires exact attention-pre runtime\n' >&2
    exit 2
fi
if [[ $acceptance_audit == 1 && $mode != runtime ]]; then
    printf 'acceptance audit requires DS4_TEST_DSPARK_MODE=runtime\n' >&2
    exit 2
fi
if [[ $acceptance_trace == 1 && $acceptance_audit != 1 ]]; then
    printf 'acceptance trace requires DS4_TEST_DSPARK_ACCEPTANCE_AUDIT=1\n' >&2
    exit 2
fi
if [[ -n $confidence_threshold && $mode != runtime ]]; then
    printf 'confidence scheduler requires DS4_TEST_DSPARK_MODE=runtime\n' >&2
    exit 2
fi
if [[ $confidence_expect_zero == 1 && -z $confidence_threshold ]]; then
    printf 'zero-prefix expectation requires a confidence threshold\n' >&2
    exit 2
fi
if [[ $compressor_pair_nr4 != 0 && $compressor_pair_nr4 != 1 ]]; then
    printf 'DS4_TEST_DSPARK_COMPRESSOR_PAIR_NR4 must be 0 or 1\n' >&2
    exit 2
fi
if [[ $compressor_pair_nr4 == 1 && $mode != runtime ]]; then
    printf 'compressor pair NR4 requires DS4_TEST_DSPARK_MODE=runtime\n' >&2
    exit 2
fi
if [[ $indexed_attn_rb16_promotion != 0 && $indexed_attn_rb16_promotion != 1 ]]; then
    printf 'DS4_TEST_DSPARK_INDEXED_ATTN_RB16_PROMOTION must be 0 or 1\n' >&2
    exit 2
fi
if [[ $indexed_attn_rb16_promotion == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'indexed-attention RB16 promotion requires exact runtime verification\n' >&2
    exit 2
fi
if [[ $dense_mixed_direct != 0 && $dense_mixed_direct != 1 ]]; then
    printf 'DS4_TEST_DSPARK_DENSE_MIXED_DIRECT must be 0 or 1\n' >&2
    exit 2
fi
if [[ $dense_mixed_gathered_legacy != 0 &&
      $dense_mixed_gathered_legacy != 1 ]]; then
    printf 'DS4_TEST_DSPARK_DENSE_MIXED_GATHERED_LEGACY must be 0 or 1\n' >&2
    exit 2
fi
if [[ $dense_mixed_direct == 1 && $dense_mixed_gathered_legacy == 1 ]]; then
    printf 'dense-mixed fused and legacy gathered controls are mutually exclusive\n' >&2
    exit 2
fi
if [[ $dense_mixed_direct == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'dense-mixed fused gather requires exact runtime verification\n' >&2
    exit 2
fi
if [[ $dense_mixed_gathered_legacy == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'legacy dense-mixed gather requires exact runtime verification\n' >&2
    exit 2
fi
if [[ $exact_q8_rows != 0 && $exact_q8_rows != 1 ]]; then
    printf 'DS4_TEST_DSPARK_EXACT_Q8_ROWS must be 0 or 1\n' >&2
    exit 2
fi
if [[ $serial_q8_rows_runtime != 0 && $serial_q8_rows_runtime != 1 ]]; then
    printf 'DS4_TEST_DSPARK_SERIAL_Q8_ROWS_RUNTIME must be 0 or 1\n' >&2
    exit 2
fi
if [[ $exact_q8_rows == 1 && $serial_q8_rows_runtime == 1 ]]; then
    printf 'exact and serial Q8-row modes are mutually exclusive\n' >&2
    exit 2
fi
if [[ $exact_q8_rows == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_attn_pre_runtime == 1) ]]; then
    printf 'exact Q8 rows require exact runtime attention-pre batching\n' >&2
    exit 2
fi
if [[ $serial_q8_rows_runtime == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_attn_pre_runtime == 1 || -n $attn_pre_observer_layer) ]]; then
    printf 'serial Q8 rows require exact runtime attention-pre batching\n' >&2
    exit 2
fi
if [[ $attn_inv_rope_fused != 0 && $attn_inv_rope_fused != 1 ]]; then
    printf 'DS4_TEST_DSPARK_ATTN_INV_ROPE_FUSED must be 0 or 1\n' >&2
    exit 2
fi
if [[ $attn_inv_rope_fused == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $attn_suffix_runtime == 1) ]]; then
    printf 'fused attention inverse-RoPE requires the retained exact runtime\n' >&2
    exit 2
fi
if [[ $exact_prefix_checkpoint != default &&
      $exact_prefix_checkpoint != 0 &&
      $exact_prefix_checkpoint != 1 ]]; then
    printf 'DS4_TEST_DSPARK_EXACT_PREFIX_CHECKPOINT must be default, 0, or 1\n' >&2
    exit 2
fi
if [[ $exact_prefix_checkpoint == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'exact prefix checkpoints require exact runtime verification\n' >&2
    exit 2
fi
if [[ $exact_q2_down_batch != 0 && $exact_q2_down_batch != 1 ]]; then
    printf 'DS4_TEST_DSPARK_EXACT_Q2_DOWN_BATCH must be 0 or 1\n' >&2
    exit 2
fi
if [[ $exact_q2_down_batch == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'exact Q2 down batching requires exact runtime verification\n' >&2
    exit 2
fi
if [[ $exact_shared_q8_rows != 0 && $exact_shared_q8_rows != 1 ]]; then
    printf 'DS4_TEST_DSPARK_EXACT_SHARED_Q8_ROWS must be 0 or 1\n' >&2
    exit 2
fi
if [[ $exact_shared_q8_rows == 1 &&
      ($mode != runtime || $fast_verify_runtime == 1 ||
       $serial_ffn_runtime == 1) ]]; then
    printf 'exact shared Q8 rows require batched exact FFN verification\n' >&2
    exit 2
fi
if [[ -n $attn_suffix_observer_layer &&
      ($mode != runtime || $fast_verify_runtime == 1) ]]; then
    printf 'attention-suffix observer requires exact runtime verification\n' >&2
    exit 2
fi

case "$mode" in
    observer) gpu_env=(DS4_DSPARK_GPU_CANDIDATES=1) ;;
    runtime)
        gpu_env=(
            DS4_DSPARK_GPU_RUNTIME=1
            DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1
        )
        if [[ $fast_verify_observer == 1 ]]; then
            gpu_env+=(DS4_DSPARK_FAST_VERIFY_OBSERVER=1)
        fi
        if [[ $fast_verify_runtime == 1 ]]; then
            gpu_env+=(DS4_DSPARK_FAST_BATCH_VERIFY=1)
        fi
        if [[ $serial_ffn_runtime == 1 ]]; then
            gpu_env+=(DS4_DSPARK_EXACT_FFN_BATCH=0)
        fi
        if [[ $serial_attn_pre_runtime == 1 ]]; then
            gpu_env+=(DS4_DSPARK_EXACT_ATTN_PRE_BATCH=0)
        fi
        if [[ $attn_suffix_runtime == 1 ]]; then
            gpu_env+=(DS4_DSPARK_EXACT_ATTN_SUFFIX_BATCH=1)
        fi
        if [[ $runtime_stats == 1 ]]; then
            gpu_env+=(DS4_DSPARK_GPU_RUNTIME_STATS=1)
        fi
        if [[ $metal_drafter == 1 ]]; then
            gpu_env+=(DS4_DSPARK_METAL_DRAFTER=1)
        fi
        if [[ $exact_attn_row_views == 1 ]]; then
            gpu_env+=(DS4_DSPARK_EXACT_ATTN_ROW_VIEWS=1)
        fi
        case "$exact_attn_out_nr4" in
            0) gpu_env+=(DS4_DSPARK_EXACT_ATTN_OUT_NR4=0) ;;
            1)
                gpu_env+=(
                    DS4_DSPARK_EXACT_ATTN_OUT_NR4=1
                    DS4_DSPARK_EXACT_ATTN_OUT_NR4_TRACE=1
                )
                ;;
        esac
        if [[ $exact_attn_out_nr8 == 1 ]]; then
            gpu_env+=(
                DS4_DSPARK_EXACT_ATTN_OUT_NR8=1
                DS4_DSPARK_EXACT_ATTN_OUT_NR8_TRACE=1
            )
        fi
        if [[ $acceptance_audit == 1 ]]; then
            gpu_env+=(DS4_DSPARK_ACCEPTANCE_AUDIT=1)
        fi
        if [[ $acceptance_trace == 1 ]]; then
            gpu_env+=(DS4_DSPARK_ACCEPTANCE_TRACE=1)
        fi
        if [[ -n $confidence_threshold ]]; then
            gpu_env+=(DS4_DSPARK_CONFIDENCE_THRESHOLD="$confidence_threshold")
        fi
        if [[ $compressor_pair_nr4 == 1 ]]; then
            gpu_env+=(DS4_METAL_COMPRESSOR_PAIR_NR4=1)
        fi
        if [[ $indexed_attn_rb16_promotion == 1 ]]; then
            gpu_env+=(
                DS4_METAL_INDEXED_ATTN_RB16_DIRECT_TRACE=1
                DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=64
            )
        fi
        if [[ $dense_mixed_direct == 1 ]]; then
            gpu_env+=(
                DS4_METAL_DENSE_MIXED_DIRECT=1
                DS4_METAL_DENSE_MIXED_DIRECT_TRACE=1
            )
        fi
        if [[ $dense_mixed_gathered_legacy == 1 ]]; then
            gpu_env+=(DS4_METAL_DENSE_MIXED_GATHERED_LEGACY=1)
        fi
        if [[ $serial_q8_rows_runtime == 1 ]]; then
            gpu_env+=(DS4_DSPARK_EXACT_Q8_ROWS=0)
        fi
        if [[ $attn_inv_rope_fused == 1 ]]; then
            gpu_env+=(
                DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED=1
                DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED_TRACE=1
            )
        fi
        case "$exact_prefix_checkpoint" in
            0) gpu_env+=(DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=0) ;;
            1) gpu_env+=(DS4_DSPARK_EXACT_PREFIX_CHECKPOINT=1) ;;
        esac
        if [[ $exact_q2_down_batch == 1 ]]; then
            gpu_env+=(
                DS4_DSPARK_EXACT_Q2_DOWN_BATCH=1
                DS4_DSPARK_EXACT_Q2_DOWN_BATCH_TRACE=1
            )
        fi
        if [[ $exact_shared_q8_rows == 1 ]]; then
            gpu_env+=(
                DS4_DSPARK_EXACT_SHARED_Q8_ROWS=1
                DS4_DSPARK_EXACT_SHARED_Q8_ROWS_TRACE=1
            )
        fi
        ;;
    *)
        printf 'invalid DS4_TEST_DSPARK_MODE: %s (expected observer or runtime)\n' "$mode" >&2
        exit 2
        ;;
esac

exact_ffn_expected=0
exact_ffn_allowed=0
if [[ $mode == runtime && $serial_ffn_runtime != 1 &&
      -z $ffn_batch_observer_layer ]]; then
    exact_ffn_allowed=1
    if [[ $fast_verify_runtime != 1 ]]; then
        exact_ffn_expected=1
    fi
fi

exact_attn_suffix_expected=0
if [[ $mode == runtime && $attn_suffix_runtime == 1 &&
      $fast_verify_runtime != 1 ]]; then
    exact_attn_suffix_expected=1
fi

exact_attn_pre_allowed=0
exact_attn_pre_expected=0
if [[ $mode == runtime && $serial_attn_pre_runtime != 1 &&
      -z $attn_pre_observer_layer ]]; then
    exact_attn_pre_allowed=1
    if [[ $fast_verify_runtime != 1 ]]; then
        exact_attn_pre_expected=1
    fi
fi

exact_q8_rows_expected=0
if [[ $mode == runtime && $fast_verify_runtime != 1 &&
      $serial_attn_pre_runtime != 1 && $serial_q8_rows_runtime != 1 &&
      -z $attn_pre_observer_layer ]]; then
    exact_q8_rows_expected=1
fi

for path in "$ds4_bin" "$base_model" "$dspark_model"; do
    if [[ ! -f "$path" ]]; then
        printf 'missing required file: %s\n' "$path" >&2
        exit 2
    fi
done

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dspark-correctness.XXXXXX")
cleanup() {
    local status=$?
    if [[ ${DS4_TEST_KEEP_LOGS:-0} == 1 || $status -ne 0 ]]; then
        printf 'logs retained in %s\n' "$tmpdir" >&2
    else
        rm -rf "$tmpdir"
    fi
    exit "$status"
}
trap cleanup EXIT

common=(--model "$base_model" --ctx 4096 --nothink --temp 0)

assert_gpu_selected() {
    local log=$1
    local verifier_batches_expected=0
    local effective_confidence_threshold=${confidence_threshold:-$default_confidence_threshold}
    if [[ $mode == runtime ]]; then
        if [[ $fast_verify_runtime == 1 ]] &&
           grep -q 'DSpark fast batch verifier .* result=pass' "$log"; then
            verifier_batches_expected=1
        elif [[ $fast_verify_runtime != 1 ]] &&
             grep -q 'DSpark exact batch verifier .* result=pass' "$log"; then
            verifier_batches_expected=1
        fi
    fi
    if [[ $mode == observer ]]; then
        grep -q 'DSpark GPU chain parity .* result=pass' "$log"
    else
        grep -q 'DSpark GPU bridge runtime .* result=pass' "$log"
        grep -q 'DSpark GPU stage 0 runtime .* result=pass' "$log"
        grep -q 'DSpark GPU stage 1 runtime .* result=pass' "$log"
        grep -q 'DSpark GPU stage 2 runtime .* result=pass' "$log"
        grep -q 'DSpark GPU head runtime .* result=pass' "$log"
        grep -q 'DSpark GPU chain runtime .* result=pass' "$log"
        if [[ $metal_drafter == 1 ]]; then
            grep -q 'DSpark Metal drafter cache .* result=pass' "$log"
            grep -q 'DSpark Metal drafter proposal .* result=pass' "$log"
            if grep -q 'DSpark Metal drafter failed at ' "$log"; then
                printf 'Metal drafter unexpectedly fell back\n' >&2
                exit 1
            fi
        elif grep -q 'DSpark Metal drafter proposal ' "$log"; then
            printf 'control mode unexpectedly ran the Metal drafter\n' >&2
            exit 1
        fi
        if [[ $fast_verify_runtime == 1 ]]; then
            grep -q 'DSpark fast batch verifier .* result=pass' "$log"
        elif [[ $verifier_batches_expected == 1 ]]; then
            grep -q 'DSpark exact batch verifier .* result=pass' "$log"
        fi
        grep -q \
            "DSpark confidence scheduler threshold=$effective_confidence_threshold" \
            "$log"
        if grep 'DSpark confidence scheduler ' "$log" |
           grep -Evq ' proposed=[1-9][0-9]* selected=[0-9][0-9]*$'; then
            printf 'confidence scheduler emitted a malformed route record\n' >&2
            exit 1
        fi
        if [[ $fast_verify_observer == 1 ]]; then
            grep -q 'DSpark fast verifier observer ' "$log"
            if grep -q 'DSpark fast verifier observer .* result=fail' "$log"; then
                printf 'fast verifier observer reported a parity failure\n' >&2
                exit 1
            fi
        fi
        if [[ $exact_ffn_allowed == 1 ]] &&
           grep -q 'DSpark exact FFN batch runtime .* result=fail' "$log"; then
            printf 'exact FFN batch runtime reported a failure\n' >&2
            exit 1
        fi
        if [[ $exact_ffn_expected == 1 && $verifier_batches_expected == 1 ]]; then
            grep -q 'DSpark exact FFN batch runtime .* result=pass' "$log"
        fi
        if [[ $exact_attn_pre_allowed == 1 ]] &&
           grep -q 'DSpark exact attention pre batch runtime proposed=' "$log"; then
            local attn_runtime_records
            attn_runtime_records=$(grep 'DSpark exact attention pre batch runtime proposed=' "$log")
            if printf '%s\n' "$attn_runtime_records" |
                grep -Evq ' layers=43 attempts=43 successes=43 result=pass$'; then
                printf 'exact attention-pre runtime drifted or fell back\n' >&2
                printf '%s\n' "$attn_runtime_records" >&2
                exit 1
            fi
        elif [[ $exact_attn_pre_expected == 1 && $verifier_batches_expected == 1 ]]; then
            printf 'default exact runtime omitted attention-pre batching\n' >&2
            exit 1
        elif [[ $exact_attn_pre_allowed != 1 ]] &&
             grep -q 'DSpark exact attention pre batch runtime ' "$log"; then
            printf 'control mode unexpectedly ran exact attention-pre batching\n' >&2
            exit 1
        fi
        if [[ $exact_attn_row_views == 1 &&
              $verifier_batches_expected == 1 ]]; then
            local row_view_records
            row_view_records=$(grep \
                'DSpark exact attention row views proposed=' "$log")
            if [[ -z $row_view_records ]] ||
               printf '%s\n' "$row_view_records" |
                   grep -Evq \
                       ' proposed=[2-9][0-9]* views=[1-9][0-9]* uses=[1-9][0-9]*/[1-9][0-9]* result=pass$'; then
                printf 'exact attention row-view cache drifted or fell back\n' >&2
                printf '%s\n' "$row_view_records" >&2
                exit 1
            fi
        elif grep -q 'DSpark exact attention row views proposed=' "$log"; then
            printf 'control mode unexpectedly cached exact attention row views\n' >&2
            exit 1
        fi
        if [[ $exact_attn_suffix_expected == 1 ]]; then
            local suffix_runtime_records
            suffix_runtime_records=$(grep \
                'DSpark exact attention suffix batch runtime proposed=' "$log")
            if printf '%s\n' "$suffix_runtime_records" |
                grep -Evq ' layers=43 attempts=43 successes=43 result=pass$'; then
                printf 'exact attention-suffix runtime drifted or fell back\n' >&2
                printf '%s\n' "$suffix_runtime_records" >&2
                exit 1
            fi
        elif grep -q 'DSpark exact attention suffix batch runtime ' "$log"; then
            printf 'control mode unexpectedly ran exact attention-suffix batching\n' >&2
            exit 1
        fi
        if [[ $exact_q8_rows_expected == 1 && $verifier_batches_expected == 1 ]]; then
            local q8_records
            q8_records=$(grep 'DSpark exact Q8 rows runtime proposed=' "$log")
            if printf '%s\n' "$q8_records" |
                grep -Evq ' projections=129 attempts=129 successes=129 result=pass$'; then
                printf 'exact Q8 rows drifted or fell back\n' >&2
                printf '%s\n' "$q8_records" >&2
                exit 1
            fi
        elif grep -q 'DSpark exact Q8 rows runtime ' "$log"; then
            printf 'control mode unexpectedly ran exact Q8 rows\n' >&2
            exit 1
        fi
        if [[ $exact_q2_down_batch == 1 &&
              $verifier_batches_expected == 1 ]]; then
            local q2_down_records
            q2_down_records=$(grep 'DSpark exact Q2 down batch layer=' "$log")
            if [[ -z $q2_down_records ]] ||
               printf '%s\n' "$q2_down_records" |
                   grep -Evq ' layer=[0-9][0-9]* width=[2-5] result=pass$'; then
                printf 'exact Q2 down batch drifted or failed\n' >&2
                printf '%s\n' "$q2_down_records" >&2
                exit 1
            fi
        elif grep -q 'DSpark exact Q2 down batch layer=' "$log"; then
            printf 'control mode unexpectedly ran exact Q2 down batching\n' >&2
            exit 1
        fi
        if [[ $exact_shared_q8_rows == 1 &&
              $verifier_batches_expected == 1 ]]; then
            local shared_q8_records shared_q8_gate shared_q8_down
            shared_q8_records=$(grep \
                'DSpark exact shared Q8 rows layer=' "$log")
            if [[ -z $shared_q8_records ]] ||
               printf '%s\n' "$shared_q8_records" |
                   grep -Evq \
                       ' layer=[0-9][0-9]* width=[2-5] stage=(gate_up|down) result=pass$'; then
                printf 'exact shared Q8 rows drifted or failed\n' >&2
                printf '%s\n' "$shared_q8_records" >&2
                exit 1
            fi
            shared_q8_gate=$(printf '%s\n' "$shared_q8_records" |
                grep -c ' stage=gate_up result=pass$')
            shared_q8_down=$(printf '%s\n' "$shared_q8_records" |
                grep -c ' stage=down result=pass$')
            if [[ $shared_q8_gate -eq 0 ||
                  $shared_q8_gate -ne $shared_q8_down ]]; then
                printf 'exact shared Q8 row stages were incomplete: %s/%s\n' \
                    "$shared_q8_gate" "$shared_q8_down" >&2
                exit 1
            fi
        elif grep -q 'DSpark exact shared Q8 rows layer=' "$log"; then
            printf 'control mode unexpectedly ran exact shared Q8 rows\n' >&2
            exit 1
        fi
        if [[ $attn_inv_rope_fused == 1 ]]; then
            grep -q 'DSpark exact attention inverse-RoPE fused route=raw ' "$log"
            grep -q 'DSpark exact attention inverse-RoPE fused route=dense_mixed ' "$log"
        elif grep -q 'DSpark exact attention inverse-RoPE fused route=' "$log"; then
            printf 'control mode unexpectedly ran fused attention inverse-RoPE\n' >&2
            exit 1
        fi
        if [[ $runtime_stats == 1 ]]; then
            local stats_record outcomes attempts successes
            local attn_outcomes attn_attempts attn_successes
            local suffix_outcomes suffix_attempts suffix_successes
            local checkpoint_outcomes checkpoint_attempts checkpoint_successes
            local checkpoint_fallbacks checkpoint_rows_avoided
            local metal_outcomes metal_attempts metal_successes metal_fallbacks
            if [[ $(grep -c '^ds4: DSpark runtime stats ' "$log") -ne 1 ]]; then
                printf 'expected exactly one DSpark runtime stats record\n' >&2
                exit 1
            fi
            stats_record=$(grep '^ds4: DSpark runtime stats ' "$log")
            outcomes=$(printf '%s\n' "$stats_record" | sed -n \
                's/.* exact_ffn_batch_attempts=\([0-9][0-9]*\) exact_ffn_batch_successes=\([0-9][0-9]*\) .*/\1 \2/p')
            if [[ -z $outcomes ]]; then
                printf 'DSpark runtime stats omitted exact FFN outcomes\n' >&2
                exit 1
            fi
            read -r attempts successes <<<"$outcomes"
            if [[ $exact_ffn_allowed == 1 ]]; then
                if [[ $successes -ne $attempts ||
                      ($exact_ffn_expected == 1 && $attempts -eq 0) ]]; then
                    printf 'exact FFN runtime outcomes were %s/%s successful\n' \
                        "$successes" "$attempts" >&2
                    exit 1
                fi
            elif [[ $attempts -ne 0 || $successes -ne 0 ]]; then
                printf 'control mode unexpectedly recorded exact FFN outcomes\n' >&2
                exit 1
            fi
            checkpoint_outcomes=$(printf '%s\n' "$stats_record" | sed -n \
                's/.* prefix_checkpoint_attempts=\([0-9][0-9]*\) prefix_checkpoint_successes=\([0-9][0-9]*\) prefix_checkpoint_fallbacks=\([0-9][0-9]*\) prefix_checkpoint_rows_avoided=\([0-9][0-9]*\) .*/\1 \2 \3 \4/p')
            if [[ -z $checkpoint_outcomes ]]; then
                printf 'DSpark runtime stats omitted prefix-checkpoint outcomes\n' >&2
                exit 1
            fi
            read -r checkpoint_attempts checkpoint_successes \
                checkpoint_fallbacks checkpoint_rows_avoided \
                <<<"$checkpoint_outcomes"
            if [[ $checkpoint_successes -gt $checkpoint_attempts ||
                  $checkpoint_fallbacks -ne \
                      $((checkpoint_attempts - checkpoint_successes)) ||
                  ($checkpoint_successes -eq 0 &&
                   $checkpoint_rows_avoided -ne 0) ]]; then
                printf 'invalid prefix-checkpoint runtime outcomes: %s\n' \
                    "$checkpoint_outcomes" >&2
                exit 1
            fi
            attn_outcomes=$(printf '%s\n' "$stats_record" | sed -n \
                's/.* exact_attn_pre_batch_attempts=\([0-9][0-9]*\) exact_attn_pre_batch_successes=\([0-9][0-9]*\) .*/\1 \2/p')
            if [[ -z $attn_outcomes ]]; then
                printf 'DSpark runtime stats omitted exact attention-pre outcomes\n' >&2
                exit 1
            fi
            read -r attn_attempts attn_successes <<<"$attn_outcomes"
            if [[ $exact_attn_pre_allowed == 1 ]]; then
                if [[ $attn_successes -ne $attn_attempts ||
                      ($exact_attn_pre_expected == 1 && $attn_attempts -eq 0) ]]; then
                    printf 'exact attention-pre outcomes were %s/%s successful\n' \
                        "$attn_successes" "$attn_attempts" >&2
                    exit 1
                fi
            elif [[ $attn_attempts -ne 0 || $attn_successes -ne 0 ]]; then
                printf 'control mode unexpectedly recorded attention-pre outcomes\n' >&2
                exit 1
            fi
            suffix_outcomes=$(printf '%s\n' "$stats_record" | sed -n \
                's/.* exact_attn_suffix_batch_attempts=\([0-9][0-9]*\) exact_attn_suffix_batch_successes=\([0-9][0-9]*\) .*/\1 \2/p')
            if [[ -z $suffix_outcomes ]]; then
                printf 'DSpark runtime stats omitted exact attention-suffix outcomes\n' >&2
                exit 1
            fi
            read -r suffix_attempts suffix_successes <<<"$suffix_outcomes"
            if [[ $exact_attn_suffix_expected == 1 ]]; then
                if [[ $suffix_successes -ne $suffix_attempts ||
                      $suffix_attempts -eq 0 ]]; then
                    printf 'exact attention-suffix outcomes were %s/%s successful\n' \
                        "$suffix_successes" "$suffix_attempts" >&2
                    exit 1
                fi
            elif [[ $suffix_attempts -ne 0 || $suffix_successes -ne 0 ]]; then
                printf 'control mode unexpectedly recorded attention-suffix outcomes\n' >&2
                exit 1
            fi
            metal_outcomes=$(printf '%s\n' "$stats_record" | sed -n \
                's/.* metal_drafter_attempts=\([0-9][0-9]*\) metal_drafter_successes=\([0-9][0-9]*\) metal_drafter_fallbacks=\([0-9][0-9]*\) .*/\1 \2 \3/p')
            if [[ -z $metal_outcomes ]]; then
                printf 'DSpark runtime stats omitted Metal drafter outcomes\n' >&2
                exit 1
            fi
            read -r metal_attempts metal_successes metal_fallbacks \
                <<<"$metal_outcomes"
            if [[ $metal_drafter == 1 ]]; then
                if [[ $metal_attempts -eq 0 ||
                      $metal_successes -ne $metal_attempts ||
                      $metal_fallbacks -ne 0 ]]; then
                    printf 'Metal drafter runtime outcomes were %s\n' \
                        "$metal_outcomes" >&2
                    exit 1
                fi
            elif [[ $metal_attempts -ne 0 || $metal_successes -ne 0 ||
                    $metal_fallbacks -ne 0 ]]; then
                printf 'control mode unexpectedly recorded Metal drafter outcomes\n' >&2
                exit 1
            fi
        fi
        if [[ $acceptance_audit == 1 ]]; then
            local acceptance_record
            if [[ $(grep -c '^ds4: DSpark acceptance audit ' "$log") -ne 1 ]]; then
                printf 'expected exactly one DSpark acceptance audit record\n' >&2
                exit 1
            fi
            acceptance_record=$(grep '^ds4: DSpark acceptance audit ' "$log")
            local field
            for field in block_size proposals paper_acceptance_sum \
                         truncated_proposals proposed_at reached_at accepted_at \
                         rejected_at confidence_sum prefix_confidence_sum; do
                if ! grep -Eq "(^| )${field}=[0-9.,]+( |$)" \
                    <<<"$acceptance_record"; then
                    printf 'DSpark acceptance audit record omitted %s\n' "$field" >&2
                    exit 1
                fi
            done
            if [[ $acceptance_trace == 1 ]]; then
                local trace_records expected_trace_records proposals truncated
                trace_records=$(grep -c '^ds4: DSpark acceptance trace ' "$log")
                proposals=$(sed -n 's/.* proposals=\([0-9][0-9]*\) .*/\1/p' \
                    <<<"$acceptance_record")
                truncated=$(sed -n 's/.* truncated_proposals=\([0-9][0-9]*\) .*/\1/p' \
                    <<<"$acceptance_record")
                expected_trace_records=$((proposals + truncated))
                if [[ $trace_records -ne $expected_trace_records ]]; then
                    printf 'acceptance trace records were %s, expected %s\n' \
                        "$trace_records" "$expected_trace_records" >&2
                    exit 1
                fi
                if grep '^ds4: DSpark acceptance trace ' "$log" | \
                   grep -Eiq 'confidences=.*(nan|inf)'; then
                    printf 'acceptance trace contains non-finite confidence values\n' >&2
                    exit 1
                fi
            elif grep -q '^ds4: DSpark acceptance trace ' "$log"; then
                printf 'non-trace audit unexpectedly emitted proposal records\n' >&2
                exit 1
            fi
        elif grep -q '^ds4: DSpark acceptance audit ' "$log"; then
            printf 'control mode unexpectedly emitted acceptance audit data\n' >&2
            exit 1
        elif grep -q '^ds4: DSpark acceptance trace ' "$log"; then
            printf 'control mode unexpectedly emitted acceptance trace data\n' >&2
            exit 1
        fi
        if [[ -n $ffn_batch_observer_layer ]]; then
            local observer_records
            observer_records=$(grep "DSpark exact FFN batch observer layer=$ffn_batch_observer_layer " "$log")
            if printf '%s\n' "$observer_records" | grep -Evq ' first=none .* result=exact$'; then
                printf 'exact FFN batch observer drifted or fell back at layer %s\n' "$ffn_batch_observer_layer" >&2
                printf '%s\n' "$observer_records" >&2
                exit 1
            fi
        fi
        if [[ -n $attn_pre_observer_layer ]]; then
            local attn_records
            attn_records=$(grep "DSpark exact attention pre batch observer layer=$attn_pre_observer_layer " "$log")
            if printf '%s\n' "$attn_records" |
                grep -Evq ' first=none.* q_rope_max=0.* kv_rope_max=0.* result=exact$'; then
                printf 'exact attention-pre batch observer drifted or fell back at layer %s\n' \
                    "$attn_pre_observer_layer" >&2
                printf '%s\n' "$attn_records" >&2
                exit 1
            fi
        fi
        if [[ -n $attn_suffix_observer_layer ]]; then
            local suffix_records
            suffix_records=$(grep \
                "DSpark exact attention suffix batch observer layer=$attn_suffix_observer_layer " \
                "$log")
            if printf '%s\n' "$suffix_records" | grep -q 'shadow-fallback'; then
                printf 'exact attention-suffix batch observer fell back at layer %s\n' \
                    "$attn_suffix_observer_layer" >&2
                printf '%s\n' "$suffix_records" >&2
                exit 1
            fi
            if printf '%s\n' "$suffix_records" |
                grep -Evq ' first=none low_max=0 low_rms=0 attn_out_max=0 attn_out_rms=0 exact_rows=[0-9]+/[0-9]+ hc_max=0 hc_rms=0 result=exact$'; then
                printf 'attention-suffix observer drifted at layer %s\n' \
                    "$attn_suffix_observer_layer" >&2
                printf '%s\n' "$suffix_records" >&2
                exit 1
            fi
        fi
        if grep -q 'DSpark GPU .* parity' "$log"; then
            printf 'runtime mode unexpectedly ran a GPU parity observer\n' >&2
            exit 1
        fi
    fi
    grep -q 'DSpark GPU candidate source selected' "$log"
}

compare_prompt_file() {
    local name=$1
    local tokens=$2
    local prompt_file=$3
    local baseline_out="$tmpdir/$name.baseline.out"
    local gpu_out="$tmpdir/$name.gpu.out"
    local baseline_log="$tmpdir/$name.baseline.log"
    local gpu_log="$tmpdir/$name.gpu.log"

    "$ds4_bin" "${common[@]}" -n "$tokens" --prompt-file "$prompt_file" \
        >"$baseline_out" 2>"$baseline_log"
    env "${gpu_env[@]}" DS4_DSPARK_MULTI_COMMIT=1 \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n "$tokens" --prompt-file "$prompt_file" >"$gpu_out" 2>"$gpu_log"

    if ! cmp -s "$baseline_out" "$gpu_out"; then
        printf 'stdout mismatch for %s\n' "$name" >&2
        diff -u "$baseline_out" "$gpu_out" >&2 || true
        exit 1
    fi
    assert_gpu_selected "$gpu_log"
    printf 'PASS prompt: %s\n' "$name"
}

compare_resumed_chat() {
    local baseline_out="$tmpdir/chat.baseline.out"
    local gpu_out="$tmpdir/chat.gpu.out"
    local baseline_log="$tmpdir/chat.baseline.log"
    local gpu_log="$tmpdir/chat.gpu.log"
    local input=$'Hello\nTell me one word\n/exit\n'

    printf '%s' "$input" | "$ds4_bin" "${common[@]}" -n 2 \
        >"$baseline_out" 2>"$baseline_log"
    printf '%s' "$input" | \
        env "${gpu_env[@]}" DS4_DSPARK_MULTI_COMMIT=1 \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" -n 2 \
        >"$gpu_out" 2>"$gpu_log"

    if ! cmp -s "$baseline_out" "$gpu_out"; then
        printf 'stdout mismatch for resumed chat\n' >&2
        diff -u "$baseline_out" "$gpu_out" >&2 || true
        exit 1
    fi
    assert_gpu_selected "$gpu_log"
    if [[ $(grep -c 'DSpark GPU candidate source selected' "$gpu_log") -lt 2 ]]; then
        printf 'resumed chat did not select GPU proposals more than once\n' >&2
        exit 1
    fi
    printf 'PASS chat: resumed two-turn session\n'
}

compare_indexed_attn_rb16_promotion() {
    local prompt_file="$tmpdir/rb16-direct.prompt"
    local baseline_out="$tmpdir/rb16-direct.baseline.out"
    local control_out="$tmpdir/rb16-direct.control.out"
    local candidate_out="$tmpdir/rb16-direct.candidate.out"
    local control_log="$tmpdir/rb16-direct.control.log"
    local candidate_log="$tmpdir/rb16-direct.candidate.log"

    for _ in {1..15}; do
        awk '1' "$root/tests/dspark_rolling_window_prompt.txt"
    done >"$prompt_file"

    "$ds4_bin" "${common[@]}" -n 6 --prompt-file "$prompt_file" \
        >"$baseline_out" 2>/dev/null
    env \
        DS4_DSPARK_GPU_RUNTIME=1 \
        DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1 \
        DS4_DSPARK_MULTI_COMMIT=1 \
        DS4_METAL_INDEXED_ATTN_RB16_LEGACY=1 \
        DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=64 \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n 6 --prompt-file "$prompt_file" >"$control_out" 2>"$control_log"
    env "${gpu_env[@]}" DS4_DSPARK_MULTI_COMMIT=1 \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n 6 --prompt-file "$prompt_file" >"$candidate_out" 2>"$candidate_log"

    cmp -s "$baseline_out" "$control_out"
    cmp -s "$baseline_out" "$candidate_out"
    cmp -s "$control_out" "$candidate_out"
    if grep -q 'Metal indexed attention route=rb16_direct' "$control_log"; then
        printf 'legacy RB16 control unexpectedly selected RB16-direct\n' >&2
        exit 1
    fi
    grep -q 'Metal indexed attention route=rb16_direct' "$candidate_log"
    assert_gpu_selected "$control_log"
    assert_gpu_selected "$candidate_log"
    printf 'PASS indexed attention: promoted RB16-direct versus legacy RB16\n'
}

compare_attn_inv_rope_fused_indexed() {
    local prompt_file="$tmpdir/attn-inv-rope-indexed.prompt"
    local baseline_out="$tmpdir/attn-inv-rope-indexed.baseline.out"
    local candidate_out="$tmpdir/attn-inv-rope-indexed.candidate.out"
    local candidate_log="$tmpdir/attn-inv-rope-indexed.candidate.log"

    for _ in {1..15}; do
        awk '1' "$root/tests/dspark_rolling_window_prompt.txt"
    done >"$prompt_file"

    "$ds4_bin" "${common[@]}" -n 6 --prompt-file "$prompt_file" \
        >"$baseline_out" 2>/dev/null
    env "${gpu_env[@]}" \
        DS4_DSPARK_MULTI_COMMIT=1 \
        DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED_OBSERVER_LAYER=42 \
        DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=64 \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n 6 --prompt-file "$prompt_file" \
        >"$candidate_out" 2>"$candidate_log"

    cmp -s "$baseline_out" "$candidate_out"
    grep -q 'DSpark exact attention inverse-RoPE fused route=sparse_indexed ' \
        "$candidate_log"
    grep -q 'DSpark exact attention inverse-RoPE fused observer route=sparse_indexed ' \
        "$candidate_log"
    if grep -q 'DSpark exact attention inverse-RoPE fused observer .* result=drift' \
        "$candidate_log"; then
        printf 'fused inverse-RoPE indexed observer exceeded its numerical bound\n' >&2
        exit 1
    fi
    assert_gpu_selected "$candidate_log"
    printf 'PASS attention: fused inverse-RoPE indexed route\n'
}

observe_attn_inv_rope_fused_layer() {
    local name=$1
    local layer=$2
    local route=$3
    local baseline_out="$tmpdir/reasoning.baseline.out"
    local candidate_out="$tmpdir/attn-inv-rope-$name.out"
    local candidate_log="$tmpdir/attn-inv-rope-$name.log"

    env "${gpu_env[@]}" \
        DS4_DSPARK_MULTI_COMMIT=1 \
        DS4_DSPARK_EXACT_ATTN_INV_ROPE_FUSED_OBSERVER_LAYER="$layer" \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n 4 \
        --prompt-file "$root/tests/test-vectors/prompts/short_reasoning_plain.txt" \
        >"$candidate_out" 2>"$candidate_log"

    cmp -s "$baseline_out" "$candidate_out"
    grep -q "DSpark exact attention inverse-RoPE fused observer route=$route layer=$layer " \
        "$candidate_log"
    if grep -q 'DSpark exact attention inverse-RoPE fused observer .* result=drift' \
        "$candidate_log"; then
        printf 'fused inverse-RoPE %s observer exceeded its numerical bound\n' \
            "$route" >&2
        exit 1
    fi
    assert_gpu_selected "$candidate_log"
    printf 'PASS attention: fused inverse-RoPE %s observer\n' "$route"
}

assert_strict_fallback() {
    local prompt_file=$1
    local baseline_out="$tmpdir/strict.baseline.out"
    local gpu_out="$tmpdir/strict.gpu.out"
    local gpu_log="$tmpdir/strict.gpu.log"

    "$ds4_bin" "${common[@]}" -n 2 --prompt-file "$prompt_file" \
        >"$baseline_out" 2>/dev/null
    DS4_DSPARK_GPU_CANDIDATES=1 \
    DS4_DSPARK_GPU_CONFIDENCE_TOLERANCE=1e-7 \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n 2 --prompt-file "$prompt_file" >"$gpu_out" 2>"$gpu_log"

    cmp -s "$baseline_out" "$gpu_out"
    grep -q 'DSpark GPU chain parity .* result=fail' "$gpu_log"
    grep -q 'DSpark GPU candidate source fallback: self-fed GPU block failed parity' "$gpu_log"
    if grep -q 'DSpark GPU candidate source selected' "$gpu_log"; then
        printf 'strict fallback unexpectedly selected GPU proposals\n' >&2
        exit 1
    fi
    printf 'PASS fallback: strict parity rejection\n'
}

compare_prompt_file reasoning 4 "$root/tests/test-vectors/prompts/short_reasoning_plain.txt"
compare_prompt_file italian 6 "$root/tests/test-vectors/prompts/short_italian_fact.txt"
compare_prompt_file medium_context 6 "$root/tests/dspark_gpu_candidates_medium_prompt.txt"
compare_prompt_file rolling_window 12 "$root/tests/dspark_rolling_window_prompt.txt"
compare_resumed_chat
if [[ $exact_attn_out_nr4 == 1 ]]; then
    if ! grep -q 'Metal exact attention output NR4 projection=A' "$tmpdir"/*.log ||
       ! grep -q 'Metal exact attention output NR4 projection=B+HC' "$tmpdir"/*.log; then
        printf 'exact attention-output NR4 did not engage both projections\n' >&2
        exit 1
    fi
fi
if [[ $exact_attn_out_nr8 == 1 ]]; then
    if ! grep -q 'Metal exact attention output NR8 projection=A' "$tmpdir"/*.log ||
       ! grep -q 'Metal exact attention output NR8 projection=B+HC' "$tmpdir"/*.log; then
        printf 'exact attention-output NR8 did not engage both projections\n' >&2
        exit 1
    fi
fi
if [[ $confidence_expect_zero == 1 ]] &&
   ! grep -q 'DSpark confidence scheduler .* selected=0$' "$tmpdir"/*.log; then
    printf 'confidence scheduler did not exercise the zero-draft fallback\n' >&2
    exit 1
fi
if [[ $indexed_attn_rb16_promotion == 1 ]]; then
    compare_indexed_attn_rb16_promotion
fi
if [[ $dense_mixed_direct == 1 && $attn_inv_rope_fused != 1 ]] &&
   ! grep -q 'Metal dense mixed attention route=fused_gather' "$tmpdir"/*.log; then
    printf 'dense-mixed fused gather did not engage\n' >&2
    exit 1
fi
if [[ $attn_inv_rope_fused == 1 ]]; then
    observe_attn_inv_rope_fused_layer raw 0 raw
    observe_attn_inv_rope_fused_layer dense 42 dense_mixed
    compare_attn_inv_rope_fused_indexed
fi
if [[ $mode == runtime && $exact_prefix_checkpoint == 1 ]] &&
   ! grep -q 'DSpark exact prefix checkpoint .* result=pass' "$tmpdir"/*.log; then
    printf 'exact prefix checkpoint did not engage on the correctness matrix\n' >&2
    exit 1
fi
if [[ $mode == observer ]]; then
    assert_strict_fallback "$root/tests/test-vectors/prompts/short_reasoning_plain.txt"
fi

printf 'DSpark GPU candidate correctness matrix (%s): PASS\n' "$mode"
