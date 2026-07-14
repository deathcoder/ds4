#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ds4_bin=${DS4_BIN:-"$root/ds4"}
base_model=${DS4_TEST_MODEL:-"$root/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"}
dspark_model=${DS4_TEST_DSPARK_MODEL:-"$root/gguf/ds4flash-dspark.gguf"}
unset DS4_DSPARK_EXACT_ATTN_PRE_BATCH

for path in "$ds4_bin" "$base_model" "$dspark_model"; do
    if [[ ! -f $path ]]; then
        printf 'missing required file: %s\n' "$path" >&2
        exit 2
    fi
done

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-exact-attn-pre-runtime.XXXXXX")
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

common=(--model "$base_model" --ctx 4096 --nothink --temp 0 --seed 1)
runtime_env=(
    DS4_DSPARK_GPU_RUNTIME=1
    DS4_DSPARK_MULTI_COMMIT=1
    DS4_DSPARK_FAST_BATCH_VERIFY=0
    DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1
)

assert_candidate_log() {
    local name=$1
    local log=$2
    local min_calls=$3
    local records
    records=$(grep 'DSpark exact attention pre batch runtime proposed=' "$log")
    if printf '%s\n' "$records" |
        grep -Evq ' layers=43 attempts=43 successes=43 result=pass$'; then
        printf 'exact attention-pre runtime drifted or fell back for %s\n' "$name" >&2
        printf '%s\n' "$records" >&2
        exit 1
    fi
    if [[ $(printf '%s\n' "$records" | wc -l | tr -d ' ') -lt $min_calls ]]; then
        printf 'exact attention-pre runtime did not run enough times for %s\n' "$name" >&2
        exit 1
    fi
    grep -q 'DSpark exact batch verifier .* result=pass' "$log"
    grep -q 'DSpark GPU candidate source selected' "$log"
    if grep -Eq 'DSpark exact attention pre batch runtime .* result=(fail|fallback)|DSpark batch capture skipped|DSpark GPU candidate source fallback' "$log"; then
        printf 'exact attention-pre runtime invariant failed for %s\n' "$name" >&2
        exit 1
    fi
}

compare_prompt() {
    local name=$1
    local tokens=$2
    local prompt_file=$3
    local min_calls=$4
    local baseline_out="$tmpdir/$name.baseline.out"
    local runtime_out="$tmpdir/$name.runtime.out"
    local baseline_log="$tmpdir/$name.baseline.log"
    local runtime_log="$tmpdir/$name.runtime.log"

    "$ds4_bin" "${common[@]}" -n "$tokens" --prompt-file "$prompt_file" \
        >"$baseline_out" 2>"$baseline_log"
    env "${runtime_env[@]}" \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n "$tokens" --prompt-file "$prompt_file" >"$runtime_out" 2>"$runtime_log"

    if ! cmp -s "$baseline_out" "$runtime_out"; then
        printf 'stdout mismatch for %s\n' "$name" >&2
        diff -u "$baseline_out" "$runtime_out" >&2 || true
        exit 1
    fi
    assert_candidate_log "$name" "$runtime_log" "$min_calls"
    printf 'PASS exact attention-pre runtime soak: %s\n' "$name"
}

compare_resumed_chat() {
    local baseline_out="$tmpdir/resumed.baseline.out"
    local runtime_out="$tmpdir/resumed.runtime.out"
    local baseline_log="$tmpdir/resumed.baseline.log"
    local runtime_log="$tmpdir/resumed.runtime.log"
    local input=$'Explain why the sky appears blue in one sentence.\nNow summarize that answer in five words.\n/exit\n'

    printf '%s' "$input" | "$ds4_bin" "${common[@]}" -n 12 \
        >"$baseline_out" 2>"$baseline_log"
    printf '%s' "$input" | env "${runtime_env[@]}" \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" -n 12 \
        >"$runtime_out" 2>"$runtime_log"

    if ! cmp -s "$baseline_out" "$runtime_out"; then
        printf 'stdout mismatch for resumed chat\n' >&2
        diff -u "$baseline_out" "$runtime_out" >&2 || true
        exit 1
    fi
    assert_candidate_log resumed_chat "$runtime_log" 2
    if ! awk '
        /DSpark exact attention pre batch runtime retained after resumed sync/ {
            resumed = 1
            next
        }
        resumed && /DSpark exact attention pre batch runtime .* result=pass/ {
            verified = 1
        }
        END { exit verified ? 0 : 1 }
    ' "$runtime_log"; then
        printf 'exact attention-pre runtime did not verify after resumed sync\n' >&2
        exit 1
    fi
    printf 'PASS exact attention-pre runtime soak: resumed_chat\n'
}

compare_prompt long_generation 64 "$root/speed-bench/dspark_prompt.txt" 5
compare_prompt rolling_window 64 "$root/tests/dspark_rolling_long_generation_prompt.txt" 5
compare_resumed_chat

printf 'DSpark exact attention-pre batch runtime correctness soak: PASS\n'
