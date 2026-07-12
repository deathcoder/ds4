#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ds4_bin=${DS4_BIN:-"$root/ds4"}
base_model=${DS4_TEST_MODEL:-"$root/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"}
dspark_model=${DS4_TEST_DSPARK_MODEL:-"$root/gguf/ds4flash-dspark.gguf"}

for path in "$ds4_bin" "$base_model" "$dspark_model"; do
    if [[ ! -f $path ]]; then
        printf 'missing required file: %s\n' "$path" >&2
        exit 2
    fi
done

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dspark-fast-soak.XXXXXX")
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
fast_env=(
    DS4_DSPARK_GPU_RUNTIME=1
    DS4_DSPARK_MULTI_COMMIT=1
    DS4_DSPARK_FAST_BATCH_VERIFY=1
    DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS=1
)

assert_fast_log() {
    local name=$1
    local log=$2
    local require_fast=$3
    case "$require_fast" in
        yes) grep -q 'DSpark fast batch verifier .* result=pass' "$log" ;;
        margin) grep -q 'DSpark multi commit fallback .* first row margin' "$log" ;;
        no) ;;
        *) printf 'invalid fast expectation for %s: %s\n' "$name" "$require_fast" >&2; exit 2 ;;
    esac
    grep -q 'DSpark GPU candidate source selected' "$log"
    if grep -Eq 'DSpark fast (batch|prefix) verifier failed|DSpark batch capture skipped|prefix tokens .* exceed sidecar window' "$log"; then
        printf 'fast verifier invariant failed for %s\n' "$name" >&2
        exit 1
    fi
}

compare_prompt() {
    local name=$1
    local tokens=$2
    local prompt_file=$3
    local require_fast=$4
    shift 4
    local baseline_out="$tmpdir/$name.baseline.out"
    local fast_out="$tmpdir/$name.fast.out"
    local baseline_log="$tmpdir/$name.baseline.log"
    local fast_log="$tmpdir/$name.fast.log"

    "$ds4_bin" "${common[@]}" -n "$tokens" --prompt-file "$prompt_file" \
        >"$baseline_out" 2>"$baseline_log"
    env "${fast_env[@]}" "$@" \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" \
        -n "$tokens" --prompt-file "$prompt_file" >"$fast_out" 2>"$fast_log"

    if ! cmp -s "$baseline_out" "$fast_out"; then
        printf 'stdout mismatch for %s\n' "$name" >&2
        diff -u "$baseline_out" "$fast_out" >&2 || true
        exit 1
    fi
    assert_fast_log "$name" "$fast_log" "$require_fast"
    printf 'PASS fast soak: %s\n' "$name"
}

compare_resumed_chat() {
    local baseline_out="$tmpdir/resumed.baseline.out"
    local fast_out="$tmpdir/resumed.fast.out"
    local baseline_log="$tmpdir/resumed.baseline.log"
    local fast_log="$tmpdir/resumed.fast.log"
    local input=$'Explain why the sky appears blue in one sentence.\nNow summarize that answer in five words.\n/exit\n'

    printf '%s' "$input" | "$ds4_bin" "${common[@]}" -n 12 \
        >"$baseline_out" 2>"$baseline_log"
    printf '%s' "$input" | env "${fast_env[@]}" \
        "$ds4_bin" "${common[@]}" --dspark "$dspark_model" -n 12 \
        >"$fast_out" 2>"$fast_log"

    if ! cmp -s "$baseline_out" "$fast_out"; then
        printf 'stdout mismatch for resumed chat\n' >&2
        diff -u "$baseline_out" "$fast_out" >&2 || true
        exit 1
    fi
    assert_fast_log resumed "$fast_log" yes
    if [[ $(grep -c 'DSpark fast batch verifier .* result=pass' "$fast_log") -lt 1 ]] ||
       ! grep -q 'DSpark fast batch verifier suspended after resumed sync' "$fast_log" ||
       ! grep -q 'DSpark exact batch verifier .* result=pass' "$fast_log"; then
        printf 'resumed chat did not transition from fast to exact verification\n' >&2
        exit 1
    fi
    printf 'PASS fast soak: resumed_chat\n'
}

compare_prompt long_generation 64 "$root/speed-bench/dspark_prompt.txt" yes
compare_prompt code_completion 32 "$root/tests/test-vectors/prompts/short_code_completion.txt" no
compare_prompt italian 32 "$root/tests/test-vectors/prompts/short_italian_fact.txt" yes
compare_prompt spanish 24 "$root/tests/dspark_fast_spanish_prompt.txt" yes
compare_prompt structured_json 24 "$root/tests/dspark_fast_structured_prompt.txt" yes
compare_prompt near_window 6 "$root/tests/dspark_gpu_candidates_medium_prompt.txt" yes
compare_prompt margin_gate 24 "$root/tests/test-vectors/prompts/short_reasoning_plain.txt" margin \
    DS4_DSPARK_VERIFY_MIN_MARGIN=100
compare_resumed_chat

printf 'DSpark fast verifier correctness soak: PASS\n'
