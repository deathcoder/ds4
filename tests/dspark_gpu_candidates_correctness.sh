#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ds4_bin=${DS4_BIN:-"$root/ds4"}
base_model=${DS4_TEST_MODEL:-"$root/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"}
dspark_model=${DS4_TEST_DSPARK_MODEL:-"$root/gguf/ds4flash-dspark.gguf"}
mode=${DS4_TEST_DSPARK_MODE:-observer}

case "$mode" in
    observer) gpu_env=(DS4_DSPARK_GPU_CANDIDATES=1) ;;
    runtime) gpu_env=(DS4_DSPARK_GPU_RUNTIME=1) ;;
    *)
        printf 'invalid DS4_TEST_DSPARK_MODE: %s (expected observer or runtime)\n' "$mode" >&2
        exit 2
        ;;
esac

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
    if [[ $mode == observer ]]; then
        grep -q 'DSpark GPU chain parity .* result=pass' "$log"
    else
        grep -q 'DSpark GPU bridge runtime .* result=pass' "$log"
        grep -q 'DSpark GPU stage 0 runtime .* result=pass' "$log"
        grep -q 'DSpark GPU stage 1 runtime .* result=pass' "$log"
        grep -q 'DSpark GPU stage 2 runtime .* result=pass' "$log"
        grep -q 'DSpark GPU head runtime .* result=pass' "$log"
        grep -q 'DSpark GPU chain runtime .* result=pass' "$log"
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
compare_resumed_chat
if [[ $mode == observer ]]; then
    assert_strict_fallback "$root/tests/test-vectors/prompts/short_reasoning_plain.txt"
fi

printf 'DSpark GPU candidate correctness matrix (%s): PASS\n' "$mode"
