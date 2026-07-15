# HumanEval acceptance pilot

This corpus freezes eight evenly spaced prompts from DeepSpec's checked-in
`eval_datasets/humaneval.jsonl` at commit
`005e03b81cec38b7da6399833d609ee89a2587f2`. DeepSpec generated that file from
the `openai/openai_humaneval` test split with its official HumanEval adapter.

`samples.jsonl` preserves the exact `turns[0]` strings used by DeepSpec.
`provenance.json` records the deterministic selection rule, upstream row and
line numbers, source-line hashes, and prompt byte hashes. The runner validates
all of these fields before materializing prompt files in the ignored run
directory; it does not add a trailing newline to the user content.

The pilot deliberately isolates corpus/domain before attempting a matched paper
reproduction. It uses non-thinking mode and no confidence scheduler, but keeps
the current V4-Flash protocol: the released five-token sidecar, greedy decoding,
and 128 output tokens. DeepSpec Table 1 used all 164 HumanEval samples, seven
draft tokens, temperature-1.0 rejection sampling, up to 2048 output tokens, and
Qwen3/Gemma4 checkpoints. Results are therefore directional acceptance evidence,
not a reproduction or a functional HumanEval score.

Do not edit `samples.jsonl`. Replace it only from a newly pinned upstream commit
and update every provenance field and hash together.
