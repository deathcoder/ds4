# HumanEval acceptance corpus

This corpus freezes all 164 prompts from DeepSpec's checked-in
`eval_datasets/humaneval.jsonl` at commit
`005e03b81cec38b7da6399833d609ee89a2587f2`. DeepSpec generated that file from
the `openai/openai_humaneval` test split with its official HumanEval adapter.

`samples.jsonl` preserves the exact `turns[0]` strings used by DeepSpec.
`provenance.json` records the full upstream file hash, every row and line number,
source-line hashes, prompt byte hashes, and the deterministic selection policy.
The runner validates all of these fields before selecting or materializing
prompts in the ignored run directory; it does not add a trailing newline to the
user content.

The runner defaults to 32 evenly spaced samples. `--sample-count 8` reproduces
the original pilot indices `0/23/47/70/93/116/140/163`, while
`--sample-count 164` processes every row in source order. Every selection is
recorded in metadata and the summary.

The study isolates corpus/domain before attempting a matched paper reproduction.
It uses non-thinking mode and explicitly sets confidence threshold `0` for its
fixed-K=5 control, while keeping the current
V4-Flash protocol: the released five-token sidecar, greedy decoding, and 128
output tokens. DeepSpec Table 1 used seven draft tokens, temperature-1.0
rejection sampling, up to 2048 output tokens, and Qwen3/Gemma4 checkpoints.
Results are therefore directional acceptance evidence, not a reproduction or a
functional HumanEval score.

After acceptance is established, `run_dspark_humaneval_throughput.py` reuses
the same deterministic selection and requires the acceptance artifact as a
validated protocol reference. It measures uninstrumented paired throughput;
it does not rerun or mix acceptance logging into the timed processes.

Do not edit `samples.jsonl`. Replace it only from a newly pinned upstream commit
and update every provenance field and hash together.
