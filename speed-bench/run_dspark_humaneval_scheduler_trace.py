#!/usr/bin/env python3
"""Collect per-proposal DSpark confidence traces on pinned HumanEval tasks."""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import analyze_dspark_confidence_scheduler as scheduler
import run_dspark_humaneval_acceptance as corpus
import run_dspark_humaneval_throughput as throughput
import run_dspark_issue468_comparison as common


DEEPSPEC_SCHEDULER_REFERENCE = {
    "repository": "https://github.com/deepseek-ai/DeepSpec",
    "commit": "005e03b81cec38b7da6399833d609ee89a2587f2",
    "file": "deepspec/eval/dspark/draft_ops.py",
    "policy": "prefix before first raw sigmoid confidence below threshold",
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect exact-runtime confidence traces using an existing "
            "HumanEval acceptance run as the correctness reference."
        )
    )
    parser.add_argument("--binary", type=Path, default=root / "ds4")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / (
            "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-"
            "SExpQ8-OutQ8-chat-v2-imatrix.gguf"
        ),
    )
    parser.add_argument(
        "--dspark-model",
        type=Path,
        default=root / "gguf/ds4flash-dspark.gguf",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "speed-bench/humaneval-acceptance",
    )
    parser.add_argument("--acceptance-reference", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0 or args.sample_count <= 0:
        parser.error("ctx, tokens, and sample-count must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to run the diagnostic without --confirm-ready")

    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = False
    args.stats_pass = False
    args.acceptance_audit = True
    args.acceptance_trace = True
    args.pairs = 1
    args.warmups = 0
    return args, root


def load_baseline_outputs(reference, labels):
    run_dir = reference["summary_path"].parent
    runs_path = run_dir / "runs.csv"
    try:
        with runs_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"cannot read acceptance reference runs: {exc}") from exc
    by_label = {}
    for label in labels:
        sample_rows = [row for row in rows if row.get("prompt") == label]
        baseline = [row for row in sample_rows if row.get("mode") == "baseline"]
        runtime = [row for row in sample_rows if row.get("mode") == "runtime"]
        if len(baseline) != 1 or len(runtime) != 1:
            raise SystemExit(f"acceptance reference run rows are incomplete for {label}")
        if baseline[0]["stdout_sha256"] != runtime[0]["stdout_sha256"]:
            raise SystemExit(f"acceptance reference output mismatch for {label}")
        path = (run_dir / baseline[0]["stdout_file"]).resolve()
        if path.parent != run_dir.resolve() or not path.is_file():
            raise SystemExit(f"invalid acceptance reference stdout for {label}")
        data = path.read_bytes()
        if common.sha256(data) != baseline[0]["stdout_sha256"]:
            raise SystemExit(f"acceptance reference stdout hash mismatch for {label}")
        by_label[label] = data
    return by_label


def validate_reproduced_acceptance(summary, reference):
    previous = reference["summary"]
    for label, item in summary["samples"].items():
        expected = previous["samples"][label]
        if item != expected:
            raise RuntimeError(
                f"traced acceptance metrics differ from prior audit for {label}"
            )
    if summary["aggregate"] != previous["aggregate"]:
        raise RuntimeError("traced aggregate acceptance differs from prior audit")


def render_trace_report(summary, reference):
    aggregate = summary["aggregate"]
    return "\n".join([
        "# DSpark HumanEval Confidence Trace",
        "",
        "Correctness diagnostic only. Throughput values are intentionally omitted.",
        "Each traced runtime output matched its previously validated baseline byte-for-byte.",
        "The aggregate acceptance audit also reproduced the reference exactly.",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Full proposals: {aggregate['proposals']}",
        f"- Block size: {aggregate['block_size']}",
        f"- Accepted length: {aggregate['paper_acceptance_length']:.3f}",
        f"- Verify rate: {aggregate['paper_verify_rate']:.3f}",
        f"- Acceptance reference: {reference['summary_path']}",
        "- Confidence values are raw sigmoid outputs without STS calibration.",
        "",
    ])


def main():
    args, root = parse_args()
    for name in (
        "binary", "model", "dspark_model", "corpus_dir",
        "acceptance_reference",
    ):
        setattr(args, name, getattr(args, name).resolve())
    all_records, provenance = corpus.load_corpus(args, root)
    records, selection = corpus.select_records(
        all_records, args.sample_count, provenance["selection_policy"]
    )
    reference = throughput.load_acceptance_reference(
        args, selection, provenance
    )
    labels = [record["label"] for record in records]
    baseline_outputs = load_baseline_outputs(reference, labels)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-scheduler-trace-"
        f"{args.sample_count}-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = corpus.prompt_paths(run_dir, records)
    for record in records:
        prompt = prompts[record["label"]]
        command = common.command_text(
            args, prompt, "runtime", acceptance_audit=True,
            acceptance_trace=True,
        )
        print(
            f"{record['label']} traced runtime: {command}"
        )
    print(
        f"HumanEval scheduler trace: {args.sample_count} exact-runtime processes; "
        "baseline outputs are reused from the validated acceptance reference."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    corpus.materialize_prompts(prompts, records)
    metadata = common.collect_metadata(
        args, root, prompts, provenance, acceptance_reference=None
    )
    metadata["experiment"] = "deepspec_humaneval_confidence_scheduler_trace"
    metadata["experiment_selection"] = selection
    metadata["deepspec_scheduler_reference"] = DEEPSPEC_SCHEDULER_REFERENCE
    metadata["acceptance_reference"] = {
        "summary": common.file_metadata(reference["summary_path"]),
        "metadata": common.file_metadata(reference["metadata_path"]),
        "baseline_outputs_reused": True,
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    runs = []
    audit_rows = []
    for record in records:
        label = record["label"]
        row, _ = common.execute(
            args, root, run_dir, "trace", label, prompts[label],
            "runtime", baseline_outputs[label], acceptance_audit=True,
            acceptance_trace=True,
        )
        runs.append(row)
        audit_rows.append(row)
        common.cooldown(args.cooldown)

    summary = corpus.summarize(
        audit_rows, records, provenance, selection, args.tokens
    )
    validate_reproduced_acceptance(summary, reference)
    summary["trace_reference"] = str(reference["summary_path"])
    trace_report = render_trace_report(summary, reference)
    corpus.write_outputs(run_dir, runs, records, summary, trace_report)
    common.finish_metadata(metadata, root, run_dir)
    _, scheduler_report = scheduler.analyze_run(run_dir)
    print("\n" + trace_report.rstrip())
    print(scheduler_report.rstrip())
    print(f"Raw trace: {run_dir / 'scheduler_trace.csv'}")
    print(f"Threshold sweep: {run_dir / 'scheduler_thresholds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
