#!/usr/bin/env python3
"""Audit exact DSpark acceptance on a frozen DeepSpec HumanEval pilot."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import run_dspark_issue468_comparison as common


HUMANEVAL_CODE_POSITION = 1


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the frozen DeepSpec HumanEval DSpark acceptance pilot."
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
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-ready", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.ctx <= 0 or args.tokens <= 0:
        parser.error("ctx and tokens must be positive")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if not args.dry_run and not args.confirm_ready:
        parser.error("refusing to run the diagnostic without --confirm-ready")

    # Attributes consumed by the shared exact-audit execution helpers.
    args.nothink = True
    args.fast_verifier = False
    args.exact_head_batch = False
    args.stats_only = False
    args.stats_pass = False
    args.acceptance_audit = True
    args.acceptance_reference = None
    args.pairs = 1
    args.warmups = 0
    return args, root


def load_corpus(args, root):
    provenance_path = args.corpus_dir / "provenance.json"
    samples_path = args.corpus_dir / "samples.jsonl"
    for label, path in (
        ("binary", args.binary),
        ("base model", args.model),
        ("DSpark model", args.dspark_model),
        ("provenance", provenance_path),
        ("samples", samples_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not os.access(args.binary, os.X_OK):
        raise SystemExit(f"binary is not executable: {args.binary}")
    dirty = common.git_output(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree changes detected; commit them or pass "
            f"--allow-dirty:\n{dirty}"
        )

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid HumanEval provenance: {exc}") from exc
    samples_data = samples_path.read_bytes()
    if (
        len(samples_data) != provenance.get("samples_file_bytes")
        or common.sha256(samples_data) != provenance.get("samples_file_sha256")
    ):
        raise SystemExit(f"HumanEval samples-file provenance mismatch: {samples_path}")

    records = []
    try:
        for line_number, line in enumerate(samples_data.splitlines(), start=1):
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"invalid HumanEval samples JSONL near line {line_number}: {exc}"
        ) from exc

    expected = provenance.get("samples", [])
    if len(records) != provenance.get("selection", {}).get("sample_count"):
        raise SystemExit("HumanEval sample count does not match selection metadata")
    if len(records) != len(expected):
        raise SystemExit("HumanEval sample count does not match provenance entries")
    selected_indices = provenance.get("selection", {}).get("indices_zero_based")
    if [record.get("source_index") for record in records] != selected_indices:
        raise SystemExit("HumanEval samples do not match the recorded selection")

    labels = set()
    for record, item in zip(records, expected):
        label = record.get("label")
        source_index = record.get("source_index")
        turns = record.get("turns")
        if (
            label != item.get("label")
            or source_index != item.get("source_index")
        ):
            raise SystemExit("HumanEval sample order or identity mismatch")
        if label in labels:
            raise SystemExit(f"duplicate HumanEval label: {label}")
        labels.add(label)
        if not (
            isinstance(turns, list)
            and len(turns) == 1
            and isinstance(turns[0], str)
        ):
            raise SystemExit(f"invalid HumanEval turns for {label}")
        prompt_data = turns[0].encode("utf-8")
        if (
            len(prompt_data) != item.get("prompt_bytes")
            or common.sha256(prompt_data) != item.get("prompt_sha256")
        ):
            raise SystemExit(f"HumanEval prompt provenance mismatch for {label}")
    return records, provenance


def prompt_paths(run_dir, records):
    return {
        record["label"]: run_dir / "prompts" / f"{record['label']}.txt"
        for record in records
    }


def materialize_prompts(paths, records):
    next(iter(paths.values())).parent.mkdir(parents=True)
    for record in records:
        paths[record["label"]].write_bytes(record["turns"][0].encode("utf-8"))


def combine_audits(audits):
    if not audits:
        raise RuntimeError("cannot aggregate an empty HumanEval audit")
    block_size = audits[0]["block_size"]
    if any(audit["block_size"] != block_size for audit in audits):
        raise RuntimeError("HumanEval audit block sizes differ")
    combined = {"block_size": block_size}
    for field in common.ACCEPTANCE_INT_FIELDS:
        if field != "block_size":
            combined[field] = sum(audit[field] for audit in audits)
    for field in common.ACCEPTANCE_INT_ARRAY_FIELDS:
        combined[field] = [
            sum(audit[field][pos] for audit in audits)
            for pos in range(block_size)
        ]
    for field in common.ACCEPTANCE_FLOAT_ARRAY_FIELDS:
        combined[field] = [
            sum(audit[field][pos] for audit in audits)
            for pos in range(block_size)
        ]
    return combined


def acceptance_metrics(label, audit):
    row = {"prompt": label, "acceptance_audit": audit}
    return common.summarize_acceptance(
        [row], nothink=True
    )["prompts"][label]


def humaneval_reference():
    values = {
        model: domains["code"][HUMANEVAL_CODE_POSITION]
        for model, domains in common.PAPER_DSPARK_TABLE1.items()
    }
    return {
        "source": "DSpark paper arXiv:2607.05147v1, Table 1, HumanEval",
        "accepted_length_by_target": values,
        "accepted_length_minimum": min(values.values()),
        "accepted_length_maximum": max(values.values()),
        "accepted_length_mean": sum(values.values()) / len(values),
        "verify_rate_minimum": min(values.values()) / 8.0,
        "verify_rate_maximum": max(values.values()) / 8.0,
        "verify_rate_mean": (sum(values.values()) / len(values)) / 8.0,
    }


def summarize(rows, records, provenance, tokens):
    by_label = {row["prompt"]: row["acceptance_audit"] for row in rows}
    samples = {}
    for record in records:
        label = record["label"]
        samples[label] = {
            "source_index": record["source_index"],
            **acceptance_metrics(label, by_label[label]),
        }
    aggregate_audit = combine_audits(list(by_label.values()))
    return {
        "dataset": "HumanEval",
        "sample_count": len(records),
        "samples": samples,
        "aggregate": acceptance_metrics("aggregate", aggregate_audit),
        "official_reference": humaneval_reference(),
        "protocol": {
            "source_repository": provenance["source_repository"],
            "source_commit": provenance["source_commit"],
            "source_file": provenance["source_file"],
            "selection": provenance["selection"],
            "exact_deepspec_turn_content": True,
            "non_thinking": True,
            "confidence_scheduler": False,
            "temperature": 0.0,
            "seed": 1,
            "draft_tokens": aggregate_audit["block_size"],
            "max_new_tokens": tokens,
            "matched_table1_reproduction": False,
        },
    }


def render_report(summary):
    aggregate = summary["aggregate"]
    reference = summary["official_reference"]
    tokens = summary["protocol"]["max_new_tokens"]
    lines = [
        "# DSpark HumanEval Acceptance Pilot",
        "",
        "Correctness diagnostic only. Throughput values are intentionally omitted.",
        "Every exact DSpark output matched its fresh non-thinking baseline byte-for-byte.",
        "Aggregate values pool proposal rounds across all selected HumanEval samples.",
        "",
        "| samples | proposals | drafts/proposal | accepted drafts/proposal | paper accept_len | verify rate | full accept |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | {aggregate['proposals']} | "
        f"{aggregate['draft_tokens_per_proposal']:.3f} | "
        f"{aggregate['accepted_draft_tokens_per_proposal']:.3f} | "
        f"{aggregate['paper_acceptance_length']:.3f} | "
        f"{aggregate['paper_verify_rate']:.3f} | "
        f"{aggregate['full_accept_rate']:.1%} |",
        "",
        "## Samples",
        "",
        "| sample | source index | proposals | accept_len | verify rate | full accept | truncated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in summary["samples"].items():
        lines.append(
            f"| {label} | {item['source_index']} | {item['proposals']} | "
            f"{item['paper_acceptance_length']:.3f} | "
            f"{item['paper_verify_rate']:.3f} | "
            f"{item['full_accept_rate']:.1%} | "
            f"{item['truncated_proposals']} |"
        )
    lines.extend([
        "",
        "## Aggregate Positions",
        "",
        "| pos | reached | accepted | conditional | prefix survival | confidence | prefix confidence | rejected here |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in aggregate["positions"]:
        lines.append(
            f"| {item['position']} | {item['reached']} | {item['accepted']} | "
            f"{common._fmt_rate(item['conditional_acceptance_rate'])} | "
            f"{common._fmt_rate(item['prefix_survival_rate'])} | "
            f"{common._fmt_rate(item['mean_conditional_confidence'])} | "
            f"{common._fmt_rate(item['mean_prefix_confidence'])} | "
            f"{item['rejected']} |"
        )
    lines.extend([
        "",
        "## Official HumanEval Reference",
        "",
        "DSpark paper Table 1 reports HumanEval accepted length "
        f"{reference['accepted_length_minimum']:.2f}-"
        f"{reference['accepted_length_maximum']:.2f} across its Qwen3/Gemma4 "
        "checkpoints. With seven drafts plus one target bonus, that is a "
        f"normalized verify-rate range of {reference['verify_rate_minimum']:.3f}-"
        f"{reference['verify_rate_maximum']:.3f}.",
        "",
        "This pilot uses the byte-exact DeepSpec HumanEval user turns and "
        "non-thinking mode, with confidence scheduling disabled. It is not a "
        "matched Table 1 reproduction: it uses eight evenly spaced samples "
        "instead of all 164, the V4-Flash IQ2XXS target and its released "
        "five-token DSpark sidecar instead of Qwen3/Gemma4 block-seven "
        f"checkpoints, {tokens} output tokens instead of 2048, and greedy "
        "decoding instead of temperature-1.0 rejection sampling. Treat the "
        "official values as directional targets; verify rate is the more useful "
        "cross-block comparison.",
        "",
        "- The prompt content matches DeepSpec `turns[0]`; ds4 applies the V4-specific chat template.",
        "- Capacity/EOS-truncated final proposals are excluded from acceptance metrics.",
        "- No HumanEval functional tests are executed; this measures draft acceptance only.",
        "- Confidence values are raw sigmoid outputs without STS calibration.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(run_dir, runs, records, summary, report):
    common.write_csv(
        run_dir / "runs.csv",
        runs,
        ("prompt", "mode", "stdout_sha256", "stdout_file", "stderr_file"),
    )
    scalar_rows = []
    position_rows = []
    for record in records:
        label = record["label"]
        item = summary["samples"][label]
        scalar_rows.append({
            "sample": label,
            "source_index": record["source_index"],
            "block_size": item["block_size"],
            "proposals": item["proposals"],
            "draft_tokens_per_proposal": item["draft_tokens_per_proposal"],
            "accepted_draft_tokens_per_proposal": (
                item["accepted_draft_tokens_per_proposal"]
            ),
            "paper_acceptance_length": item["paper_acceptance_length"],
            "paper_verify_rate": item["paper_verify_rate"],
            "full_accept_rate": item["full_accept_rate"],
            "truncated_proposals": item["truncated_proposals"],
        })
        for position in item["positions"]:
            position_rows.append({"sample": label, **position})
    aggregate_row = {
        "sample": "aggregate",
        "source_index": "",
        "block_size": summary["aggregate"]["block_size"],
        "proposals": summary["aggregate"]["proposals"],
        "draft_tokens_per_proposal": summary["aggregate"]["draft_tokens_per_proposal"],
        "accepted_draft_tokens_per_proposal": (
            summary["aggregate"]["accepted_draft_tokens_per_proposal"]
        ),
        "paper_acceptance_length": summary["aggregate"]["paper_acceptance_length"],
        "paper_verify_rate": summary["aggregate"]["paper_verify_rate"],
        "full_accept_rate": summary["aggregate"]["full_accept_rate"],
        "truncated_proposals": summary["aggregate"]["truncated_proposals"],
    }
    scalar_rows.append(aggregate_row)
    for position in summary["aggregate"]["positions"]:
        position_rows.append({"sample": "aggregate", **position})
    common.write_csv(
        run_dir / "acceptance.csv", scalar_rows, tuple(scalar_rows[0])
    )
    common.write_csv(
        run_dir / "positions.csv", position_rows, tuple(position_rows[0])
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(report, encoding="utf-8")


def main():
    args, root = parse_args()
    for name in ("binary", "model", "dspark_model", "corpus_dir"):
        setattr(args, name, getattr(args, name).resolve())
    records, provenance = load_corpus(args, root)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    default_dir = root / (
        f"speed-bench/local-runs/humaneval-acceptance-{stamp}"
    )
    run_dir = (args.output_dir or default_dir).resolve()
    prompts = prompt_paths(run_dir, records)
    for record in records:
        label = record["label"]
        prompt = prompts[label]
        print(
            f"{label} baseline reference: "
            f"{common.command_text(args, prompt, 'baseline')}"
        )
        print(
            f"{label} acceptance runtime: "
            f"{common.command_text(args, prompt, 'runtime', acceptance_audit=True)}"
        )
    print(
        "HumanEval acceptance pilot: eight fresh baseline/exact-runtime pairs; "
        "non-thinking, greedy, five drafts, no throughput conclusions."
    )
    if args.dry_run:
        print("Dry run only; no prompts materialized and no model execution performed.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    materialize_prompts(prompts, records)
    metadata = common.collect_metadata(
        args, root, prompts, provenance, acceptance_reference=None
    )
    metadata["experiment"] = "deepspec_humaneval_acceptance_pilot"
    metadata["official_protocol"] = {
        "samples": 164,
        "max_new_tokens": 2048,
        "temperature": 1.0,
        "seed": 980406,
        "draft_tokens": 7,
        "non_thinking": True,
        "confidence_scheduler": False,
    }
    (run_dir / "metadata.start.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    runs = []
    audit_rows = []
    for record in records:
        label = record["label"]
        prompt = prompts[label]
        baseline_row, reference = common.execute(
            args, root, run_dir, "reference", label, prompt,
            "baseline", None,
        )
        runs.append(baseline_row)
        common.cooldown(args.cooldown)
        audit_row, _ = common.execute(
            args, root, run_dir, "acceptance", label, prompt,
            "runtime", reference, acceptance_audit=True,
        )
        runs.append(audit_row)
        audit_rows.append(audit_row)
        common.cooldown(args.cooldown)

    summary = summarize(audit_rows, records, provenance, args.tokens)
    report = render_report(summary)
    write_outputs(run_dir, runs, records, summary, report)
    common.finish_metadata(metadata, root, run_dir)
    print("\n" + report.rstrip())
    print(f"Raw acceptance: {run_dir / 'acceptance.csv'}")
    print(f"Position details: {run_dir / 'positions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
