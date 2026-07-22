#!/usr/bin/env python3
"""Measure one pinned oMLX mode on the frozen HumanEval workload.

Run this script with the Python interpreter from the pinned oMLX checkout.
It intentionally measures one mode per process so model construction and the
process-wide MTP patch state cannot leak between configurations.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


MODES = {
    "baseline": {
        "mtp_enabled": False,
        "mtp_num_draft_tokens": None,
        "custom_verify_qmm": True,
    },
    "mtp1": {
        "mtp_enabled": True,
        "mtp_num_draft_tokens": 1,
        "custom_verify_qmm": True,
    },
    "mtp2": {
        "mtp_enabled": True,
        "mtp_num_draft_tokens": 2,
        "custom_verify_qmm": True,
    },
    "mtp2_stock_qmm": {
        "mtp_enabled": True,
        "mtp_num_draft_tokens": 2,
        "custom_verify_qmm": False,
    },
    "mtp3": {
        "mtp_enabled": True,
        "mtp_num_draft_tokens": 3,
        "custom_verify_qmm": True,
    },
}
OMLX_COMMIT = "a20d60de2e843395819969e61d8845d2497c49f0"
MODEL_REPO = "Jundot/DeepSeek-V4-Flash-oQ2e-mtp"
MODEL_REVISION = "f42b63224cfed5cff40185004b77d7ff935a6c47"
DEFAULT_OMLX_SOURCE = Path("/Users/deathcodevision/dev/local-inference-lab/omlx")
DEFAULT_MODEL = Path(
    "/Users/deathcodevision/dev/local-inference-lab/omlx-models/"
    "Jundot/DeepSeek-V4-Flash-oQ2e-mtp"
)
MTP_STATS_RE = re.compile(
    r"MTP\[[^]]+\].*?tokens=(?P<tokens>\d+) cycles=(?P<cycles>\d+) "
    r"tok/cycle=(?P<tokens_per_cycle>[0-9.]+) "
    r"accept=(?P<accepted>\d+)/(?P<drafted>\d+) "
    r"\((?P<accept_percent>[0-9.]+)%\)"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command_output(cwd: Path, *command: str) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    return result.stdout.decode("utf-8", "replace").strip()


def mode_settings(mode: str) -> dict[str, int | bool | None]:
    try:
        return dict(MODES[mode])
    except KeyError as exc:
        raise ValueError(f"unsupported oMLX mode: {mode}") from exc


def install_stock_verify_qmm_route(module) -> None:
    """Keep oMLX verification on MLX's stock quantized-matmul route."""
    module.vk_eligible = lambda *_args, **_kwargs: False


def generation_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    wall_start: float,
    first_token_at: float | None,
    last_token_at: float | None,
    wall_end: float,
) -> dict[str, float]:
    first = first_token_at if first_token_at is not None else wall_end
    last = last_token_at if last_token_at is not None else wall_end
    prefill_seconds = max(first - wall_start, 0.0)
    generation_seconds = max(last - first, 0.0)
    reported_tps = (
        completion_tokens / generation_seconds
        if completion_tokens > 0 and generation_seconds > 0
        else 0.0
    )
    interval_tps = (
        (completion_tokens - 1) / generation_seconds
        if completion_tokens > 1 and generation_seconds > 0
        else 0.0
    )
    return {
        "prefill_seconds": prefill_seconds,
        "prefill_tps": (
            prompt_tokens / prefill_seconds
            if prompt_tokens > 0 and prefill_seconds > 0
            else 0.0
        ),
        "generation_seconds": generation_seconds,
        "generation_tps": reported_tps,
        "interval_generation_tps": interval_tps,
        "wall_seconds": max(wall_end - wall_start, 0.0),
    }


def parse_mtp_stats(messages: list[str]) -> dict[str, float | int] | None:
    for message in reversed(messages):
        match = MTP_STATS_RE.search(message)
        if not match:
            continue
        return {
            "tokens": int(match.group("tokens")),
            "cycles": int(match.group("cycles")),
            "tokens_per_cycle": float(match.group("tokens_per_cycle")),
            "accepted": int(match.group("accepted")),
            "drafted": int(match.group("drafted")),
            "accept_rate": float(match.group("accept_percent")) / 100.0,
        }
    return None


class MessageCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def take(self) -> list[str]:
        messages = self.messages
        self.messages = []
        return messages


def parse_args() -> tuple[argparse.Namespace, Path]:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Measure one oMLX V4 baseline or Lightning-MTP mode."
    )
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--omlx-source", type=Path, default=DEFAULT_OMLX_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / "speed-bench/humaneval-acceptance",
    )
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-idle", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sample_count <= 0 or args.tokens <= 1:
        parser.error("sample-count must be positive and tokens must exceed one")
    if args.cooldown < 0:
        parser.error("cooldown cannot be negative")
    if args.dry_run and args.validate_only:
        parser.error("dry-run and validate-only are mutually exclusive")
    if not args.dry_run and not args.validate_only and not args.confirm_idle:
        parser.error("refusing to benchmark without --confirm-idle")
    return args, root


def load_corpus(args: argparse.Namespace, root: Path):
    sys.path.insert(0, str(root / "speed-bench"))
    import run_dspark_humaneval_acceptance as corpus

    holder = argparse.Namespace(
        binary=root / "ds4",
        model=root / "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-"
        "SExpQ8-OutQ8-chat-v2-imatrix.gguf",
        dspark_model=root / "gguf/ds4flash-dspark.gguf",
        corpus_dir=args.corpus_dir,
        allow_dirty=args.allow_dirty,
    )
    records, provenance = corpus.load_corpus(holder, root)
    return (
        *corpus.select_records(
            records, args.sample_count, provenance["selection_policy"]
        ),
        provenance,
    )


def validate_checkpoint(model: Path) -> dict[str, int | str]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    for label, path in (
        ("model config", config_path),
        ("model tensor index", index_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "deepseek_v4":
        raise SystemExit("oMLX model is not a DeepSeek V4 checkpoint")
    if int(config.get("num_nextn_predict_layers", 0) or 0) < 1:
        raise SystemExit("oMLX checkpoint does not declare native MTP weights")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not any(key.startswith("mtp.0.") for key in weight_map):
        raise SystemExit("oMLX checkpoint tensor index has no mtp.0.* weights")
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise SystemExit("oMLX checkpoint tensor index has no shard entries")
    missing = [name for name in shard_names if not (model / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise SystemExit(
            f"oMLX checkpoint is incomplete; missing {len(missing)} shards: "
            f"{preview}{suffix}"
        )
    return {
        "config_sha256": sha256(config_path.read_bytes()),
        "index_sha256": sha256(index_path.read_bytes()),
        "shard_count": len(shard_names),
        "shard_bytes": sum((model / name).stat().st_size for name in shard_names),
        "mtp_tensor_count": sum(key.startswith("mtp.0.") for key in weight_map),
    }


def validate_native_kernels(source: Path) -> dict[str, object]:
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast
    except Exception as exc:
        raise SystemExit(f"cannot import oMLX native DeepSeek kernels: {exc}") from exc
    required = (
        "deepseek_mxfp4_gather_qmm_blocks",
        "deepseek_mxfp4_gather_qmm_pair_blocks",
        "deepseek_mxfp4_gather_qmm_pair_concat_blocks",
        "deepseek_affine_gather_qmm_blocks",
        "deepseek_affine_gather_qmm_pair_concat_blocks",
    )
    missing = [name for name in required if not fast.has_symbol(name)]
    if not fast.is_native_available() or missing:
        raise SystemExit(
            "oMLX native DeepSeek kernels are unavailable; build the pinned "
            "checkout with OMLX_WITH_CUSTOM_KERNEL=1 before benchmarking. "
            f"Missing symbols: {', '.join(missing) or 'native extension'}"
        )
    kernel_dir = source / "omlx/custom_kernels/glm_moe_dsa"
    artifacts = sorted(
        path
        for path in kernel_dir.iterdir()
        if path.is_file() and path.suffix in (".so", ".dylib", ".metallib")
    )
    if not artifacts:
        raise SystemExit("oMLX native DeepSeek kernel artifacts are missing")
    manifest = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()),
        }
        for path in artifacts
    ]
    return {
        "available": True,
        "required_symbols": list(required),
        "artifacts": manifest,
    }


def validate_paths(
    args: argparse.Namespace, root: Path
) -> tuple[str, dict | None, dict | None]:
    source = args.omlx_source.resolve()
    model = args.model.resolve()
    if not (source / "omlx/engine/batched.py").is_file():
        raise SystemExit(f"invalid oMLX source checkout: {source}")
    source_commit = command_output(source, "git", "rev-parse", "HEAD")
    if source_commit != OMLX_COMMIT:
        raise SystemExit(
            "oMLX source commit drifted; audit the new revision before benchmarking: "
            f"{source_commit}"
        )
    checkpoint = None if args.dry_run else validate_checkpoint(model)
    native_kernels = None if args.dry_run else validate_native_kernels(source)
    dirty = command_output(root, "git", "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "tracked ds4 changes detected; commit them or pass --allow-dirty:\n" + dirty
        )
    return source_commit, checkpoint, native_kernels


async def execute_task(engine, record: dict, tokens: int, collector: MessageCollector):
    collector.take()
    first_token_at = None
    last_token_at = None
    final = None
    wall_start = time.perf_counter()
    async for output in engine.stream_chat(
        messages=[{"role": "user", "content": record["turns"][0]}],
        max_tokens=tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=1,
        chat_template_kwargs={"enable_thinking": False},
    ):
        if first_token_at is None and output.generated_at is not None:
            first_token_at = float(output.generated_at)
        if output.generated_until is not None:
            last_token_at = float(output.generated_until)
        final = output
    wall_end = time.perf_counter()
    if final is None:
        raise RuntimeError(f"{record['label']} produced no oMLX output")
    if final.cached_tokens:
        raise RuntimeError(
            f"{record['label']} used {final.cached_tokens} cached prompt tokens; "
            "the comparison requires a cache-disabled request"
        )
    metrics = generation_metrics(
        final.prompt_tokens,
        final.completion_tokens,
        wall_start,
        first_token_at,
        last_token_at,
        wall_end,
    )
    output_data = final.text.encode("utf-8")
    return {
        "prompt": record["label"],
        "source_index": record["source_index"],
        "prompt_tokens": final.prompt_tokens,
        "completion_tokens": final.completion_tokens,
        "cached_tokens": final.cached_tokens,
        "finish_reason": final.finish_reason,
        "output_data": output_data,
        "output_sha256": sha256(output_data),
        "mtp_stats": parse_mtp_stats(collector.take()),
        **metrics,
    }


def render_report(summary: dict) -> str:
    lines = [
        f"# oMLX HumanEval Mode: {summary['mode']}",
        "",
        "Single-process, cache-disabled, uninstrumented oMLX generation.",
        "Generation t/s uses the same completion-tokens / producer-interval convention as oMLX's built-in benchmark.",
        "Custom verify QMM: "
        + (
            "enabled." if summary["mode_settings"]["custom_verify_qmm"] else "disabled."
        ),
        "",
        "| samples | median generation | median interval | median prefill |",
        "|---:|---:|---:|---:|",
        f"| {summary['sample_count']} | {summary['generation_tps_median']:.2f} t/s | "
        f"{summary['interval_generation_tps_median']:.2f} t/s | "
        f"{summary['prefill_tps_median']:.2f} t/s |",
        "",
        "| task | prompt tok | output tok | generation | interval | prefill | MTP accept | tok/cycle |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["samples"]:
        mtp = item.get("mtp_stats") or {}
        accept = f"{mtp['accept_rate']:.3f}" if mtp else "n/a"
        tpc = f"{mtp['tokens_per_cycle']:.2f}" if mtp else "n/a"
        lines.append(
            f"| {item['prompt']} | {item['prompt_tokens']} | "
            f"{item['completion_tokens']} | {item['generation_tps']:.2f} t/s | "
            f"{item['interval_generation_tps']:.2f} t/s | "
            f"{item['prefill_tps']:.2f} t/s | {accept} | {tpc} |"
        )
    lines.extend(
        [
            "",
            "- The first excluded task warms the loaded model and Metal kernels.",
            "- Prefix caching and paged SSD caching are disabled.",
            "- Thinking is disabled and sampling is greedy with seed 1.",
            "- Process startup and model loading are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace, root: Path, records: list[dict], run_dir: Path):
    import statistics

    sys.path.insert(0, str(args.omlx_source.resolve()))
    from omlx.engine.batched import BatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig

    config = mode_settings(args.mode)
    if not config["custom_verify_qmm"]:
        from omlx.patches import qwen35_verify_qmm

        install_stock_verify_qmm_route(qwen35_verify_qmm)
    settings = ModelSettings(
        mtp_enabled=bool(config["mtp_enabled"]),
        mtp_num_draft_tokens=config["mtp_num_draft_tokens"],
        enable_thinking=False,
        index_cache_freq=None,
        turboquant_kv_enabled=False,
        specprefill_enabled=False,
        dflash_enabled=False,
    )
    scheduler = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=16384,
        chunked_prefill=False,
        paged_ssd_cache_dir=None,
        hot_cache_max_size=0,
    )
    collector = MessageCollector()
    mtp_logger = logging.getLogger("omlx.patches.mlx_lm_mtp.batch_generator")
    prior_level = mtp_logger.level
    mtp_logger.setLevel(logging.INFO)
    mtp_logger.addHandler(collector)
    engine = BatchedEngine(
        str(args.model.resolve()),
        scheduler_config=scheduler,
        stream_interval=1,
        enable_thinking=False,
        model_settings=settings,
    )
    rows = []
    try:
        await engine.start()
        print(f"[{args.mode}] excluded warmup: {records[0]['label']}", flush=True)
        await execute_task(engine, records[0], args.tokens, collector)
        for position, record in enumerate(records, start=1):
            if args.cooldown:
                print(f"cooldown: {args.cooldown:g}s", flush=True)
                await asyncio.sleep(args.cooldown)
            print(
                f"[{args.mode}] measured {position:02d}/{len(records)}: "
                f"{record['label']}",
                flush=True,
            )
            row = await execute_task(engine, record, args.tokens, collector)
            output_name = f"{record['label']}.stdout"
            (run_dir / output_name).write_bytes(row.pop("output_data"))
            row["output_file"] = output_name
            rows.append(row)
    finally:
        mtp_logger.removeHandler(collector)
        mtp_logger.setLevel(prior_level)
        await engine.stop()

    summary = {
        "mode": args.mode,
        "mode_settings": config,
        "sample_count": len(rows),
        "generation_tps_median": statistics.median(
            row["generation_tps"] for row in rows
        ),
        "interval_generation_tps_median": statistics.median(
            row["interval_generation_tps"] for row in rows
        ),
        "prefill_tps_median": statistics.median(row["prefill_tps"] for row in rows),
        "samples": rows,
    }
    fields = [
        "prompt",
        "source_index",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "finish_reason",
        "prefill_seconds",
        "prefill_tps",
        "generation_seconds",
        "generation_tps",
        "interval_generation_tps",
        "wall_seconds",
        "output_sha256",
        "output_file",
        "mtp_accept_rate",
        "mtp_tokens_per_cycle",
        "mtp_cycles",
    ]
    with (run_dir / "throughput.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            mtp = row.get("mtp_stats") or {}
            flat = {
                **row,
                "mtp_accept_rate": mtp.get("accept_rate"),
                "mtp_tokens_per_cycle": mtp.get("tokens_per_cycle"),
                "mtp_cycles": mtp.get("cycles"),
            }
            writer.writerow({key: flat.get(key) for key in fields})
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> int:
    args, root = parse_args()
    args.omlx_source = args.omlx_source.resolve()
    args.model = args.model.resolve()
    args.corpus_dir = args.corpus_dir.resolve()
    source_commit, checkpoint, native_kernels = validate_paths(args, root)
    if args.validate_only:
        print(f"oMLX source: {args.omlx_source} @ {source_commit}")
        print(f"model: {args.model}")
        print(
            "Checkpoint validation: PASS; "
            f"{checkpoint['shard_count']} shards, "
            f"{checkpoint['shard_bytes'] / (1024**3):.2f} GiB, "
            f"{checkpoint['mtp_tensor_count']} mtp.0.* tensors."
        )
        print(
            "Native DeepSeek kernels: PASS; "
            f"{len(native_kernels['artifacts'])} fingerprinted artifacts."
        )
        return 0
    records, selection, provenance = load_corpus(args, root)
    measured = records
    print(
        f"oMLX mode={args.mode} config={mode_settings(args.mode)}; "
        f"warmup={records[0]['label']}; measured="
        + ",".join(record["label"] for record in measured)
    )
    print(f"oMLX source: {args.omlx_source} @ {source_commit}")
    print(f"model: {args.model}")
    if args.dry_run:
        print("Dry run only; no model loading or benchmark execution performed.")
        return 0
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_dir
        or root / f"speed-bench/local-runs/omlx-humaneval-{args.mode}-{stamp}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "experiment": "omlx_humaneval_mode",
        "mode": args.mode,
        "mode_settings": mode_settings(args.mode),
        "omlx_source": str(args.omlx_source),
        "omlx_commit": source_commit,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model": str(args.model),
        "model_config_sha256": checkpoint["config_sha256"],
        "model_index_sha256": checkpoint["index_sha256"],
        "model_shard_count": checkpoint["shard_count"],
        "model_shard_bytes": checkpoint["shard_bytes"],
        "model_mtp_tensor_count": checkpoint["mtp_tensor_count"],
        "native_deepseek_kernels": native_kernels,
        "ds4_commit": command_output(root, "git", "rev-parse", "HEAD"),
        "platform": platform.platform(),
        "python": sys.version,
        "selection": selection,
        "provenance_source_commit": provenance["source_commit"],
        "protocol": {
            "warmup_tasks": 1,
            "measured_tasks": len(measured),
            "tokens": args.tokens,
            "temperature": 0.0,
            "seed": 1,
            "thinking": False,
            "prefix_cache": False,
            "single_request": True,
        },
        "initial_processes": command_output(
            root, "ps", "-Ao", "pid,pcpu,pmem,comm", "-r"
        ).splitlines()[:25],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    asyncio.run(run(args, root, records, run_dir))
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    print(f"Raw results: {run_dir / 'throughput.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
