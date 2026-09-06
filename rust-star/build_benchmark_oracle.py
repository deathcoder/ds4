#!/usr/bin/env python3
"""Build oracle-v3 inference with the sampling-inclusive benchmark driver."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ORACLE_COMMIT = "d35fb12d01d500b9cefcef24092c295687ceaf7e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ORACLE_COMMIT], cwd=root, text=True
    ).splitlines()
    sources = [name for name in names if name == "Makefile" or name.startswith("metal/")
               or ("/" not in name and Path(name).suffix in {".c", ".h", ".m", ".inc"})]
    archive = subprocess.check_output(["git", "archive", ORACLE_COMMIT, "--", *sources], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError("unsafe member in pinned source archive")
        tar.extractall(output)
    original_driver_sha = sha256(output / "ds4_bench.c")
    shutil.copyfile(root / "ds4_bench.c", output / "ds4_bench.c")
    # No inference source is taken from the evolving research checkout.
    subprocess.run(["make", "-j", str(args.jobs), "ds4-bench"], cwd=output, check=True)
    provenance = {
        "schema": "rust-star-benchmark-oracle-build-v1",
        "inference_source_commit": ORACLE_COMMIT,
        "inference_source_tree": subprocess.check_output(
            ["git", "rev-parse", ORACLE_COMMIT + "^{tree}"], cwd=root, text=True).strip(),
        "benchmark_source_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "original_benchmark_sha256": original_driver_sha,
        "benchmark_sha256": sha256(output / "ds4_bench.c"),
        "executable_sha256": sha256(output / "ds4-bench"),
        "source_sha256": {name: sha256(output / name) for name in sources},
        "timing_contract": "prefill includes initial argmax; each decode evaluation includes next-token argmax and commitment",
        "inference_source_overlay": False,
    }
    (output / "benchmark-build.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output / "benchmark-build.json")


if __name__ == "__main__":
    main()
