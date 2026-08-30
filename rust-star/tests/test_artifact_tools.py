#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


RUST_STAR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUST_STAR_DIR))

from artifact_lib import (  # noqa: E402
    ArtifactError,
    ORACLE_V2_SOURCE_COMMIT,
    ORACLE_V2_SOURCE_TREE,
    SOURCE_COMMIT,
    SOURCE_TREE,
    compare_logit_artifacts,
    load_logit_artifact,
    open_bundle,
    sha256_file,
    validate_oracle_bundle,
)


def write_logits(path: Path, logits: list[float], **overrides: object) -> None:
    argmax = max(range(len(logits)), key=lambda index: logits[index])
    payload: dict[str, object] = {
        "source": "test",
        "backend": "metal",
        "model": "model.gguf",
        "quality": False,
        "quant_bits": 2,
        "prompt_tokens": 2048,
        "frontier_tokens": 2048,
        "ctx": 2177,
        "vocab": len(logits),
        "argmax_id": argmax,
        "logits": logits,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class LogitComparisonTests(unittest.TestCase):
    def test_c0_exact_including_negative_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            values = [-0.0, 1.25, -2.5, 8.0]
            write_logits(reference_path, values)
            write_logits(candidate_path, values)
            report = compare_logit_artifacts(
                load_logit_artifact(reference_path),
                load_logit_artifact(candidate_path),
            )
            self.assertTrue(report["c0_exact"])
            self.assertEqual(report["bit_mismatches"], 0)
            self.assertEqual(report["classification"], "C0")

    def test_signed_zero_is_not_c0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            write_logits(reference_path, [-0.0, 2.0])
            write_logits(candidate_path, [0.0, 2.0])
            report = compare_logit_artifacts(
                load_logit_artifact(reference_path),
                load_logit_artifact(candidate_path),
            )
            self.assertFalse(report["c0_exact"])
            self.assertEqual(report["bit_mismatches"], 1)
            self.assertEqual(report["max_absolute_error"], 0.0)

    def test_drift_metrics_and_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            write_logits(reference_path, [0.0, 1.0, 2.0, 3.0])
            write_logits(candidate_path, [0.0, 1.0, 2.25, 3.0], ctx=4096)
            report = compare_logit_artifacts(
                load_logit_artifact(reference_path),
                load_logit_artifact(candidate_path),
            )
            self.assertFalse(report["c0_exact"])
            self.assertEqual(report["bit_mismatches"], 1)
            self.assertEqual(report["max_absolute_error"], 0.25)
            self.assertEqual(report["max_absolute_error_index"], 2)
            self.assertGreater(report["max_ulp_distance"], 0)
            self.assertTrue(report["argmax_match"])
            self.assertEqual(report["metadata_mismatches"][0]["field"], "ctx")

    def test_invalid_argmax_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            write_logits(path, [0.0, 1.0], argmax_id=0)
            with self.assertRaisesRegex(ArtifactError, "argmax metadata mismatch"):
                load_logit_artifact(path)

    def test_missing_required_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            write_logits(path, [0.0, 1.0])
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["quality"]
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "quality flag"):
                load_logit_artifact(path)


class BundleValidationTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        bundle = root / "oracle-v1-test"
        bundle.mkdir()
        artifact_path = bundle / "evidence.txt"
        artifact_path.write_text("evidence\n", encoding="utf-8")
        manifest = {
            "schema": "rust-star-oracle-manifest-v1",
            "oracle_id": "oracle-v1",
            "status": "complete",
            "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
            "capture_kit": {
                "commit": "a" * 40,
                "tree": "b" * 40,
                "branch": "agent/rust-star-bootstrap",
                "tracked_worktree": "clean",
            },
            "model": {
                "filename": "model.gguf",
                "bytes": 123,
                "sha256": "c" * 64,
                "absolute_path_recorded": False,
            },
            "configuration": {
                "contexts": [2048],
                "correctness_enabled": False,
                "conformance_enabled": False,
                "performance_enabled": False,
            },
            "correctness": {"status": "skipped", "runs": []},
            "conformance": {"status": "skipped", "runs": []},
            "performance": {"status": "skipped", "runs": []},
            "test_artifact": {
                "path": "evidence.txt",
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            },
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return bundle

    def make_v2_bundle(self, root: Path) -> Path:
        bundle = root / "oracle-v2-test"
        bundle.mkdir()
        runs: list[dict[str, object]] = []
        for context in (2048, 32768):
            for repetition in (1, 2):
                relative = Path("conformance") / f"ctx_{context}" / f"run_{repetition}.json"
                path = bundle / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_logits(
                    path,
                    [0.0, 2.0, 1.0],
                    prompt_tokens=context,
                    frontier_tokens=context,
                    ctx=context,
                )
                runs.append({
                    "context": context,
                    "repetition": repetition,
                    "logits": {
                        "path": str(relative),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    },
                })
        manifest = {
            "schema": "rust-star-oracle-manifest-v1",
            "oracle_id": "oracle-v2",
            "status": "complete",
            "source": {
                "commit": ORACLE_V2_SOURCE_COMMIT,
                "tree": ORACLE_V2_SOURCE_TREE,
            },
            "capture_kit": {
                "commit": "a" * 40,
                "tree": "b" * 40,
                "branch": "agent/rust-star-bootstrap",
                "tracked_worktree": "clean",
            },
            "model": {
                "filename": "model.gguf",
                "bytes": 123,
                "sha256": "c" * 64,
                "absolute_path_recorded": False,
            },
            "configuration": {
                "contexts": [2048, 32768],
                "conformance_repetitions": 2,
                "correctness_enabled": True,
                "conformance_enabled": True,
                "performance_enabled": False,
            },
            "correctness": {"status": "passed", "runs": []},
            "conformance": {"status": "passed", "runs": runs},
            "performance": {"status": "skipped", "runs": []},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return bundle

    def test_directory_and_archive_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.make_bundle(root)
            report = validate_oracle_bundle(bundle)
            self.assertTrue(report["valid"])
            self.assertEqual(report["verified_artifacts"], 1)

            archive = root / "bundle.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(bundle, arcname=bundle.name)
            with open_bundle(archive) as (extracted, archive_digest):
                self.assertEqual(archive_digest, hashlib.sha256(archive.read_bytes()).hexdigest())
                self.assertTrue(validate_oracle_bundle(extracted)["valid"])

    def test_tampered_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_bundle(Path(temporary))
            (bundle / "evidence.txt").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "integrity mismatch"):
                validate_oracle_bundle(bundle)

    def test_oracle_v2_repeated_conformance_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_v2_bundle(Path(temporary))
            report = validate_oracle_bundle(bundle)
            self.assertEqual(report["oracle_id"], "oracle-v2")
            self.assertEqual(report["conformance_repetitions"], 2)

    def test_oracle_v2_conformance_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_v2_bundle(Path(temporary))
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run = manifest["conformance"]["runs"][-1]
            path = bundle / run["logits"]["path"]
            write_logits(
                path,
                [0.0, 2.0, 1.25],
                prompt_tokens=32768,
                frontier_tokens=32768,
                ctx=32768,
            )
            run["logits"]["bytes"] = path.stat().st_size
            run["logits"]["sha256"] = sha256_file(path)
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "not C0 exact at context 32768"):
                validate_oracle_bundle(bundle)

    def test_oracle_v2_requires_two_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.make_v2_bundle(Path(temporary))
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["configuration"]["conformance_repetitions"] = 1
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "at least two"):
                validate_oracle_bundle(bundle)

    def test_early_partial_manifest_is_accepted_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            manifest = {
                "schema": "rust-star-oracle-manifest-v1",
                "oracle_id": "oracle-v1",
                "status": "failed",
                "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
                "configuration": {
                    "contexts": [2048],
                    "correctness_enabled": True,
                    "conformance_enabled": True,
                    "performance_enabled": True,
                },
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "not complete"):
                validate_oracle_bundle(bundle)
            report = validate_oracle_bundle(bundle, allow_partial=True)
            self.assertTrue(report["valid"])
            self.assertIsNone(report["model_sha256"])

    def test_archive_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.txt"
            payload.write_text("bad\n", encoding="utf-8")
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(payload, arcname="../escape.txt")
            with self.assertRaisesRegex(ArtifactError, "escapes extraction root"):
                with open_bundle(archive):
                    pass


if __name__ == "__main__":
    unittest.main()
