from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paired_benchmark_lib import (  # noqa: E402
    PairedBenchmarkError,
    load_paired_run,
    summarize_paired_run,
)


def metric_row(context: int, steady_tps: float) -> dict[str, int | float]:
    return {
        "ctx_tokens": context,
        "prefill_tokens": context,
        "gen_tokens": 128,
        "gen_steady_tokens": 127,
        "kvcache_bytes": context * 16,
        "process_peak_bytes": context * 32,
        "prefill_tps": steady_tps * 20,
        "prefill_ms": context / (steady_tps * 20) * 1000,
        "gen_tps": steady_tps * 0.95,
        "gen_ms": 128 / (steady_tps * 0.95) * 1000,
        "gen_first_ms": 1000 / steady_tps,
        "gen_steady_tps": steady_tps,
        "gen_steady_ms": 127 / steady_tps * 1000,
        "process_wall_ms": 100000 / steady_tps,
        "model_load_ms": 0.0,
    }


def valid_payload() -> dict:
    pairs = []
    oracle = [10.0, 20.0, 30.0]
    candidate = [15.0, 20.0, 60.0]
    for repetition, (oracle_tps, candidate_tps) in enumerate(zip(oracle, candidate), 1):
        pairs.append(
            {
                "context": 262144,
                "repetition": repetition,
                "attempt": 1,
                "order": "AB" if repetition % 2 else "BA",
                "valid": True,
                "oracle": metric_row(262144, oracle_tps),
                "candidate": metric_row(262144, candidate_tps),
            }
        )
    return {
        "schema": "rust-star-paired-raw-v1",
        "protocol": "rust-star-paired-benchmark-v1",
        "correctness_class": "C0",
        "host_manifest_sha256": "f" * 64,
        "correctness_manifest_sha256": "e" * 64,
        "configuration": {
            "contexts": [262144],
            "repetitions": 3,
            "gen_tokens": 128,
            "sampling": "greedy argmax excluding EOS",
            "primary_metric": "gen_steady_tps",
        },
        "oracle": {
            "source_commit": "a" * 40,
            "source_tree": "1" * 40,
            "executable_sha256": "2" * 64,
            "model_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "backend": "metal",
            "build_configuration": {"command": "make -j"},
            "runtime_configuration": {"threads": 20},
        },
        "candidate": {
            "source_commit": "d" * 40,
            "source_tree": "3" * 40,
            "executable_sha256": "4" * 64,
            "model_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "backend": "metal",
            "build_configuration": {"profile": "release"},
            "runtime_configuration": {"threads": 20},
        },
        "pairs": pairs,
    }


class PairedBenchmarkTests(unittest.TestCase):
    def load(self, payload: dict):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "paired.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_paired_run(path)

    def test_pairwise_ratio_is_aggregated_within_pairs(self) -> None:
        summary = summarize_paired_run(self.load(valid_payload()))
        context = summary["contexts"][0]
        self.assertTrue(summary["headline_eligible"])
        self.assertEqual(context["valid_pairs"], 3)
        self.assertEqual(
            context["candidate_over_oracle"]["gen_steady_tps"]["median"],
            1.5,
        )

    def test_invalid_pair_is_retained_and_excluded(self) -> None:
        payload = valid_payload()
        successful_retry = copy.deepcopy(payload["pairs"][1])
        successful_retry["attempt"] = 2
        payload["pairs"][1] = {
            "context": 262144,
            "repetition": 2,
            "attempt": 1,
            "order": "BA",
            "valid": False,
            "invalid_reason": "documented unrelated background load",
        }
        payload["pairs"].append(successful_retry)
        context = summarize_paired_run(self.load(payload))["contexts"][0]
        self.assertEqual(context["valid_pairs"], 3)
        self.assertEqual(len(context["invalid_pairs"]), 1)
        self.assertEqual(
            context["candidate_over_oracle"]["gen_steady_tps"]["median"],
            1.5,
        )

    def test_model_identity_mismatch_is_rejected(self) -> None:
        payload = valid_payload()
        payload["candidate"]["model_sha256"] = "e" * 64
        with self.assertRaisesRegex(PairedBenchmarkError, "model SHA-256 differ"):
            self.load(payload)

    def test_duplicate_and_missing_pairs_are_rejected(self) -> None:
        payload = valid_payload()
        payload["pairs"][2] = copy.deepcopy(payload["pairs"][0])
        with self.assertRaisesRegex(PairedBenchmarkError, "duplicate"):
            self.load(payload)

        payload = valid_payload()
        payload["pairs"].pop()
        with self.assertRaisesRegex(PairedBenchmarkError, "missing predeclared pair"):
            self.load(payload)

    def test_wrong_alternating_order_is_rejected(self) -> None:
        payload = valid_payload()
        payload["pairs"][1]["order"] = "AB"
        with self.assertRaisesRegex(PairedBenchmarkError, "expected 'BA'"):
            self.load(payload)

    def test_context_order_alternates_by_repetition(self) -> None:
        payload = valid_payload()
        payload["configuration"]["contexts"] = [2048, 262144]
        large_pairs = payload["pairs"]
        small_pairs = []
        for pair in large_pairs:
            small = copy.deepcopy(pair)
            small["context"] = 2048
            small["oracle"] = metric_row(2048, small["oracle"]["gen_steady_tps"])
            small["candidate"] = metric_row(2048, small["candidate"]["gen_steady_tps"])
            small_pairs.append(small)
        payload["pairs"] = [
            small_pairs[0],
            large_pairs[0],
            large_pairs[1],
            small_pairs[1],
            small_pairs[2],
            large_pairs[2],
        ]
        self.load(payload)
        payload["pairs"][0], payload["pairs"][1] = payload["pairs"][1], payload["pairs"][0]
        with self.assertRaisesRegex(PairedBenchmarkError, "alternating context order"):
            self.load(payload)

    def test_experimental_result_is_not_headline_eligible(self) -> None:
        payload = valid_payload()
        payload["correctness_class"] = "C2"
        summary = summarize_paired_run(self.load(payload))
        self.assertFalse(summary["headline_eligible"])


if __name__ == "__main__":
    unittest.main()
