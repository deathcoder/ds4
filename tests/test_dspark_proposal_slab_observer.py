#!/usr/bin/env python3
"""Model-free safety and parser tests for the proposal-slab observer."""

from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speed-bench"))
import run_dspark_proposal_slab_observer as observer  # noqa: E402


class DSparkProposalSlabObserverTests(unittest.TestCase):
    def test_runner_freezes_exact_diagnostic_modes(self):
        self.assertEqual(observer.LAYERS, (41, 42))
        self.assertEqual(observer.THRESHOLD, "0.75")
        self.assertEqual(observer.PARTIAL_THRESHOLD, "0")
        self.assertEqual(
            observer.SCHEDULES,
            (("scheduled", "0.75"), ("fixed-k5", "0")),
        )
        env = observer.clean_env(42)
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME"], "1")
        self.assertEqual(env["DS4_DSPARK_MULTI_COMMIT"], "1")
        self.assertEqual(env["DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS"], "1")
        self.assertEqual(env["DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER"], "42")
        self.assertNotIn("DS4_DSPARK_FAST_BATCH_VERIFY", env)
        self.assertNotIn("DS4_DSPARK_GPU_RUNTIME_STATS", env)

    def test_parser_accepts_exact_ratio4_records(self):
        data = (
            b"ds4: DSpark proposal slab observer layer=42 ratio=4 proposed=5 "
            b"raw_rows=5/5 attn_prefixes=5/5 index_prefixes=5/5 "
            b"counters=5/5 result=exact\n"
            b"ds4: DSpark proposal slab publication layer=42 ratio=4 "
            b"proposed=5 accepted=3 prepared=exact result=exact\n"
        )
        parsed = observer.parse_observer(data, 42)
        self.assertEqual(parsed["observer"][0]["ratio"], 4)
        self.assertEqual(parsed["publication"][0]["accepted"], 3)

    def test_parser_exposes_partial_publication(self):
        data = (
            b"ds4: DSpark proposal slab observer layer=41 ratio=128 proposed=5 "
            b"raw_rows=5/5 attn_prefixes=5/5 index_prefixes=0/0 "
            b"counters=5/5 result=exact\n"
            b"ds4: DSpark proposal slab publication layer=41 ratio=128 "
            b"proposed=5 accepted=2 prepared=exact result=exact\n"
        )
        publications = observer.parse_observer(data, 41)["publication"]
        self.assertTrue(any(
            item["accepted"] < item["proposed"] for item in publications
        ))

    def test_parser_rejects_drift(self):
        data = (
            b"ds4: DSpark proposal slab observer layer=41 ratio=128 proposed=5 "
            b"raw_rows=5/5 attn_prefixes=4/5 index_prefixes=0/0 "
            b"counters=5/5 result=drift\n"
            b"ds4: DSpark proposal slab publication layer=41 ratio=128 "
            b"proposed=5 accepted=5 prepared=unverified result=drift\n"
        )
        with self.assertRaisesRegex(RuntimeError, "state drift"):
            observer.parse_observer(data, 41)

    def test_source_requires_all_safety_gates(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        gate_start = source.index("static int dspark_proposal_slab_observer_layer")
        gate_end = source.index(
            "static void dspark_proposal_slab_shadow_free", gate_start
        )
        gate = source[gate_start:gate_end]
        for name in (
            "DS4_DSPARK_GPU_RUNTIME",
            "DS4_DSPARK_MULTI_COMMIT",
            "DS4_DSPARK_GPU_RUNTIME_DIAGNOSTICS",
            "DS4_DSPARK_PROPOSAL_SLAB_OBSERVER_LAYER",
        ):
            self.assertIn(name, gate)

    def test_shadow_is_report_only_and_published_after_commit(self):
        source = (ROOT / "ds4.c").read_text(encoding="utf-8")
        publication = source.index(
            "dspark_proposal_slab_observer_report_publication(\n"
            "        &s->graph, n_commit);"
        )
        commit_branch = source.rindex(
            "runtime_batch_verify_full_accepts", 0, publication
        )
        logits_copy = source.index("memcpy(s->logits", publication)
        self.assertLess(commit_branch, publication)
        self.assertLess(publication, logits_copy)
        helper_start = source.index(
            "static void dspark_proposal_slab_observer_report_publication"
        )
        helper_end = source.index(
            "/* Exact multi-token target verifier for DSpark.", helper_start
        )
        helper = source[helper_start:helper_end]
        self.assertNotIn("return false", helper)
        self.assertNotIn("s->logits", helper)

    def test_metal_kernels_write_only_prefix_outputs(self):
        source = (ROOT / "metal/dsv4_kv.metal").read_text(encoding="utf-8")
        start = source.index("kernel_dsv4_proposal_prefix_state_ratio4")
        kernels = source[start:]
        self.assertIn("prefix_kv[output] =", kernels)
        self.assertIn("prefix_score[output] =", kernels)
        self.assertIsNone(re.search(
            r"(?:proposal_kv|proposal_score|base_kv|base_score)\s*\[[^]]+\]\s*=",
            kernels,
        ))


if __name__ == "__main__":
    unittest.main()
