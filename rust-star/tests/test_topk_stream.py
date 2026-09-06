"""Execute the production Metal streaming top-k with adversarial scheduling."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'darwin' and shutil.which('clang'), 'requires macOS Metal and clang')
class TopkStreamTests(unittest.TestCase):
    def test_topk_matches_cpu_sort_when_simd_groups_progress_unevenly(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / 'metal/argsort.metal').read_text()
        kernel = ('#include <metal_stdlib>\nusing namespace metal;\n'
                  + source[:source.index('struct ds4_metal_args_argsort_merge')]
                  + source[source.index('kernel void kernel_topk_stream512'):])
        marker = '        const uint used = atomic_load_explicit(&count, memory_order_relaxed);'
        skew = '''        // Exercise legal uneven progress between SIMD groups.
        if ((tid / 32u) % 2u != 0u) {
            volatile uint delay = tid;
            for (uint k = 0; k < 128u; ++k) delay = delay*1664525u+1013904223u;
        }
'''
        with tempfile.TemporaryDirectory() as temp:
            work = pathlib.Path(temp)
            executable = work / 'probe'
            subprocess.run(['clang', '-O2', '-Wno-deprecated-declarations', '-fobjc-arc',
                            '-framework', 'Foundation', '-framework', 'Metal',
                            str(root / 'rust-star/tests/topk_stream_probe.m'), '-o', str(executable)],
                           check=True, capture_output=True, text=True)
            for uneven in [False, True]:
                shader = work / 'topk.metal'
                shader.write_text(kernel.replace(marker, skew + marker) if uneven else kernel)
                for columns in [1025, 3072, 8192, 8224]:
                    with self.subTest(uneven=uneven, columns=columns):
                        result = subprocess.run([str(executable), str(shader), '50', str(columns)],
                                                capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn('PASS 50 repetitions, 1600 rows, 819200 selected indices', result.stdout)


if __name__ == '__main__':
    unittest.main()
