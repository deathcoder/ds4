#!/usr/bin/env python3
"""Verify this portable evidence bundle without loading the model or running Metal."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    median = statistics.median(values)
    return dict(count=len(values), median=median,
                mad=statistics.median(abs(value - median) for value in values),
                min=min(values), max=max(values))


manifest = read(ROOT / 'artifact-hashes.json')
actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*')
          if path.is_file() and path.name != 'artifact-hashes.json'
          and '__pycache__' not in path.parts}
assert actual == set(manifest['artifacts']), 'Missing or unexpected evidence'
for name, expected in manifest['artifacts'].items():
    path = ROOT / name
    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], name

results = ROOT / 'results'
raw = read(results / 'paired-raw.json')
summary = read(results / 'paired-summary.json')
state = read(results / 'state.json')
audit = read(ROOT / 'independent-audit.json')
assert state['phase'] == 'complete' and audit['passed']
assert len(raw['pairs']) == 5
assert [pair['order'] for pair in raw['pairs']] == ['AB', 'BA', 'AB', 'BA', 'AB']
assert all(pair['valid'] and pair['attempt'] == 1 for pair in raw['pairs'])
assert digest(results / 'paired-raw.json') == audit['raw_sha256']
assert digest(results / 'paired-summary.json') == audit['summary_sha256']
assert digest(ROOT / 'inputs/host-before.json') == raw['host_manifest_sha256']
assert digest(ROOT / 'inputs/correctness-manifest.json') == raw['correctness_manifest_sha256']
context = summary['contexts'][0]
for role in ('oracle', 'candidate'):
    for metric, expected in context[role].items():
        assert distribution([pair[role][metric] for pair in raw['pairs']]) == expected, metric
    evidence = [pair['evidence'][role] for pair in raw['pairs']] + state['warmups'][role]
    assert len(evidence) == 6
    for record in evidence:
        directory = results / record['output']
        measurement = directory / 'measurement.json'
        assert digest(measurement) == record['measurement_sha256']
        data = read(measurement)
        assert data['status'] == 'passed'
        for artifact in data['artifacts'].values():
            path = directory / artifact['path']
            assert path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256']
        if role == 'candidate':
            engine = read(directory / 'engine-run.json')
            assert engine['selection']['oracle_transcript_match']
            assert engine['selection']['selected_tokens_checksum'] == 11977321378026793466
for metric, expected in context['candidate_over_oracle'].items():
    assert distribution([pair['candidate'][metric] / pair['oracle'][metric]
                         for pair in raw['pairs']]) == expected, metric
profile = read(ROOT / 'profile32k.json')
assert digest(ROOT / 'profile32k.json') == audit['profile_sha256']
assert not profile['paired_protocol_eligible'] and profile['gen_tokens'] == 128
assert profile['selection']['selected_tokens_checksum'] == 11977321378026793466
assert profile['selection']['oracle_transcript_match']
print(f"PASS: {len(actual)} artifacts, 5 valid pairs, 2 warmups, exact transcripts and independent statistics")
