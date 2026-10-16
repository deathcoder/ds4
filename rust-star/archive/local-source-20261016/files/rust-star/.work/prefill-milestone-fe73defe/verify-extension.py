
# Bind live correctness gates and independently reconcile both stage profiles.
correctness = read(ROOT / 'inputs/correctness-manifest.json')
for name, evidence in correctness['fresh_candidate_full_output_runs'].items():
    path = ROOT / 'correctness' / (name + '.json')
    assert digest(path) == evidence['sha256']
    gate = read(path)
    assert gate['complete_32k_c0_claim'] and gate['checked_decode_steps'] == 128
for name, expected in correctness['shorter_controls'].items():
    assert digest(ROOT / 'correctness' / (name + '.json')) == expected
retained = ROOT / 'correctness/batch-retained.json'
assert digest(retained) == correctness['retained_two_cycle_control_sha256']
assert read(retained)['selection']['both_oracle_transcripts_match']
assert not read(retained)['paired_protocol_eligible']
assert digest(ROOT / 'correctness/accepted-reference-correctness.json') == correctness['accepted_reference_correctness_manifest_sha256']
for name in ['profile32k', 'correctness/profile-before']:
    path = ROOT / (name + '.json')
    report = read(path)
    analysis = read(ROOT / (name + '-analysis.json'))
    assert analysis['passed'] and analysis['profile_sha256'] == digest(path)
    rows = report['long_prefill_profile']['rows']
    assert len(rows) == analysis['rows']
    assert {row['chunk_start'] for row in rows} == set(range(0, 32768, 4096))
    for chunk in range(4096, 32768, 4096):
        covered = {layer for row in rows if row['chunk_start'] == chunk
                   for layer in range(row['first_layer'], row['last_layer'] + 1)}
        assert covered == set(range(43))
        for layer in range(2, 43):
            stages = {row['stage'] for row in rows if row['chunk_start'] == chunk
                      and row['first_layer'] == row['last_layer'] == layer}
            assert {'bind', 'attention', 'tail'} <= stages
    groups = {}
    for row in rows:
        family = ('prefix' if row['chunk_start'] == 0 else 'bootstrap'
                  if row['first_layer'] != row['last_layer'] else 'even'
                  if row['first_layer'] % 2 == 0 else 'odd')
        group = groups.setdefault(family + '/' + row['stage'],
                                  dict(calls=0, host_elapsed_ms=0., command_wall_ms=0., gpu_ms=0.))
        for key in group:
            assert row[key] >= 0
            group[key] += row[key]
    assert groups == analysis['group_totals']
    for key, parts in [('gpu_ms', ['prefill_transformer_gpu_ms', 'prefill_tile_gpu_ms']),
                       ('command_wall_ms', ['prefill_transformer_wall_ms', 'prefill_tile_wall_ms'])]:
        assert abs(sum(row[key] for row in rows) - sum(report['timing'][part] for part in parts)) < .01
    assert not report['paired_protocol_eligible']
    assert report['selection']['oracle_transcript_match']
    assert report['selection']['selected_tokens_checksum'] == 11977321378026793466
assert read(ROOT / 'profile32k-analysis.json')['group_totals']['even/tail']['calls'] == 147
print('PASS: live C0 evidence, retained reuse, and both complete stage profiles')
interruption = read(ROOT / 'interruption.json')
assert digest(ROOT / 'interrupted-state.json') == interruption['interrupted_state_sha256']
assert invalid[0]['repetition'] == 5 and invalid[0]['attempt'] == 1
assert invalid[0]['invalid_reason'] == interruption['reason']
record = invalid[0]['evidence']['oracle']
directory = results / record['output']
assert digest(directory / 'measurement.json') == record['measurement_sha256']
for artifact in read(directory / 'measurement.json')['artifacts'].values():
    path = directory / artifact['path']
    assert path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256']
print('PASS: external interruption and excluded complete reference evidence retained')
