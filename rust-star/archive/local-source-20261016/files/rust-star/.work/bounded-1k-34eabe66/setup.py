from pathlib import Path
import json,hashlib,datetime,subprocess
w=Path(__file__).resolve().parent;r=w.parents[2];old=r/'rust-star/.work/bootstrap-1k-current';life=r/'rust-star/.work/bootstrap-lifetimes-bff07722';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
ref=read(old/'plan.json')['source_reference'];assert all(sha(r/n)==h for n,h in ref['sources'].items());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();assert head.startswith('34eabe66')
for name in ['engine-runner.py','engine-audit.py','audit-phases.py','run-validation.py','audit-profile.py','full-input-pins.py','audit-ring.py','router-probe.m','ffn_router.metal']:(w/name).write_bytes((old/name).read_bytes())
for name in ['observer.inc','observer-proof.m']:(w/name).write_bytes((life/name).read_bytes())
(w/'prior-1k-build.json').write_bytes((old/'candidate-build.json').read_bytes());(w/'prior-1k-engine-plan.json').write_bytes((old/'engine-plan.json').read_bytes())
for n in ['metal.rs','long_prefill.rs','metal_shim.m']:
 (w/('baseline-'+n)).write_bytes((r/'rust-star/runtime/src'/n).read_bytes());(w/('prior-'+n)).write_bytes((old/('candidate-'+n)).read_bytes())
subprocess.run(['cp','-c',str(old/'baseline-rust-star'),str(w/'baseline-rust-star')],check=True);assert sha(w/'baseline-rust-star')==ref['executable_sha256']
engine=read(old/'engine-plan.json');engine.update(schema='rust-star-bounded-1k-engine-plan-v1',created_utc=utc(),dependency='Fresh 12K internal, normal/profiled 32K full C0, strict profile audit and separate one-in-flight lifetime/full-C0 proof must pass.',baseline_runtime_commit=head,role_B='Native 1K bootstrap with checked completion after every nonfinal tile, Rust 1.74');(w/'engine-plan.json').write_text(json.dumps(engine,indent=2)+'\n');assert engine['retention_criteria']==read(old/'engine-plan.json')['retention_criteria']
plan=dict(schema='rust-star-bounded-1k-milestone-v1',created_utc=utc(),baseline_commit=head,source_reference=ref,scope='The preserved 1K candidate plus a checked completion wait after each nonfinal native 1K tile; no new arithmetic, aliases, pooling, fixture or oracle changes.',hypothesis='Bounding unfinished work to one 1K tile may reduce footprint while preserving useful prefill gains. Prior lifetime observations do not prove the OS-footprint cause.',validation=['320 Rust tests; release/fmt; unchanged Metal router and modulo ring proofs','12K internal; normal/profiled 32K/128 full C0 and strict 4466-interval coverage','Separate observed 32K/128 full C0: 28 native tiles,119 snapshots; no previous scratch at entry/registration, zero after every pool, at most one unfinished command; recorded per-tile scratch equals prior one-set lengths','Two warmups then five fixed alternating engine pairs with all unchanged retention gates','Full input-before/after hashes, raw evidence, retain only on every gate, portable verification and local commit'],lifetime_reference=read(life/'supplemental-analysis.json')['roles']['wide']['one_set_per_frontier'],indexed_components=read(old/'plan.json')['indexed_components'],compressed_components=read(old/'plan.json')['compressed_components'],failure_policy='Stop on any correctness/protocol failure; preserve all attempts. No discretionary repeats or threshold changes. Restore accepted source/default on failure. Complete only this milestone then stop.',no_push=True,no_256k=True,stop_after_milestone=True)
(w/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
prior=(w/'prior-metal_shim.m').read_text();before='            [context.prefillBoundaryCommands addObject:command];';after=before+'''\n            // Drain this tile before another call allocates or initializes scratch.
            // The final tile already waits below; keep its complete batch audit.
            if (native_1k_bootstrap && position_start + rows < prefill_rows) {
                if (!command_succeeded(command, error, error_bytes)) return 0;
            }''';assert prior.count(before)==1;candidate=prior.replace(before,after);assert candidate.replace(after,before)==prior
(w/'candidate-metal_shim.m').write_text(candidate)
for n in ['metal.rs','long_prefill.rs']:(w/('candidate-'+n)).write_bytes((w/('prior-'+n)).read_bytes())
(w/'scheduling-edit.json').write_text(json.dumps(dict(before=before,after=after,prior_sha256=sha(w/'prior-metal_shim.m'),candidate_sha256=sha(w/'candidate-metal_shim.m'),inverse_exact=True),indent=2)+'\n')
# Reuse the proven observer transformation with only its input role narrowed.
s=(life/'prepare.py').read_text().replace("for role in ['accepted','wide']:","for role in ['candidate']:").replace("assert manifests['accepted']['patches']==manifests['wide']['patches']", "assert len(manifests)==1")
(w/'prepare-observer.py').write_text(s)
subprocess.run(['python3',str(w/'prepare-observer.py')],cwd=r,check=True)
print('Prospective plan and candidate/observer snapshots prepared; runtime still accepted')
