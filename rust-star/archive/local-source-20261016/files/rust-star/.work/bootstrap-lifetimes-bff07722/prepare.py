from pathlib import Path
import json,re,hashlib
w=Path(__file__).resolve().parent;r=w.parents[2];sha=lambda b:hashlib.sha256(b).hexdigest()
helper=(w/'observer.inc').read_text(); manifests={}
for role in ['accepted','wide']:
 s=(w/(role+'-metal_shim.m')).read_text();start=s.index('int rust_star_metal_run_prefill_layer0_boundary(');end=s.index('\nint ',start+4);original=s[start:end];f=original
 names=re.findall(r'id<MTLBuffer>\s+(\w+)',f);names=[n for n in names if n not in ['model_buffers','grown']];assert len(names)==len(set(names))
 persistent=set(re.findall(r'(\w+_buffer)\s*=\s*context\.\w+',f));persistent.add('full_kv_buffer');persistent.add('transition_diagnostics_buffer')
 assert persistent.issubset(set(names)),persistent-set(names)
 patches=[]
 def edit(before,after):
  global f
  assert f.count(before)==1,(role,before[:100],f.count(before));f=f.replace(before,after);patches.append(dict(before=before,after=after))
 active='batch_production_tiles && consume_kv_state && context.prefillLayerChunkStart != nil'
 def snap(phase):return f'rs_lifetime_snapshot(context.device, context.prefillBoundaryCommands, @"{phase}", rows, position_start, prefill_rows);'
 before='        if (kv_state_mode > 2u ||'
 edit(before,f'        const BOOL lifetime_trace = {active};\n        if (lifetime_trace) {snap("entry")}\n'+before)
 before='        rust_star_fp8_quantize_args kv_fp8_args = {'
 block='        if (lifetime_trace) {\n'+''.join(f'            rs_lifetime_register({n}, @"{n}", {"YES" if n in persistent else "NO"}, position_start);\n' for n in names)+f'            {snap("registered")}\n        }}\n'
 edit(before,block+before)
 before='            [context.prefillBoundaryCommands addObject:command];'
 edit(before,before+f'\n            if (lifetime_trace) {snap("submitted")}')
 before='                context.prefillBoundaryCommands = nil;'
 edit(before,f'                if (lifetime_trace) {snap("waited")}\n'+before)
 before='        return 1;\n    }\n}\n'
 after='    }\n    RustStarMetalContext *context = (__bridge RustStarMetalContext *)opaque_context;\n    if (!collect_outputs && continue_layer2_compressors && kv_state_mode == 2u && context.prefillLayerChunkStart != nil) {\n        '+snap('after_pool')+'\n    }\n    return 1;\n}\n'
 edit(before,after)
 full=s[:start]+f+s[end:];anchor='int rust_star_metal_run_prefill_layer0_boundary(';assert full.count(anchor)==1;full=full.replace(anchor,helper+'\n'+anchor)
 check=f
 for p in reversed(patches):assert check.count(p['after'])==1;check=check.replace(p['after'],p['before'])
 assert check==original
 (w/(role+'-instrumented.m')).write_text(full)
 manifests[role]=dict(original_sha256=sha(s.encode()),instrumented_sha256=sha(full.encode()),observer_sha256=sha(helper.encode()),patches=patches,names=names,persistent=sorted(persistent),scratch=sorted(set(names)-persistent),inverse_exact=True)
assert manifests['accepted']['patches']==manifests['wide']['patches']
(w/'instrumentation.json').write_text(json.dumps(manifests,indent=2)+'\n')
print('Identical reversible observation patches; buffers',len(names),'persistent',len(persistent))
