from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;root=w.parents[3];r=json.loads((w.parent/'profile-audit.json').read_text());assert r['passed'];assert r['stages']['bootstrap_ingress']['gpu_ms']/sum(x['gpu_ms'] for x in r['stages'].values())>=.10
allops={}
def edit(name,fn):
 p=root/name;s=p.read_text();original=s;ops=[]
 def replace(a,b):
  nonlocal s
  assert s.count(a)==1,(name,a[:140],s.count(a));s=s.replace(a,b);ops.append(dict(before=a,after=b))
 fn(replace,s);assert s!=original;p.write_text(s);allops[name]=ops
 expected=subprocess.check_output(['git','show','4ced6ee8:'+name],cwd=root,text=True);assert original==expected,name

def rust(replace,s):
 old='''        if rows != 32 && rows != 64 {
            return Err(Error::invalid("invalid prefill boundary tile width"));
        }'''
 new='''        let native_wide = rows == 4096
            && !collect_outputs
            && complete_layer1
            && include_layer2_compressors
            && kv_state_mode == 2
            && shared_context.is_some()
            && token_source.is_some()
            && position_start > 0
            && prefill_rows <= 32768
            && position_start.checked_add(rows) == Some(prefill_rows);
        if rows != 32 && rows != 64 && !native_wide {
            return Err(Error::invalid("invalid prefill boundary tile width"));
        }'''
 replace(old,new)
 a=s.index('    fn run_prefill_layers012_compressor_loop_probe_in_context(');b=s.index('        if position_begin > prefill_rows',a);old=s[a:b]
 signature=old[:old.index('        let bootstrap_started')]
 wrapper=signature+'''        let tile_rows = if collect_outputs { 32 } else { 64 };
        run_prefill_layers012_compressor_loop_with_tile_rows(
            model, context, collect_outputs, position_begin, prefill_rows, tokens, tile_rows,
        )
    }

'''
 wide=signature.replace('run_prefill_layers012_compressor_loop_probe_in_context','run_prefill_layers012_compressor_loop_with_tile_rows').replace('        tokens: Option<&[u32]>,\n','        tokens: Option<&[u32]>,\n        tile_rows: usize,\n')
 wide+='''        let bootstrap_started = Instant::now();
        if (collect_outputs && tile_rows != 32)
            || (!collect_outputs && tile_rows != 64 && tile_rows != 4096)
        {
            return Err(Error::invalid("invalid compressor loop tile width"));
        }
'''
 replace(old,wrapper+wide)
edit('rust-star/runtime/src/metal.rs',rust)

def long(replace,s):
 old='''        let (_, bootstrap_timing) = run_prefill_layers012_compressor_loop_probe_in_context(
            model,
            &context,
            false,
            start,
            end,
            Some(&tokens[..end as usize]),
        )?;'''
 new='''        // Only the validated lower-context native continuation uses a 4K
        // bootstrap batch. Prefix, legacy callers and larger frontiers keep 64.
        let bootstrap_tile_rows = if frontier <= 32768 { 4096 } else { 64 };
        let (_, bootstrap_timing) = run_prefill_layers012_compressor_loop_with_tile_rows(
            model,
            &context,
            false,
            start,
            end,
            Some(&tokens[..end as usize]),
            bootstrap_tile_rows,
        )?;'''
 replace(old,new)
edit('rust-star/runtime/src/long_prefill.rs',long)

def host(replace,s):
 old='''    if (n_vocab != 129280u || (rows != 32u && rows != 64u) ||
        (prefill_rows != 2048u'''
 new='''    const BOOL native_wide_bootstrap = rows == 4096u && !collect_outputs &&
        complete_layer1 && continue_layer2_compressors && kv_state_mode == 2u &&
        position_start > 0u && prefill_rows <= 32768u &&
        (uint64_t)position_start + rows == prefill_rows;
    if (n_vocab != 129280u ||
        (rows != 32u && rows != 64u && !native_wide_bootstrap) ||
        (prefill_rows != 2048u'''
 replace(old,new)
 old='''        const BOOL retain_kv_state = kv_state_mode == 1u;
        const BOOL consume_kv_state = kv_state_mode == 2u;'''
 new='''        if (native_wide_bootstrap && (!context.prefillLayerChunkStart ||
            context.prefillLayerChunkStart.unsignedIntValue != position_start)) {
            return fail_with_message(error, error_bytes,
                @"wide bootstrap requires the active native continuation chunk");
        }
'''+old
 replace(old,new)
 replace('        int32_t positions[64];','        int32_t positions[4096];')
 replace('        int32_t compressed_positions[16];','        int32_t compressed_positions[1024];')
 for indent,prefix in [('        ',''),('            ','next_')]:
  old=indent+'''[encoder setComputePipelineState:context.cpyF16F32Pipeline];
'''+indent+'''[encoder setBytes:&kv_elements length:sizeof(kv_elements) atIndex:0];
'''+indent+f'''[encoder setBuffer:{prefix}kv_half_buffer offset:0 atIndex:1];
'''+indent+f'''[encoder setBuffer:{prefix}raw_cache_buffer
'''+indent+'''             offset:raw_cache_target_row*kv_dim*sizeof(float) atIndex:2];
'''+indent+'''[encoder dispatchThreadgroups:MTLSizeMake(conversion_groups,1,1)
'''+indent+'''     threadsPerThreadgroup:MTLSizeMake(256,1,1)];'''
  new=indent+'''// A wide batch leaves exactly its final 128 rows in the raw ring.
'''+indent+'''const uint32_t raw_copy_rows = MIN(rows, (uint32_t)raw_cache_rows);
'''+indent+'''const uint32_t raw_copy_elements = raw_copy_rows*kv_dim;
'''+indent+'''const NSUInteger raw_source_row = rows-raw_copy_rows;
'''+indent+'''const NSUInteger raw_target_row =
'''+indent+'''    (position_start+raw_source_row)%raw_cache_rows;
'''+indent+'''[encoder setComputePipelineState:context.cpyF16F32Pipeline];
'''+indent+'''[encoder setBytes:&raw_copy_elements length:sizeof(raw_copy_elements) atIndex:0];
'''+indent+f'''[encoder setBuffer:{prefix}kv_half_buffer
'''+indent+'''             offset:raw_source_row*kv_dim*sizeof(uint16_t) atIndex:1];
'''+indent+f'''[encoder setBuffer:{prefix}raw_cache_buffer
'''+indent+'''             offset:raw_target_row*kv_dim*sizeof(float) atIndex:2];
'''+indent+'''[encoder dispatchThreadgroups:MTLSizeMake((raw_copy_elements+1023u)/1024u,1,1)
'''+indent+'''     threadsPerThreadgroup:MTLSizeMake(256,1,1)];'''
  replace(old,new)
 # The old derived value now has no consumers; its replacement above is exact
 # for 32/64 rows and the final aligned 128 rows of a 4096-row batch.
 replace('        const uint32_t raw_cache_target_row = position_start % raw_cache_rows;\n','')
edit('rust-star/runtime/src/metal_shim.m',host)
(w/'source-edits.json').write_text(json.dumps(allops,indent=2)+'\n')
(w/'source-selection.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile_audit_sha256=hashlib.sha256((w.parent/'profile-audit.json').read_bytes()).hexdigest(),target_gpu_ms=r['stages']['bootstrap_ingress']['gpu_ms'],target_recorded_stage_share=r['stages']['bootstrap_ingress']['gpu_ms']/sum(x['gpu_ms'] for x in r['stages'].values()),baseline_source_commit='4ced6ee8',files=list(allops)),indent=2)+'\n')
print('Applied scoped native 4K bootstrap candidate to three source files')
