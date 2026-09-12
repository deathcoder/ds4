from pathlib import Path
import json,re,hashlib
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());p=read(w/'plan.json');s=(w/'candidate-metal_shim.m').read_text();a=s.index('int rust_star_metal_run_prefill_layer0_boundary(');b=s.index('\nint ',a+4);f=s[a:b];base=s[:a].count('\n');sizes={x['label']:x['bytes'] for x in read(w/'inventory.json')};rows=[]
assert 'q_rank = 1024, kv_dim = 512, q_dim = 32768,' in f
assert 'const NSUInteger q_bytes = rows*q_dim*sizeof(float);' in f
assert 'const BOOL native_1k_bootstrap = rows == 1024u && !collect_outputs' in f
assert 'complete_layer1 && continue_layer2_compressors' in f
for dst,src in p['copy_pairs'].items():
 expr=re.compile(r'\[blit copyFromBuffer:'+src+r' sourceOffset:0\s+toBuffer:'+dst+r'\s+destinationOffset:0 size:q_bytes\];');matches=list(expr.finditer(f));assert len(matches)==1;copy=matches[0];before=f[:copy.start()]
 assert not re.search(r'\b'+dst+r'\.contents',before)
 assert not re.search(r'(setBuffer:|copyFromBuffer:)'+dst+r'\b',before)
 assert not re.search(r'\b\w+\s*=\s*'+dst+r'\s*;',before)
 alloc=re.compile(r'\b'+dst+r' = bootstrap_scratch_buffer\(context.device, scratch_pool,\s*@"'+dst+r'", \(q_bytes\), NULL, error, error_bytes\);');assert len(list(alloc.finditer(before)))==1
 # Complete object-reference inventory makes unexpected access reviewable.
 refs=[dict(line=base+i+1,text=line.strip()) for i,line in enumerate(f.splitlines()) if re.search(r'\b'+dst+r'\b',line)]
 content_refs=[r for r in refs if '.contents' in r['text'] or 'setBuffer:' in r['text'] or 'COPY_TRANSITION' in r['text'] or 'copyFromBuffer:' in r['text'] or 'toBuffer:' in r['text']]
 copyline=base+f[:copy.start()].count('\n')+1
 assert min(r['line'] for r in content_refs)>=copyline
 assert sizes[src]==sizes[dst]==1024*32768*4==134217728
 rows.append(dict(destination=dst,source=src,bytes=sizes[dst],copy_line=copyline,copy_expression=copy[0],references=refs,full_range=[0,sizes[dst]],first_content_access_is_full_copy=True))
result=dict(passed=True,candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),selected_labels=sorted(p['copy_pairs']),selected_bytes=sum(x['bytes'] for x in rows),buffers=rows,scope='Native1024-row branch only, preserving source buffers and every other initialization. This proves destination initial bytes irrelevant; it makes no assertion about full writes of Q-raw, attention-output or the other101 buffers.',synchronization='Retain existing prior-command checked completion. Source producer encoder ends before copy; copy encoder ends before any destination read. Failed encoder creation returns without submission/consumption.')
(w/'source-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='buffers'},indent=2))
