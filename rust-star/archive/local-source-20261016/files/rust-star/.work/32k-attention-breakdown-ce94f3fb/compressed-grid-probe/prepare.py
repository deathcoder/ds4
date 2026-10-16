from pathlib import Path
import re,hashlib,json
w=Path(__file__).resolve().parent;root=w.parents[3]
flash=(root/'metal/flash_attn.metal').read_text()
start=flash.index('// Batched FlashAttention for prompt/prefill rows.')
end=flash.index('#undef FA_NONVEC_TYPES',start)+len('#undef FA_NONVEC_TYPES')
original=flash[start:end]+'\n'
candidate=re.sub(r'\bkernel_flash_attn_ext\b','probe_flash_head_first_impl',original)
candidate=candidate.replace('flash_attn_ext_dk512_t','probe_flash_head_first_512_t').replace('flash_attn_ext_dk256_t','probe_flash_head_first_256_t')
candidate=candidate.replace('kernel_flash_attn_ext_f16_dk512_dv512','probe_flash_head_first').replace('kernel_flash_attn_ext_f16_dk256_dv256','probe_flash_head_first_256')
old='#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg'
new=old.replace('tgpig,','uint3(tgpig.y, tgpig.x, tgpig.z),')
assert candidate.count(old)==1;candidate=candidate.replace(old,new)
restored=candidate.replace(new,old).replace('probe_flash_head_first_512_t','flash_attn_ext_dk512_t').replace('probe_flash_head_first_256_t','flash_attn_ext_dk256_t').replace('probe_flash_head_first_impl','kernel_flash_attn_ext').replace('"probe_flash_head_first"','"kernel_flash_attn_ext_f16_dk512_dv512"').replace('"probe_flash_head_first_256"','"kernel_flash_attn_ext_f16_dk256_dv256"')
assert restored==original
(w/'baseline-wrapper.metal').write_text(original);(w/'candidate-wrapper.metal').write_text(candidate)
# Exact library assembly order used by runtime/build.rs for attention ingress.
files=['rust-star/runtime/src/attention_ingress.metal','rust-star/runtime/src/ffn_router.metal','metal/dsv4_rope.metal','metal/dsv4_kv.metal','metal/cpy.metal','metal/flash_attn.metal','metal/softmax.metal','metal/sum_rows.metal']
combined='\n'.join((root/p).read_text() for p in files)+'\n'+candidate
(w/'combined.metal').write_text(combined)
structs=[]
for name in ['ds4_metal_args_flash_attn_ext_pad','ds4_metal_args_flash_attn_ext_blk','ds4_metal_args_flash_attn_ext']:
 a=flash.index('struct '+name+' {');b=flash.index('\n};',a)+3;structs.append(flash[a:b])
(w/'arguments.h').write_text('#include <stdint.h>\n'+'\n\n'.join(structs)+'\n')
sha=lambda b:hashlib.sha256(b).hexdigest()
r=dict(passed=True,scope='Unchanged original FlashAttention implementation; standalone wrapper forwards swapped grid axes only. Original wrapper restored exactly.',sources={p:sha((root/p).read_bytes()) for p in files},baseline_wrapper_sha256=sha(original.encode()),candidate_wrapper_sha256=sha(candidate.encode()),combined_sha256=sha(combined.encode()))
(w/'source-audit.json').write_text(json.dumps(r,indent=2)+'\n');print('Prepared exact wrapper and original library assembly')
