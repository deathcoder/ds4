from pathlib import Path
import hashlib,json,re
w=Path(__file__).resolve().parent;root=w.parents[2];parent=root/'rust-star/reports/compressed-grid-warm-c21ebfda'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((parent/'parent/source-audit.json').read_text())['sources']
for n,h in inputs.items():assert sha(root/n)==h,n
flash=(root/'metal/flash_attn.metal').read_text();start=flash.index('// DS4 non-vector FlashAttention.');end=flash.index('#undef FA_NONVEC_TYPES',start)+len('#undef FA_NONVEC_TYPES');original=flash[start:end]+'\n'
renames=[('kernel_flash_attn_ext_impl','probe_flash_skip_runs_impl'),('kernel_flash_attn_ext_f16_dk512_dv512','probe_flash_skip_runs'),('kernel_flash_attn_ext_f16_dk256_dv256','probe_flash_skip_runs_256'),('flash_attn_ext_dk512_t','probe_flash_skip_runs_512_t'),('flash_attn_ext_dk256_t','probe_flash_skip_runs_256_t'),('kernel_flash_attn_ext','probe_flash_skip_runs_ext')]
candidate=original
for a,b in renames:candidate=re.sub(r'\b'+a+r'\b',b,candidate)
anchor='''        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;'''
insertion='''        for (int ic0 = 0; ; ++ic0) {
            // Skip only complete zero-marked blocks. Leave the partial padded
            // tail on the original path, and preserve every active-block step.
            if (FC_flash_attn_ext_has_mask) {
                const int nfull = args.ne11/C;
                const int first = ic0;
                while (ic0 + 4 <= nfull &&
                       (blk[ic0] | blk[ic0 + 1] | blk[ic0 + 2] | blk[ic0 + 3]) == 0) {
                    ic0 += 4;
                }
                while (ic0 < nfull && blk[ic0] == 0) {
                    ++ic0;
                }
                if (ic0 != first) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += (ic0 - first)*NW;
                    }
                }
            }
            int ic = ic0*C;'''
assert candidate.count(anchor)==1;candidate=candidate.replace(anchor,insertion)
restored=candidate.replace(insertion,anchor)
for a,b in reversed(renames):restored=re.sub(r'\b'+b+r'\b',a,restored)
assert restored==original
(w/'baseline-kernel.metal').write_text(original);(w/'candidate-kernel.metal').write_text(candidate)
combined='\n'.join((root/n).read_text() for n in inputs)+'\n'+candidate;(w/'combined.metal').write_text(combined)
(w/'arguments.h').write_bytes((parent/'arguments.h').read_bytes())
(w/'parent-shape-audit.json').write_bytes((parent/'parent-shape-audit.json').read_bytes())
# The timing/check harness is unchanged except the candidate symbol and removal
# of the old grid transpose. The role tag now describes candidate selection.
s=(parent/'probe.m').read_text().replace('probe_flash_head_first','probe_flash_skip_runs').replace('transposed?MTLSizeMake(HEADS,ROWS/8,1):MTLSizeMake(ROWS/8,HEADS,1)','MTLSizeMake(ROWS/8,HEADS,1)').replace('transposed','candidate_variant').replace('rust-star-compressed-grid-warm-v1','rust-star-compressed-skip-runs-v1')
(w/'probe.m').write_text(s)
proof=dict(passed=True,scope='Only the integer/metadata run-skip insertion and renamed symbols differ from the original complete FlashAttention template. All active-block arithmetic and padded-tail code restore byte-for-byte.',sources=inputs,parent_probe_sha256=sha(parent/'probe.m'),baseline_kernel_sha256=sha(w/'baseline-kernel.metal'),candidate_kernel_sha256=sha(w/'candidate-kernel.metal'),combined_sha256=sha(w/'combined.metal'),insertion=insertion,anchor=anchor,renames=renames)
(w/'source-audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('Prepared exact run-skip source and inherited timing harness')
