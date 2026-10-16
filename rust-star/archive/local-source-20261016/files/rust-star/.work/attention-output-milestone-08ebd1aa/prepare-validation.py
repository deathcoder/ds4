from pathlib import Path
import json,re,shutil
w=Path(__file__).resolve().parent;root=w.parents[2];d=w/'integration';d.mkdir(exist_ok=True)
prev=root/'rust-star/.work/32k-compressed-skip-2f06a2e8'
s=(prev/'preflight.py').read_text().replace("w=Path(__file__).resolve().parent;root=w.parents[2]","w=Path(__file__).resolve().parent;root=w.parents[3]")
s=s.replace("32k-attention-breakdown-ce94f3fb/candidate-rust-star","32k-compressed-skip-2f06a2e8/candidate-rust-star").replace('a174c1e06ff399056b5946bbac1a34583e43ec7aaac0b078c280e4f04a81f3c8','631b4844599d814ebfcae6706d26c0854169d4573feed86391c41880f71cfd99')
(d/'preflight.py').write_text(s)
s=(prev/'validate.py').read_text().replace('root = w.parents[2]','root = w.parents[3]').replace("queue = [('candidate-32k-output','candidate-rust-star',False),('baseline-32k-profiled','baseline-rust-star',True),('candidate-32k-profiled','candidate-rust-star',True)]","queue = [('candidate-32k-output','candidate-rust-star',False)]").replace('ALL THREE 32K C0 RUNS PASS','FULL 32K C0 RUN PASSES')
(d/'validate.py').write_text(s)
p=json.loads((w/'engine-plan.json').read_text());p.update(live_sequence=['candidate-32k-output'],full_fp32_words_per_run=16677120)
(d/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
(d/'compile-shader.m').write_text('''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc,const char**argv){@autoreleasepool{
 if(argc!=2)return 1;NSError*error=nil;id<MTLDevice>d=MTLCreateSystemDefaultDevice();
 NSString*s=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
 id<MTLLibrary>l=s?[d newLibraryWithSource:s options:[MTLCompileOptions new] error:&error]:nil;
 if(!l){fprintf(stderr,"%s\\n",error.description.UTF8String);return 2;}
 bool no=false;MTLFunctionConstantValues*v=[MTLFunctionConstantValues new];
 [v setConstantValue:&no type:MTLDataTypeBool atIndex:700];[v setConstantValue:&no type:MTLDataTypeBool atIndex:701];
 for(NSString*n in @[@"kernel_mul_mm_id_q8_0_f16",@"rust_star_attention_output_half"]){
 id<MTLFunction>f=[l newFunctionWithName:n constantValues:v error:&error];
 id<MTLComputePipelineState>p=f?[d newComputePipelineStateWithFunction:f error:&error]:nil;
 if(!p||p.threadExecutionWidth!=32||p.maxTotalThreadsPerThreadgroup<128){fprintf(stderr,"%s: %s\\n",n.UTF8String,error.description.UTF8String);return 3;}}
 puts("{\\"passed\\":true,\\"production_moe_library\\":true,\\"both_half_projection_pipelines\\":true,\\"bounds_input\\":false,\\"bounds_output\\":false}");return 0;}}
''')
# Reconstruct the exact production assembly, including build.rs's newline joining.
build=(root/'rust-star/runtime/build.rs').read_text();end=build.index('"kMoeOutputSource"');start=build.rfind('write_metal_source_include(',0,end)
paths=re.findall(r'manifest.join\("([^"]+)"\)',build[start:end]);assert 'src/attention_output_half.metal' in paths
src=''.join(line+'\n' for p in paths for line in (root/'rust-star/runtime'/p).read_text().splitlines())
(d/'assembled.metal').write_text(src)
(d/'assembly.json').write_text(json.dumps(dict(paths=paths),indent=2)+'\n')
print('Prepared production assembly, C0 runner and preflight')
