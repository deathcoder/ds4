#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc,const char**argv){@autoreleasepool{
 if(argc!=2)return 1;NSError*error=nil;id<MTLDevice>d=MTLCreateSystemDefaultDevice();
 NSString*s=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
 id<MTLLibrary>l=s?[d newLibraryWithSource:s options:[MTLCompileOptions new] error:&error]:nil;
 if(!l){fprintf(stderr,"%s\n",error.description.UTF8String);return 2;}
 bool no=false;MTLFunctionConstantValues*v=[MTLFunctionConstantValues new];
 [v setConstantValue:&no type:MTLDataTypeBool atIndex:700];[v setConstantValue:&no type:MTLDataTypeBool atIndex:701];
 for(NSString*n in @[@"kernel_mul_mm_id_q8_0_f16",@"rust_star_attention_output_half"]){
 id<MTLFunction>f=[l newFunctionWithName:n constantValues:v error:&error];
 id<MTLComputePipelineState>p=f?[d newComputePipelineStateWithFunction:f error:&error]:nil;
 if(!p||p.threadExecutionWidth!=32||p.maxTotalThreadsPerThreadgroup<128){fprintf(stderr,"%s: %s\n",n.UTF8String,error.description.UTF8String);return 3;}}
 puts("{\"passed\":true,\"production_moe_library\":true,\"both_half_projection_pipelines\":true,\"bounds_input\":false,\"bounds_output\":false}");return 0;}}
