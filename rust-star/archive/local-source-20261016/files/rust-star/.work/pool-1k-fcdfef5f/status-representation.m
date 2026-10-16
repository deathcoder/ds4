#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
int main(void) { @autoreleasepool {
 id<MTLCommandBuffer> command=nil;
 NSNumber *value=@(command ? command.status : -1);
 NSData *data=[NSJSONSerialization dataWithJSONObject:@{@"no_prior_status":value,@"NSUIntegerMax":@(NSUIntegerMax)} options:0 error:nil];
 printf("%.*s\n",(int)data.length,(const char *)data.bytes);
 return 0;
}}
