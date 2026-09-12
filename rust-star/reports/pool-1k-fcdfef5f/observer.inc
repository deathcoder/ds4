// Diagnostic only. The ledger never owns an MTLBuffer between calls.
#import <mach/mach.h>
@interface RSLifetimeRecord : NSObject
@property(nonatomic, weak) id<MTLBuffer> buffer;
@property(nonatomic, copy) NSString *label;
@property(nonatomic) NSUInteger bytes;
@property(nonatomic) NSUInteger position;
@property(nonatomic) BOOL persistent;
@end
@implementation RSLifetimeRecord
@end
static NSMutableArray<RSLifetimeRecord *> *rs_lifetime_records;
static NSUInteger rs_lifetime_registered_count;
static NSUInteger rs_lifetime_registered_bytes;
static NSUInteger rs_lifetime_dead_count;
static void rs_lifetime_register(id<MTLBuffer> buffer, NSString *label,
                                 BOOL persistent, NSUInteger position) {
    @autoreleasepool {
        if (!buffer) return;
        if (!rs_lifetime_records) rs_lifetime_records = [NSMutableArray new];
        for (RSLifetimeRecord *record in rs_lifetime_records) {
            if (record.buffer == buffer) return;
        }
        RSLifetimeRecord *record = [RSLifetimeRecord new];
        record.buffer = buffer;
        record.label = label;
        record.bytes = buffer.length;
        record.position = position;
        record.persistent = persistent;
        [rs_lifetime_records addObject:record];
        rs_lifetime_registered_count++;
        rs_lifetime_registered_bytes += record.bytes;
    }
}
static void rs_lifetime_snapshot(id<MTLDevice> device,
    NSArray<id<MTLCommandBuffer>> *commands, NSString *phase,
    NSUInteger rows, NSUInteger position, NSUInteger frontier) {
    @autoreleasepool {
        struct task_vm_info vm = {0};
        mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
        kern_return_t vm_result = task_info(mach_task_self(), TASK_VM_INFO,
            (task_info_t)&vm, &vm_count);
        NSUInteger device_bytes = device.currentAllocatedSize;
        NSMutableDictionary *labels = [NSMutableDictionary new];
        NSMutableIndexSet *dead = [NSMutableIndexSet new];
        NSUInteger index = 0, scratch = 0, persistent = 0, previous_scratch = 0;
        for (RSLifetimeRecord *record in rs_lifetime_records) {
            // This temporary strong reference ends before returning to runtime.
            id<MTLBuffer> live = record.buffer;
            if (!live) { [dead addIndex:index++]; continue; }
            NSMutableDictionary *entry = labels[record.label];
            if (!entry) {
                entry = [@{@"count":@0, @"bytes":@0,
                    @"oldest_position":@(record.position),
                    @"newest_position":@(record.position),
                    @"persistent":@(record.persistent)} mutableCopy];
                labels[record.label] = entry;
            }
            entry[@"count"] = @([entry[@"count"] unsignedLongLongValue]+1u);
            entry[@"bytes"] = @([entry[@"bytes"] unsignedLongLongValue]+record.bytes);
            entry[@"oldest_position"] = @(MIN([entry[@"oldest_position"] unsignedLongLongValue],record.position));
            entry[@"newest_position"] = @(MAX([entry[@"newest_position"] unsignedLongLongValue],record.position));
            if (record.persistent) persistent += record.bytes;
            else {
                scratch += record.bytes;
                if (record.position < position) previous_scratch += record.bytes;
            }
            index++;
        }
        rs_lifetime_dead_count += dead.count;
        [rs_lifetime_records removeObjectsAtIndexes:dead];
        NSUInteger status[7] = {0};
        for (id<MTLCommandBuffer> command in commands) {
            NSUInteger state = command.status;
            status[state <= 5 ? state : 6]++;
        }
        NSDictionary *row = @{@"phase":phase,@"rows":@(rows),
            @"position":@(position),@"frontier":@(frontier),
            @"registered_count":@(rs_lifetime_registered_count),
            @"registered_bytes":@(rs_lifetime_registered_bytes),
            @"dead_count":@(rs_lifetime_dead_count),
            @"live_count":@(rs_lifetime_records.count),
            @"scratch_bytes":@(scratch),@"persistent_bytes":@(persistent),
            @"previous_scratch_bytes":@(previous_scratch),
            @"device_allocated_bytes":@(device_bytes),
            @"vm_result":@(vm_result),@"vm_count":@(vm_count),
            @"phys_footprint_bytes":@(vm.phys_footprint),
            @"resident_size_bytes":@(vm.resident_size),
            @"commands":@(commands.count),
            @"command_status_counts":@[@(status[0]),@(status[1]),@(status[2]),
                @(status[3]),@(status[4]),@(status[5]),@(status[6])],
            @"labels":labels};
        NSError *error = nil;
        NSData *json = [NSJSONSerialization dataWithJSONObject:row options:NSJSONWritingSortedKeys error:&error];
        if (!json) { fprintf(stderr,"RS_LIFETIME_ERROR\n"); abort(); }
        fprintf(stderr,"RS_LIFETIME %.*s\n",(int)json.length,(const char *)json.bytes);
    }
}
