#ifndef DS4_TCIM_XH2RT_KERNEL_H
#define DS4_TCIM_XH2RT_KERNEL_H

/* Shared host/device contract. HDPL_CC selects the device implementation;
 * ordinary host C translation units receive the launch interface. */
#include <stddef.h>
#include <stdint.h>

#define XH2RT_CORE_COUNT UINT32_C(2)
#define XH2RT_TILE_COUNT UINT32_C(4)
#define XH2RT_HARTS_PER_TILE UINT32_C(4)

#define XH2RT_CORE_TILE_COUNT (XH2RT_CORE_COUNT * XH2RT_TILE_COUNT)
#define XH2RT_HART_COUNT (XH2RT_CORE_TILE_COUNT * XH2RT_HARTS_PER_TILE)
#define XH2RT_KERNEL_COMPLETION_RECORD_BYTES (XH2RT_HART_COUNT * sizeof(uint32_t))

#define XH2RT_COMPLETION_UNWRITTEN UINT32_MAX

#define XH2RT_KERNEL_OUTCOME_OK UINT32_C(0)
#define XH2RT_KERNEL_OUTCOME_INVALID_ARGUMENT UINT32_C(1)
#define XH2RT_KERNEL_OUTCOME_PRIMITIVE_FAILURE UINT32_C(2)

#ifndef HDPL_CC

#include "xh2rt.h"

#define XH2RT_KERNEL_MAX_SLOTS UINT32_C(64)

/* Only typed-wrapper preflight failures cross this private host seam. */
enum xh2rt_kernel_rejection {
    XH2RT_KERNEL_REJECT_OVERFLOW = 0,
    XH2RT_KERNEL_REJECT_OUT_OF_RANGE
};

enum xh2rt_kernel_slot_kind {
    XH2RT_KERNEL_SLOT_WORD = 0,
    XH2RT_KERNEL_SLOT_BUFFER
};

/* Buffer slots describe the complete accessed range. The runtime resolves the
 * IOMAP word, validates alignment and owns the allocation pin through group
 * completion. Word slots are copied unchanged into the native envelope. */
typedef struct xh2rt_kernel_slot {
    enum xh2rt_kernel_slot_kind kind;
    uint64_t word;
    xh2rt_buffer_view *buffer;
    uint64_t offset;
    uint64_t bytes;
    uint64_t alignment;
} xh2rt_kernel_slot;

static inline xh2rt_kernel_slot xh2rt_kernel_word(uint64_t word)
{
    xh2rt_kernel_slot slot = {
        XH2RT_KERNEL_SLOT_WORD, word, NULL, 0, 0, 0
    };

    return slot;
}

static inline xh2rt_kernel_slot xh2rt_kernel_buffer(
    xh2rt_buffer_view *buffer, uint64_t offset, uint64_t bytes,
    uint64_t alignment)
{
    xh2rt_kernel_slot slot = {
        XH2RT_KERNEL_SLOT_BUFFER, 0, buffer, offset, bytes, alignment
    };

    return slot;
}

int xh2rt_kernel_reject(xh2rt_context *context,
                        enum xh2rt_kernel_rejection rejection,
                        const char *operation);
/* submit == 0 validates slots without creating a HAL group. */
int xh2rt_kernel_invoke(xh2rt_context *context, const char *name,
                        const xh2rt_kernel_slot *slots, size_t slot_count,
                        int submit);

#endif

#ifdef HDPL_CC

#ifndef __cplusplus
#error "xh2rt kernel device interface requires HDPL C++"
#endif

#ifndef TARGET_XH2
#error "xh2rt kernel device interface requires TARGET_XH2"
#endif

#include "device/target/DeviceActiveInfo.h"
#include "device/BarrierUtils.h"
#include "device/DeviceBuiltinIntrinsic.h"
#include "device/DeviceContext.h"
#include "device/DeviceInstrument.h"

struct xh2rt_kernel_scope {
    device::target::ActiveCoreInfo active_info;
    device::Context context;
    volatile uint32_t *hart_outcome;
    uint32_t core_tile_index;
    uint32_t hart_lane;
    uint32_t hart_index;
    bool completion_publishable;
};

static __device__ inline bool xh2rt_kernel_completion_address_well_formed(
        uint64_t address) {
    return address != 0 && (address & (sizeof(uint32_t) - 1)) == 0 &&
           address <= UINT64_MAX -
               (XH2RT_KERNEL_COMPLETION_RECORD_BYTES - 1);
}

/* Scope is a fresh, value-initialized caller-owned local whose lifetime reaches
 * finish(); the fixed XH2 active topology is an invocation invariant. */
static __device__ inline uint32_t xh2rt_kernel_scope_begin(
        xh2rt_kernel_scope &scope,
        uint64_t completion_iomap,
        uint32_t primitive_inputs) {
    scope.active_info = device::target::getActiveInfo();
    scope.active_info.coreNum = XH2RT_CORE_COUNT;
    const bool topology_addressable =
        scope.active_info.logicalCoreId < XH2RT_CORE_COUNT &&
        scope.active_info.tileId < XH2RT_TILE_COUNT;

    scope.context.deviceId = 0;
    scope.context.corePhyId = topology_addressable
        ? scope.active_info.physicalCoreId[scope.active_info.logicalCoreId]
        : XH2RT_CORE_COUNT;
    const bool topology_valid = topology_addressable &&
        scope.context.corePhyId < XH2RT_CORE_COUNT;
    scope.context.coreId = scope.active_info.logicalCoreId;
    scope.context.tileId = scope.active_info.tileId;
    scope.context.hartId = device::target::getHartId();
    scope.context.setChipVersion();
    scope.context.compatVersion = 1;
    scope.context.hartLogicId = scope.context.hartId %
        (XH2RT_TILE_COUNT * XH2RT_HARTS_PER_TILE);
    scope.context.enabledRvvNumPerTile = XH2RT_HARTS_PER_TILE;
    scope.context.enabledVpNumPerTile = XH2RT_HARTS_PER_TILE;
    scope.context.numOfCores = XH2RT_CORE_COUNT;
    scope.context.numOfMpus = 0;
    scope.context.numOfMpuGroup = 0;
    scope.context.numOfInput = primitive_inputs;
    scope.context.stopAtPrimitiveOpIndex = UINT64_MAX;
    scope.context.activeInfo = (void *)&scope.active_info;

    /* Do not call clearIndicatorsInAllCores() from this per-hart prologue: on
     * XH2 it races consecutive invocations. finish() owns the barrier path. */
    scope.hart_outcome = reinterpret_cast<volatile uint32_t *>(
        completion_iomap);
    scope.core_tile_index =
        scope.context.corePhyId * XH2RT_TILE_COUNT + scope.context.tileId;
    scope.hart_lane = scope.context.hartId % XH2RT_HARTS_PER_TILE;
    scope.hart_index =
        scope.core_tile_index * XH2RT_HARTS_PER_TILE + scope.hart_lane;
    scope.completion_publishable =
        topology_valid &&
        xh2rt_kernel_completion_address_well_formed(completion_iomap) &&
        scope.hart_index < XH2RT_HART_COUNT;

    return scope.completion_publishable
        ? XH2RT_KERNEL_OUTCOME_OK
        : XH2RT_KERNEL_OUTCOME_INVALID_ARGUMENT;
}

/* Every hart and every operator branch must reach this single epilogue. */
static __device__ inline void xh2rt_kernel_scope_finish(
        xh2rt_kernel_scope &scope,
        uint32_t outcome) {
    device::allRvWaitForEverythingDoneInThisCore(scope.context);

    uint32_t final_outcome = outcome;
    if (final_outcome == XH2RT_KERNEL_OUTCOME_OK &&
        scope.context.unsupportedOpNum != 0)
        final_outcome = XH2RT_KERNEL_OUTCOME_PRIMITIVE_FAILURE;

    if (scope.completion_publishable)
        scope.hart_outcome[scope.hart_index] = final_outcome;
    RISCV_FENCE_RW_RW;
    device::barrierForRVsInThisTile(scope.context);
    RISCV_FENCE_RW_RW;

    device::barrierForTilesInAllCores(scope.context, true);
    device::waitForEverythingDoneInThisCoreBeforeKernelExit(scope.context);
}

#endif

#endif
