#include "xh2rt.h"

#include <ipu_api.h>
#include <memory_allocator.h>
#include <memory_transfer.h>

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define XH2RT_IOMAP_DELTA UINT64_C(0x1000000000)
#define XH2RT_GROUP_MAX_KERNELS UINT32_C(255)
#define XH2RT_GROUP_SPM_LIMIT UINT64_C(1048576)
#define XH2RT_GROUP_SYNC_TIMEOUT_US UINT32_C(0)

#define XH2RT_CORE_COUNT UINT32_C(2)
#define XH2RT_TILE_COUNT UINT32_C(4)

/* Complete declarations make empty/truncated xxd output fail host compilation. */
#include "fill_f32.hex"
#include "add_f32.hex"

enum xh2rt_kernel {
    XH2RT_FILL_F32,
    XH2RT_ADD_F32,
    XH2RT_KERNEL_COUNT
};

static const struct {
    const uint8_t *data;
    size_t bytes;
} xh2rt_kernel_code[XH2RT_KERNEL_COUNT] = {
    [XH2RT_FILL_F32] = {xh2rt_fill_f32_code, sizeof(xh2rt_fill_f32_code)},
    [XH2RT_ADD_F32] = {xh2rt_add_f32_code, sizeof(xh2rt_add_f32_code)}
};

enum xh2rt_context_state {
    XH2RT_CONTEXT_OPEN = 0,
    XH2RT_CONTEXT_CLOSING
};

typedef struct xh2rt_allocation xh2rt_allocation;
typedef struct xh2rt_group xh2rt_group;

struct xh2rt_buffer_view {
    xh2rt_buffer_view *handle;
    xh2rt_allocation *allocation;
    uint64_t offset;
    uint64_t bytes;
    uint64_t references;
    int active;
    int release_pending;
    struct xh2rt_buffer_view *next;
};

struct xh2rt_allocation {
    struct xh2a_memory_allocator_buffer_object bo;
    uint64_t iomap;
    uint64_t accessible_bytes;
    uint64_t logical_references;
    uint64_t inflight;
    int bo_live;
    int quarantined;
    struct xh2rt_allocation *next;
};

struct xh2rt_group {
    group_id_t id;
    uint64_t spm_bytes;
    uint32_t kernel_count;
    xh2rt_allocation **pins;
    size_t pin_count;
    size_t pin_capacity;
    int unknown_quiescence;
    int destroy_pending;
};

struct xh2rt_context {
    enum xh2rt_context_state state;
    fd_handle_t ipu;
    fd_handle_t ddr;
    fd_handle_t transfer;
    int ipu_live;
    int ddr_live;
    int transfer_live;
    xh2rt_allocation *allocations;
    xh2rt_buffer_view *views;
    xh2rt_buffer_view *kernels[XH2RT_KERNEL_COUNT];
    xh2rt_group *group;
    int batch_active;
    int poisoned;
    xh2rt_result last_error;
};

static xh2rt_result xh2rt_drain_group(xh2rt_context *context,
                                      const char *operation);
static void xh2rt_stamp_group(xh2rt_result *result,
                              const xh2rt_group *group);
static void xh2rt_group_release_host(xh2rt_context *context,
                                     xh2rt_group *group);
static int xh2rt_group_reserve_pins(xh2rt_group *group, size_t add);

/*
 * Public view pointers are opaque numeric handles, not wrapper addresses.
 * Never reusing them prevents a stale handle from becoming valid after a
 * context is destroyed and malloc later recycles the old wrapper address.
 */
static pthread_mutex_t xh2rt_handle_lock = PTHREAD_MUTEX_INITIALIZER;
static uintptr_t xh2rt_next_handle = (uintptr_t)UINT32_C(0x10000);

static int xh2rt_assign_handle(xh2rt_buffer_view *view)
{
    uintptr_t value;
    int rc;

    rc = pthread_mutex_lock(&xh2rt_handle_lock);
    if (rc != 0)
        return 0;
    value = xh2rt_next_handle;
    if (value > UINTPTR_MAX - (uintptr_t)16) {
        (void)pthread_mutex_unlock(&xh2rt_handle_lock);
        return 0;
    }
    xh2rt_next_handle = value + (uintptr_t)16;
    (void)pthread_mutex_unlock(&xh2rt_handle_lock);
    view->handle = (xh2rt_buffer_view *)value;
    return 1;
}

static xh2rt_result xh2rt_make_result(enum xh2rt_status status,
                                      const char *operation)
{
    xh2rt_result result;

    memset(&result, 0, sizeof(result));
    result.status = status;
    result.operation = operation;
    result.group_id = -1;
    return result;
}

static xh2rt_result xh2rt_hal_result(const char *operation, int raw_rc,
                                     int saved_errno)
{
    xh2rt_result result =
        xh2rt_make_result(XH2RT_STATUS_HAL_FAILURE, operation);

    result.raw_rc = raw_rc;
    result.saved_errno = saved_errno;
    return result;
}

static xh2rt_result xh2rt_record(xh2rt_context *context,
                                 xh2rt_result result)
{
    if (context != NULL && result.status != XH2RT_STATUS_OK &&
        !context->poisoned)
        context->last_error = result;
    return result;
}

static xh2rt_result xh2rt_poison(xh2rt_context *context,
                                 xh2rt_result primary)
{
    if (context != NULL && !context->poisoned) {
        context->last_error = primary;
        context->poisoned = 1;
        context->batch_active = 0;
    }
    return primary;
}

static int xh2rt_u64_add(uint64_t left, uint64_t right, uint64_t *out)
{
    if (right > UINT64_MAX - left)
        return 0;
    *out = left + right;
    return 1;
}

static void xh2rt_put_le64(uint8_t *bytes, uint64_t value)
{
    unsigned int index;

    for (index = 0; index < 8; index++)
        bytes[index] = (uint8_t)(value >> (index * 8));
}

static int xh2rt_range_fits(uint64_t container, uint64_t offset,
                            uint64_t bytes)
{
    return offset <= container && bytes <= container - offset;
}

static int xh2rt_alignment_valid(uint64_t alignment)
{
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

static xh2rt_buffer_view *xh2rt_find_view(
    xh2rt_context *context, const xh2rt_buffer_view *candidate)
{
    xh2rt_buffer_view *view = NULL;

    if (context == NULL || candidate == NULL)
        return NULL;
    for (view = context->views; view != NULL; view = view->next) {
        if (view->handle == candidate)
            return view;
    }
    return NULL;
}

/* Handles are never reused, so stale detection does not need tombstones.
 * Retain only records that still own a reference, an inflight BO or a retry. */
static void xh2rt_forget_view(xh2rt_context *context,
                              xh2rt_buffer_view *view)
{
    xh2rt_buffer_view **link = &context->views;

    while (*link != view)
        link = &(*link)->next;
    *link = view->next;
    free(view);
}

static void xh2rt_forget_allocation(xh2rt_context *context,
                                    xh2rt_allocation *allocation)
{
    xh2rt_allocation **link = &context->allocations;

    while (*link != allocation)
        link = &(*link)->next;
    *link = allocation->next;
    free(allocation);
}

static xh2rt_result xh2rt_require_open(xh2rt_context *context,
                                       const char *operation)
{
    if (context == NULL)
        return xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT, operation);
    if (context->state != XH2RT_CONTEXT_OPEN)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_STATE, operation));
    return xh2rt_make_result(XH2RT_STATUS_OK, operation);
}

static xh2rt_result xh2rt_require_healthy(xh2rt_context *context,
                                          const char *operation)
{
    xh2rt_result result = xh2rt_require_open(context, operation);

    if (!xh2rt_result_is_ok(result))
        return result;
    if (context->poisoned)
        return xh2rt_make_result(XH2RT_STATUS_POISONED, operation);
    return result;
}

static xh2rt_result xh2rt_require_view(
    xh2rt_context *context, const xh2rt_buffer_view *candidate,
    const char *operation, xh2rt_buffer_view **out_view)
{
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result = xh2rt_require_open(context, operation);

    if (!xh2rt_result_is_ok(result))
        return result;
    view = xh2rt_find_view(context, candidate);
    if (view == NULL || !view->active || view->references == 0 ||
        view->allocation == NULL || !view->allocation->bo_live) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_STALE_BUFFER, operation));
    }
    *out_view = view;
    return result;
}

static xh2rt_result xh2rt_checked_range(
    xh2rt_context *context, const xh2rt_buffer_view *candidate,
    uint64_t offset, uint64_t bytes, const char *operation,
    xh2rt_buffer_view **out_view, uint64_t *out_absolute,
    uint64_t *out_global, uint64_t *out_iomap)
{
    uint64_t absolute;
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result =
        xh2rt_require_view(context, candidate, operation, &view);

    if (!xh2rt_result_is_ok(result))
        return result;
    if (!xh2rt_range_fits(view->bytes, offset, bytes)) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OUT_OF_RANGE, operation));
    }
    if (!xh2rt_u64_add(view->offset, offset, &absolute)) {
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OVERFLOW, operation));
    }
    if (absolute > view->allocation->accessible_bytes ||
        bytes > view->allocation->accessible_bytes - absolute) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OUT_OF_RANGE, operation));
    }
    if (out_global != NULL &&
        !xh2rt_u64_add(view->allocation->bo.start, absolute,
                       out_global)) {
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OVERFLOW, operation));
    }
    if (out_iomap != NULL &&
        !xh2rt_u64_add(view->allocation->iomap, absolute, out_iomap)) {
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OVERFLOW, operation));
    }
    if (out_view != NULL)
        *out_view = view;
    if (out_absolute != NULL)
        *out_absolute = absolute;
    return result;
}

static int xh2rt_close_handle(int (*close_function)(fd_handle_t),
                               fd_handle_t handle, int *saved_errno)
{
    int rc;

    errno = 0;
    rc = close_function(handle);
    *saved_errno = errno;
    return rc;
}

static void xh2rt_preserve_first(xh2rt_result *first, int *have_first,
                                 xh2rt_result candidate)
{
    if (!*have_first) {
        *first = candidate;
        *have_first = 1;
    }
}

static int xh2rt_any_live_bo(const xh2rt_context *context)
{
    const xh2rt_allocation *allocation;

    for (allocation = context->allocations; allocation != NULL;
         allocation = allocation->next) {
        if (allocation->bo_live)
            return 1;
    }
    return 0;
}

static void xh2rt_destroy_context(xh2rt_context *context)
{
    xh2rt_buffer_view *view = context->views;
    xh2rt_allocation *allocation = context->allocations;

    while (view != NULL) {
        xh2rt_buffer_view *next = view->next;
        free(view);
        view = next;
    }
    while (allocation != NULL) {
        xh2rt_allocation *next = allocation->next;
        free(allocation);
        allocation = next;
    }
    if (context->group != NULL) {
        free(context->group->pins);
        free(context->group);
    }
    free(context);
}

static void xh2rt_cleanup_handles(xh2rt_context *context,
                                  xh2rt_result *first, int *have_first)
{
    int rc;
    int saved_errno;

    if (context->transfer_live) {
        rc = xh2rt_close_handle(xh2a_memory_transfer_close,
                                  context->transfer, &saved_errno);
        if (rc == 0)
            context->transfer_live = 0;
        else
            xh2rt_preserve_first(
                first, have_first,
                xh2rt_hal_result("xh2a_memory_transfer_close", rc,
                                 saved_errno));
    }
    if (context->ddr_live && !xh2rt_any_live_bo(context)) {
        rc = xh2rt_close_handle(xh2a_memory_allocator_close, context->ddr,
                                  &saved_errno);
        if (rc == 0)
            context->ddr_live = 0;
        else
            xh2rt_preserve_first(
                first, have_first,
                xh2rt_hal_result("xh2a_memory_allocator_close", rc,
                                 saved_errno));
    }
    if (context->ipu_live &&
        (context->group == NULL ||
         (!context->group->unknown_quiescence &&
          !context->group->destroy_pending))) {
        rc = xh2rt_close_handle(xh2a_ipu_close, context->ipu,
                                  &saved_errno);
        if (rc == 0)
            context->ipu_live = 0;
        else
            xh2rt_preserve_first(
                first, have_first,
                xh2rt_hal_result("xh2a_ipu_close", rc,
                                 saved_errno));
    }
}

int xh2rt_result_is_ok(xh2rt_result result)
{
    return result.status == XH2RT_STATUS_OK;
}

const char *xh2rt_status_string(enum xh2rt_status status)
{
    switch (status) {
    case XH2RT_STATUS_OK:
        return "ok";
    case XH2RT_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case XH2RT_STATUS_INVALID_STATE:
        return "invalid state";
    case XH2RT_STATUS_NO_MEMORY:
        return "out of memory";
    case XH2RT_STATUS_OVERFLOW:
        return "integer overflow";
    case XH2RT_STATUS_OUT_OF_RANGE:
        return "out of range";
    case XH2RT_STATUS_MISALIGNED:
        return "misaligned";
    case XH2RT_STATUS_STALE_BUFFER:
        return "stale or unknown buffer";
    case XH2RT_STATUS_BUSY:
        return "busy";
    case XH2RT_STATUS_HAL_FAILURE:
        return "HAL failure";
    case XH2RT_STATUS_HAL_PROTOCOL:
        return "invalid HAL response";
    case XH2RT_STATUS_POISONED:
        return "context poisoned";
    case XH2RT_STATUS_EXECUTION_FAILURE:
        return "device execution failure";
    }
    return "unknown status";
}

xh2rt_result xh2rt_context_open(uint32_t logical_device,
                                xh2rt_context **out_context)
{
    xh2rt_context *context;
    xh2rt_result result;
    size_t index;
    int rc;
    int saved_errno;

    if (out_context == NULL)
        return xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                                 "context_open");
    *out_context = NULL;
    if (logical_device != 0)
        return xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                                 "context_open.logical_device");

    context = (xh2rt_context *)calloc(1, sizeof(*context));
    if (context == NULL)
        return xh2rt_make_result(XH2RT_STATUS_NO_MEMORY, "context_open");
    context->ipu = INVALID_FD_HANDLE_VAL;
    context->ddr = INVALID_FD_HANDLE_VAL;
    context->transfer = INVALID_FD_HANDLE_VAL;
    context->last_error = xh2rt_make_result(XH2RT_STATUS_OK, "none");

    errno = 0;
    rc = xh2a_ipu_open(logical_device, &context->ipu);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_open", rc, saved_errno);
        context->last_error = result;
        free(context);
        return result;
    }
    if (context->ipu == INVALID_FD_HANDLE_VAL) {
        result = xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                   "xh2a_ipu_open");
        context->last_error = result;
        free(context);
        return result;
    }
    context->ipu_live = 1;

    errno = 0;
    rc = xh2a_memory_allocator_open(
        (int)logical_device, XH2A_MEMORY_ALLOCATOR_MEMPOOL_DDR,
        &context->ddr);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_allocator_open", rc, saved_errno);
        goto fail;
    }
    if (context->ddr == INVALID_FD_HANDLE_VAL) {
        result = xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                   "xh2a_memory_allocator_open");
        goto fail;
    }
    context->ddr_live = 1;

    errno = 0;
    rc = xh2a_memory_transfer_open((int)logical_device, &context->transfer);
    saved_errno = errno;
    if (rc != 0) {
        result =
            xh2rt_hal_result("xh2a_memory_transfer_open", rc, saved_errno);
        goto fail;
    }
    if (context->transfer == INVALID_FD_HANDLE_VAL) {
        result = xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                   "xh2a_memory_transfer_open");
        goto fail;
    }
    context->transfer_live = 1;

    /* Upload once before any batch can start. Code uses the same BO pinning
     * and failure quarantine as data, and stays owned by this context. */
    for (index = 0; index < XH2RT_KERNEL_COUNT; index++) {
        result = xh2rt_buffer_alloc(context, xh2rt_kernel_code[index].bytes,
                                     &context->kernels[index]);
        if (!xh2rt_result_is_ok(result))
            goto fail;
        result = xh2rt_buffer_write(context, context->kernels[index], 0,
                                     xh2rt_kernel_code[index].data,
                                     xh2rt_kernel_code[index].bytes);
        if (!xh2rt_result_is_ok(result))
            goto fail;
    }
    *out_context = context;
    return xh2rt_make_result(XH2RT_STATUS_OK, "context_open");

fail:
    (void)xh2rt_context_close(&context);
    if (context != NULL)
        context->last_error = result;
    *out_context = context;
    return result;
}

xh2rt_result xh2rt_context_close(xh2rt_context **context_pointer)
{
    xh2rt_context *context;
    xh2rt_allocation *allocation;
    xh2rt_buffer_view *view;
    xh2rt_result first = xh2rt_make_result(XH2RT_STATUS_OK,
                                           "context_close");
    int have_first = 0;

    if (context_pointer == NULL)
        return xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                                 "context_close");
    context = *context_pointer;
    if (context == NULL)
        return first;

    if (context->group != NULL &&
        !context->group->unknown_quiescence &&
        !context->group->destroy_pending) {
        xh2rt_result result =
            xh2rt_drain_group(context, "context_close.drain");

        if (!xh2rt_result_is_ok(result))
            xh2rt_preserve_first(&first, &have_first, result);
    }
    if (context->group != NULL && context->group->destroy_pending) {
        int rc;
        int saved_errno;

        errno = 0;
        rc = xh2a_ipu_destroy_group(context->ipu, context->group->id);
        saved_errno = errno;
        if (rc == 0) {
            xh2rt_group_release_host(context, context->group);
        } else {
            xh2rt_result result =
                xh2rt_hal_result("xh2a_ipu_destroy_group", rc,
                                 saved_errno);
            xh2rt_stamp_group(&result, context->group);
            xh2rt_preserve_first(&first, &have_first, result);
        }
    }
    context->batch_active = 0;
    context->state = XH2RT_CONTEXT_CLOSING;
    for (view = context->views; view != NULL; view = view->next) {
        view->active = 0;
        view->references = 0;
    }
    for (allocation = context->allocations; allocation != NULL;
         allocation = allocation->next) {
        int rc;
        int saved_errno;

        allocation->logical_references = 0;
        if (!allocation->bo_live)
            continue;
        if (allocation->inflight != 0 || allocation->quarantined) {
            xh2rt_preserve_first(
                &first, &have_first,
                xh2rt_make_result(XH2RT_STATUS_BUSY,
                                  allocation->quarantined
                                      ? "context_close.quarantine"
                                      : "context_close.inflight"));
            continue;
        }
        errno = 0;
        rc = xh2a_memory_allocator_free_buffer_object(&allocation->bo);
        saved_errno = errno;
        (void)saved_errno; /* Primary metadata failure remains authoritative. */
        if (rc == 0)
            allocation->bo_live = 0;
        else
            xh2rt_preserve_first(
                &first, &have_first,
                xh2rt_hal_result("xh2a_memory_allocator_free_buffer_object",
                                  rc, saved_errno));
    }

    xh2rt_cleanup_handles(context, &first, &have_first);
    if (context->group == NULL && !xh2rt_any_live_bo(context) &&
        !context->transfer_live && !context->ddr_live &&
        !context->ipu_live) {
        xh2rt_destroy_context(context);
        *context_pointer = NULL;
        return first;
    }
    if (!have_first)
        first = xh2rt_make_result(XH2RT_STATUS_BUSY, "context_close");
    if (!context->poisoned)
        context->last_error = first;
    return first;
}

xh2rt_result xh2rt_context_last_error(const xh2rt_context *context)
{
    if (context == NULL)
        return xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                                 "context_last_error");
    return context->last_error;
}

static int xh2rt_allocations_overlap(const xh2rt_context *context,
                                     uint64_t start, uint64_t bytes)
{
    const xh2rt_allocation *allocation;
    uint64_t end = start + bytes;

    for (allocation = context->allocations; allocation != NULL;
         allocation = allocation->next) {
        uint64_t other_end;

        if (!allocation->bo_live)
            continue;
        other_end = allocation->bo.start + allocation->bo.size;
        if (start < other_end && allocation->bo.start < end)
            return 1;
    }
    return 0;
}

static xh2rt_result xh2rt_validate_bo(
    xh2rt_context *context,
    const struct xh2a_memory_allocator_buffer_object *bo,
    uint64_t requested, uint64_t *iomap)
{
    uint64_t global_end;
    uint64_t iomap_end;

    if (bo->memory_allocator_fd != context->ddr || bo->size < requested ||
        bo->size == 0 || bo->start < XH2RT_IOMAP_DELTA) {
        return xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                 "xh2a_memory_allocator_alloc_buffer_object.metadata");
    }
    *iomap = bo->start - XH2RT_IOMAP_DELTA;
    if (!xh2rt_u64_add(bo->start, bo->size, &global_end) ||
        !xh2rt_u64_add(*iomap, bo->size, &iomap_end) ||
        *iomap > (uint64_t)UINTPTR_MAX ||
        bo->size > (uint64_t)UINTPTR_MAX - *iomap) {
        return xh2rt_make_result(XH2RT_STATUS_OVERFLOW,
                                 "xh2a_memory_allocator_alloc_buffer_object.metadata");
    }
    (void)global_end;
    (void)iomap_end;
    if (xh2rt_allocations_overlap(context, bo->start, bo->size))
        return xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                 "xh2a_memory_allocator_alloc_buffer_object.overlap");
    return xh2rt_make_result(XH2RT_STATUS_OK, "buffer_alloc");
}

static xh2rt_result xh2rt_free_allocation(xh2rt_context *context,
                                          xh2rt_allocation *allocation,
                                          const char *operation)
{
    int rc;
    int saved_errno;

    if (allocation == NULL || !allocation->bo_live)
        return xh2rt_make_result(XH2RT_STATUS_OK, operation);
    if (allocation->inflight != 0 || allocation->quarantined)
        return xh2rt_make_result(XH2RT_STATUS_BUSY, operation);
    errno = 0;
    rc = xh2a_memory_allocator_free_buffer_object(&allocation->bo);
    saved_errno = errno;
    if (rc != 0)
        return xh2rt_record(
            context,
            xh2rt_hal_result("xh2a_memory_allocator_free_buffer_object",
                              rc, saved_errno));
    xh2rt_forget_allocation(context, allocation);
    return xh2rt_make_result(XH2RT_STATUS_OK, operation);
}

static void xh2rt_stamp_group(xh2rt_result *result,
                              const xh2rt_group *group)
{
    result->group_id = group->id;
    result->group_kernel_count = group->kernel_count;
    result->group_core_count = XH2RT_CORE_COUNT;
}

static xh2rt_result xh2rt_capture_group_info(xh2rt_context *context,
                                             xh2rt_group *group,
                                             xh2rt_result *details)
{
    struct xh2a_group_info info = {.group_id = group->id};
    xh2rt_result result;
    int rc;
    int saved_errno;

    xh2rt_stamp_group(details, group);
    errno = 0;
    rc = xh2a_ipu_get_group_info(context->ipu, &info);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_get_group_info", rc,
                                  saved_errno);
        xh2rt_stamp_group(&result, group);
        return result;
    }
    details->group_id = info.group_id;
    details->group_state = info.state;
    details->group_flags = info.flags;
    details->group_kernel_count = info.kernel_num;
    details->group_core_count = info.core_num;
    details->group_core_mask = info.core_mask;
    if (info.group_id != group->id) {
        result = xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                   "xh2a_ipu_get_group_info");
        xh2rt_stamp_group(&result, group);
        result.group_id = info.group_id;
        result.group_state = info.state;
        result.group_flags = info.flags;
        result.group_kernel_count = info.kernel_num;
        result.group_core_count = info.core_num;
        result.group_core_mask = info.core_mask;
        return result;
    }
    return xh2rt_make_result(XH2RT_STATUS_OK,
                             "xh2a_ipu_get_group_info");
}

static void xh2rt_group_release_host(xh2rt_context *context,
                                     xh2rt_group *group)
{
    if (context->group == group)
        context->group = NULL;
    free(group->pins);
    free(group);
}

static void xh2rt_group_unpin(xh2rt_context *context,
                              xh2rt_group *group,
                              xh2rt_result *first, int *have_first)
{
    size_t index;

    for (index = 0; index < group->pin_count; index++) {
        xh2rt_allocation *allocation = group->pins[index];
        xh2rt_result result;

        allocation->inflight--;
        if (allocation->logical_references == 0 && allocation->bo_live &&
            !allocation->quarantined && allocation->inflight == 0) {
            result = xh2rt_free_allocation(context, allocation,
                                           "group_unpin");
            if (!xh2rt_result_is_ok(result))
                xh2rt_preserve_first(first, have_first, result);
        }
    }
    group->pin_count = 0;
}

static xh2rt_result xh2rt_ensure_group(xh2rt_context *context,
                                       size_t pin_slots)
{
    xh2rt_group *group;
    xh2rt_result result;
    int rc;
    int saved_errno;

    if (context->group != NULL) {
        if (!xh2rt_group_reserve_pins(context->group, pin_slots)) {
            return xh2rt_record(
                context,
                xh2rt_make_result(XH2RT_STATUS_NO_MEMORY,
                                  "launch.group_resources"));
        }
        return xh2rt_make_result(XH2RT_STATUS_OK, "group_reuse");
    }
    group = (xh2rt_group *)calloc(1, sizeof(*group));
    if (group == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_NO_MEMORY, "group_create"));
    group->id = INVALID_GROUP_ID;
    /* Reserve fallible host bookkeeping before creating a HAL group.  Once
     * the group exists, every error path must either submit or destroy it. */
    if (!xh2rt_group_reserve_pins(group, pin_slots)) {
        free(group);
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_NO_MEMORY,
                              "launch.group_resources"));
    }
    errno = 0;
    rc = xh2a_ipu_create_group(context->ipu, &group->id);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_create_group", rc,
                                  saved_errno);
        free(group->pins);
        free(group);
        return xh2rt_record(context, result);
    }
    if (group->id == INVALID_GROUP_ID || group->id < 0) {
        result = xh2rt_make_result(XH2RT_STATUS_HAL_PROTOCOL,
                                   "xh2a_ipu_create_group");
        free(group->pins);
        free(group);
        return xh2rt_record(context, result);
    }
    context->group = group;
    return xh2rt_make_result(XH2RT_STATUS_OK, "group_create");
}

static int xh2rt_group_reserve_pins(xh2rt_group *group, size_t add)
{
    xh2rt_allocation **pins;
    size_t required;
    size_t capacity;

    if (add > SIZE_MAX - group->pin_count)
        return 0;
    required = group->pin_count + add;
    if (required <= group->pin_capacity)
        return 1;
    capacity = group->pin_capacity == 0 ? 8 : group->pin_capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2)
            return 0;
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*pins))
        return 0;
    pins = (xh2rt_allocation **)realloc(
        group->pins, capacity * sizeof(*pins));
    if (pins == NULL)
        return 0;
    group->pins = pins;
    group->pin_capacity = capacity;
    return 1;
}

static void xh2rt_group_pin_one(xh2rt_group *group,
                                xh2rt_allocation *allocation)
{
    size_t index;

    for (index = 0; index < group->pin_count; index++) {
        if (group->pins[index] == allocation)
            return;
    }
    group->pins[group->pin_count++] = allocation;
    allocation->inflight++;
}

static xh2rt_result xh2rt_drain_group(xh2rt_context *context,
                                      const char *operation)
{
    xh2rt_group *group = context->group;
    xh2rt_result primary = xh2rt_make_result(XH2RT_STATUS_OK, operation);
    xh2rt_result info_result;
    xh2rt_result cleanup = xh2rt_make_result(XH2RT_STATUS_OK,
                                             "group_cleanup");
    int have_cleanup = 0;
    uint32_t execution_result = 0;
    int rc;
    int saved_errno;
    int destroy_rc;
    int destroy_errno;

    if (group == NULL)
        return primary;
    if (group->unknown_quiescence)
        return xh2rt_make_result(XH2RT_STATUS_POISONED, operation);
    if (group->destroy_pending)
        return xh2rt_make_result(XH2RT_STATUS_POISONED, operation);

    errno = 0;
    rc = xh2a_ipu_sync_group(context->ipu, group->id,
                              XH2RT_GROUP_SYNC_TIMEOUT_US,
                              &execution_result);
    saved_errno = errno;
    if (rc != 0) {
        primary = xh2rt_hal_result("xh2a_ipu_sync_group", rc,
                                   saved_errno);
        primary.execution_result = execution_result;
        xh2rt_stamp_group(&primary, group);
        info_result = xh2rt_capture_group_info(context, group, &primary);
        (void)info_result;
        group->unknown_quiescence = 1;
        return xh2rt_poison(context, primary);
    }

    primary.execution_result = execution_result;
    xh2rt_stamp_group(&primary, group);
    info_result = xh2rt_capture_group_info(context, group, &primary);
    if (execution_result != 0) {
        primary.status = XH2RT_STATUS_EXECUTION_FAILURE;
        primary.operation = "xh2a_ipu_sync_group.result";
        primary.raw_rc = 0;
        primary.saved_errno = saved_errno;
    } else if (!xh2rt_result_is_ok(info_result)) {
        primary = info_result;
    }

    errno = 0;
    destroy_rc = xh2a_ipu_destroy_group(context->ipu, group->id);
    destroy_errno = errno;
    if (destroy_rc != 0 && xh2rt_result_is_ok(primary)) {
        /* Preserve the group_info captured after successful completion. */
        primary.status = XH2RT_STATUS_HAL_FAILURE;
        primary.operation = "xh2a_ipu_destroy_group";
        primary.raw_rc = destroy_rc;
        primary.saved_errno = destroy_errno;
    }
    xh2rt_group_unpin(context, group, &cleanup, &have_cleanup);
    if (destroy_rc == 0) {
        xh2rt_group_release_host(context, group);
    } else {
        group->destroy_pending = 1;
    }
    if (xh2rt_result_is_ok(primary) && have_cleanup) {
        /* Keep the completed group's diagnostics while reporting the first
         * host cleanup failure that followed that completion. */
        primary.status = cleanup.status;
        primary.operation = cleanup.operation;
        primary.raw_rc = cleanup.raw_rc;
        primary.saved_errno = cleanup.saved_errno;
    }
    if (execution_result != 0 || destroy_rc != 0)
        return xh2rt_poison(context, primary);
    if (!xh2rt_result_is_ok(primary))
        return xh2rt_record(context, primary);
    return xh2rt_make_result(XH2RT_STATUS_OK, operation);
}

/*
 * The SDK startup code consumes a 16-byte header followed by uint64_t
 * arguments. Keep device entry arguments uniformly 64-bit; float values travel
 * as integer bits. This is the verified SDK subset, not a general C ABI packer.
 *
 * The typed wrappers put all buffer arguments first and the element count
 * last. Check every extent and address before creating or draining a group,
 * then pin the backing BOs (including code).
 */
static xh2rt_result xh2rt_launch_native(
    xh2rt_context *context, enum xh2rt_kernel kernel,
    xh2rt_buffer_view *const *buffers, size_t buffer_count,
    uint64_t *arguments, size_t argument_count)
{
    xh2rt_buffer_view *views[4];
    uint8_t envelope[16 + 4 * 8] = {0};
    uint32_t envelope_bytes = (uint32_t)(16 + argument_count * 8);
    uint64_t count = arguments[argument_count - 1];
    uint64_t bytes;
    size_t index;
    xh2rt_group *group;
    struct kernel_launch_data launch;
    xh2rt_result result = xh2rt_require_healthy(context, "launch");
    int one_shot;
    int rc;
    int saved_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    if (count > UINT64_MAX / sizeof(uint32_t))
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OVERFLOW,
                                        "launch.count"));
    if (kernel == XH2RT_ADD_F32 && count > UINT32_MAX)
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OUT_OF_RANGE,
                                        "launch.count"));
    bytes = count * sizeof(uint32_t);
    result = xh2rt_require_view(context, context->kernels[kernel],
                                 "launch.kernel", &views[0]);
    if (!xh2rt_result_is_ok(result))
        return result;
    for (index = 0; index < buffer_count; index++) {
        result = xh2rt_checked_range(
            context, buffers[index], 0, bytes, "launch.buffer",
            &views[index + 1], NULL, NULL, &arguments[index]);
        if (!xh2rt_result_is_ok(result))
            return result;
        if (count != 0 && (arguments[index] & (sizeof(uint32_t) - 1)) != 0)
            return xh2rt_record(
                context, xh2rt_make_result(XH2RT_STATUS_MISALIGNED,
                                            "launch.buffer"));
    }
    if (count == 0)
        return xh2rt_make_result(XH2RT_STATUS_OK, "launch");
    xh2rt_put_le64(envelope + 8, argument_count);
    for (index = 0; index < argument_count; index++)
        xh2rt_put_le64(envelope + 16 + index * 8, arguments[index]);

    one_shot = !context->batch_active;
    group = context->group;
    if (group != NULL &&
        (group->kernel_count == XH2RT_GROUP_MAX_KERNELS ||
         group->spm_bytes > XH2RT_GROUP_SPM_LIMIT - envelope_bytes)) {
        result = xh2rt_drain_group(context, "launch.split");
        if (!xh2rt_result_is_ok(result))
            return result;
    }
    result = xh2rt_ensure_group(context, buffer_count + 1);
    if (!xh2rt_result_is_ok(result))
        return result;
    group = context->group;
    for (index = 0; index <= buffer_count; index++)
        xh2rt_group_pin_one(group, views[index]->allocation);

    memset(&launch, 0, sizeof(launch));
    launch.kernel_addr = views[0]->allocation->iomap + views[0]->offset;
    launch.kernel_size = (uint32_t)views[0]->bytes;
    launch.param_phy_addr = (uint64_t)(uintptr_t)envelope;
    launch.param_type = XH2A_GROUP_PARAM_SPM;
    launch.param_size = envelope_bytes;
    /* These defaults match the 2-core x 4-tile partition in the kernels. */
    launch.core_num = XH2RT_CORE_COUNT;
    launch.tile_num = XH2RT_TILE_COUNT;
    launch.ilm_mode = UINT32_C(1);
    launch.timeout_us = UINT32_C(1000000);
    errno = 0;
    /* SPM launch copies these host bytes during the call, not during sync. */
    rc = xh2a_ipu_launch_kernel(context->ipu, group->id, &launch);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_launch_kernel", rc,
                                   saved_errno);
        xh2rt_stamp_group(&result, group);
        (void)xh2rt_capture_group_info(context, group, &result);
        group->unknown_quiescence = 1;
        return xh2rt_poison(context, result);
    }
    group->spm_bytes += envelope_bytes;
    group->kernel_count++;
    if (one_shot)
        return xh2rt_drain_group(context, "launch.one_shot");
    return xh2rt_make_result(XH2RT_STATUS_OK, "launch");
}

xh2rt_result xh2rt_fill_f32(xh2rt_context *context,
                             xh2rt_buffer_view *destination,
                             uint32_t value_bits, uint64_t count)
{
    xh2rt_buffer_view *buffers[] = {destination};
    uint64_t arguments[] = {0, value_bits, count};

    return xh2rt_launch_native(context, XH2RT_FILL_F32, buffers, 1,
                                arguments, 3);
}

xh2rt_result xh2rt_add_f32(xh2rt_context *context,
                            xh2rt_buffer_view *lhs,
                            xh2rt_buffer_view *rhs,
                            xh2rt_buffer_view *destination,
                            uint64_t count)
{
    xh2rt_buffer_view *buffers[] = {lhs, rhs, destination};
    uint64_t arguments[] = {0, 0, 0, count};

    return xh2rt_launch_native(context, XH2RT_ADD_F32, buffers, 3,
                                arguments, 4);
}

xh2rt_result xh2rt_commands_begin(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "commands_begin");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (context->batch_active)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_STATE,
                              "commands_begin"));
    context->batch_active = 1;
    return result;
}

int xh2rt_commands_active(const xh2rt_context *context)
{
    return context != NULL && context->state == XH2RT_CONTEXT_OPEN &&
           context->batch_active;
}

xh2rt_result xh2rt_commands_flush(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "commands_flush");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (!context->batch_active)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_STATE,
                              "commands_flush"));
    return xh2rt_drain_group(context, "commands_flush");
}

xh2rt_result xh2rt_commands_end(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "commands_end");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (!context->batch_active)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_STATE,
                              "commands_end"));
    result = xh2rt_drain_group(context, "commands_end");
    context->batch_active = 0;
    return result;
}

xh2rt_result xh2rt_synchronize(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "synchronize");

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_drain_group(context, "synchronize");
    context->batch_active = 0;
    return result;
}

xh2rt_result xh2rt_buffer_alloc(xh2rt_context *context, uint64_t bytes,
                                xh2rt_buffer_view **out_buffer)
{
    xh2rt_allocation *allocation;
    xh2rt_buffer_view *view;
    xh2rt_result result = xh2rt_require_healthy(context, "buffer_alloc");
    int rc;
    int saved_errno;

    if (out_buffer == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_alloc.out_buffer"));
    *out_buffer = NULL;
    if (!xh2rt_result_is_ok(result))
        return result;
    if (bytes == 0)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_alloc.bytes"));

    allocation = (xh2rt_allocation *)calloc(1, sizeof(*allocation));
    view = (xh2rt_buffer_view *)calloc(1, sizeof(*view));
    if (allocation == NULL || view == NULL || !xh2rt_assign_handle(view)) {
        free(allocation);
        free(view);
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_NO_MEMORY, "buffer_alloc"));
    }
    allocation->bo.memory_allocator_fd = INVALID_FD_HANDLE_VAL;
    errno = 0;
    rc = xh2a_memory_allocator_alloc_buffer_object(context->ddr, bytes,
                                                  &allocation->bo);
    saved_errno = errno;
    if (rc != 0) {
        free(allocation);
        free(view);
        return xh2rt_record(
            context,
            xh2rt_hal_result("xh2a_memory_allocator_alloc_buffer_object",
                              rc, saved_errno));
    }
    allocation->bo_live = 1;
    allocation->accessible_bytes = bytes;
    result = xh2rt_validate_bo(context, &allocation->bo, bytes,
                               &allocation->iomap);
    if (!xh2rt_result_is_ok(result)) {
        errno = 0;
        rc = xh2a_memory_allocator_free_buffer_object(&allocation->bo);
        saved_errno = errno;
        (void)saved_errno; /* Snapshot before retaining the primary error. */
        if (rc == 0) {
            free(allocation);
        } else {
            allocation->next = context->allocations;
            context->allocations = allocation;
            context->state = XH2RT_CONTEXT_CLOSING;
        }
        free(view);
        return xh2rt_record(context, result);
    }

    allocation->logical_references = 1;
    allocation->next = context->allocations;
    context->allocations = allocation;
    view->allocation = allocation;
    view->bytes = bytes;
    view->references = 1;
    view->active = 1;
    view->next = context->views;
    context->views = view;
    *out_buffer = view->handle;
    return xh2rt_make_result(XH2RT_STATUS_OK, "buffer_alloc");
}

xh2rt_result xh2rt_buffer_retain(xh2rt_context *context,
                                 xh2rt_buffer_view *buffer)
{
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result =
        xh2rt_require_view(context, buffer, "buffer_retain", &view);

    if (!xh2rt_result_is_ok(result))
        return result;
    if (view->references == UINT64_MAX ||
        view->allocation->logical_references == UINT64_MAX) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW, "buffer_retain"));
    }
    view->references++;
    view->allocation->logical_references++;
    return result;
}

xh2rt_result xh2rt_buffer_release(xh2rt_context *context,
                                  xh2rt_buffer_view **buffer_pointer)
{
    xh2rt_allocation *allocation;
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result;

    if (buffer_pointer == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_release"));
    if (*buffer_pointer == NULL)
        return xh2rt_make_result(XH2RT_STATUS_OK, "buffer_release");
    result = xh2rt_require_open(context, "buffer_release");
    if (!xh2rt_result_is_ok(result))
        return result;
    view = xh2rt_find_view(context, *buffer_pointer);
    if (view == NULL || !view->release_pending) {
        result = xh2rt_require_view(context, *buffer_pointer,
                                    "buffer_release", &view);
        if (!xh2rt_result_is_ok(result))
            return result;
        view->references--;
        view->allocation->logical_references--;
        if (view->references == 0)
            view->active = 0;
    }
    allocation = view->allocation;
    if (allocation->logical_references == 0) {
        if (allocation->inflight != 0 || allocation->quarantined) {
            result = xh2rt_make_result(XH2RT_STATUS_OK,
                                        "buffer_release.deferred");
        } else {
            result = xh2rt_free_allocation(context, allocation,
                                            "buffer_release");
            if (!xh2rt_result_is_ok(result)) {
                view->release_pending = 1;
                return result;
            }
        }
    }
    if (view->references == 0)
        xh2rt_forget_view(context, view);
    *buffer_pointer = NULL;
    return result;
}

xh2rt_result xh2rt_buffer_view_create(xh2rt_context *context,
                                      xh2rt_buffer_view *buffer,
                                      uint64_t offset, uint64_t bytes,
                                      xh2rt_buffer_view **out_view)
{
    xh2rt_buffer_view *parent = NULL;
    xh2rt_buffer_view *view = NULL;
    uint64_t absolute;
    xh2rt_result result;

    if (out_view == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_view_create.out_view"));
    *out_view = NULL;
    result = xh2rt_require_view(context, buffer, "buffer_view_create",
                                &parent);
    if (!xh2rt_result_is_ok(result))
        return result;
    if (!xh2rt_range_fits(parent->bytes, offset, bytes)) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OUT_OF_RANGE,
                              "buffer_view_create"));
    }
    if (!xh2rt_u64_add(parent->offset, offset, &absolute)) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW,
                              "buffer_view_create"));
    }
    if (parent->allocation->logical_references == UINT64_MAX) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW,
                              "buffer_view_create"));
    }
    view = (xh2rt_buffer_view *)calloc(1, sizeof(*view));
    if (view == NULL || !xh2rt_assign_handle(view)) {
        free(view);
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_NO_MEMORY,
                              "buffer_view_create"));
    }

    view->allocation = parent->allocation;
    view->offset = absolute;
    view->bytes = bytes;
    view->references = 1;
    view->active = 1;
    view->next = context->views;
    context->views = view;
    view->allocation->logical_references++;
    *out_view = view->handle;
    return result;
}

xh2rt_result xh2rt_buffer_bytes(xh2rt_context *context,
                                const xh2rt_buffer_view *buffer,
                                uint64_t *out_bytes)
{
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result;

    if (out_bytes == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_bytes"));
    *out_bytes = 0;
    result = xh2rt_require_view(context, buffer, "buffer_bytes", &view);
    if (!xh2rt_result_is_ok(result))
        return result;
    *out_bytes = view->bytes;
    return result;
}

static xh2rt_result xh2rt_check_host_range(xh2rt_context *context,
                                           const void *pointer,
                                           uint64_t bytes,
                                           const char *operation)
{
    uintptr_t address;

    if (bytes == 0)
        return xh2rt_make_result(XH2RT_STATUS_OK, operation);
    if (pointer == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT, operation));
    if (bytes > (uint64_t)SIZE_MAX)
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OVERFLOW, operation));
    address = (uintptr_t)pointer;
    if (bytes > (uint64_t)UINTPTR_MAX - (uint64_t)address)
        return xh2rt_record(
            context, xh2rt_make_result(XH2RT_STATUS_OVERFLOW, operation));
    return xh2rt_make_result(XH2RT_STATUS_OK, operation);
}

xh2rt_result xh2rt_buffer_write(xh2rt_context *context,
                                xh2rt_buffer_view *destination,
                                uint64_t destination_offset,
                                const void *source, uint64_t bytes)
{
    xh2rt_buffer_view *destination_view = NULL;
    uint64_t global;
    xh2rt_result result = xh2rt_require_healthy(context, "buffer_write");
    int rc;
    int saved_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_checked_range(
        context, destination, destination_offset, bytes, "buffer_write",
        &destination_view, NULL, &global, NULL);

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_check_host_range(context, source, bytes, "buffer_write");
    if (!xh2rt_result_is_ok(result) || bytes == 0)
        return result;
    result = xh2rt_drain_group(context, "buffer_write.drain");
    if (!xh2rt_result_is_ok(result))
        return result;
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(context->transfer, (uint64_t)(uintptr_t)source,
                        global, (size_t)bytes,
                        XH2A_MEMORY_TRANSFER_TYPE_HOST_TO_DEVICE);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_transfer_copy_buffer(H2D)", rc,
                                  saved_errno);
        destination_view->allocation->quarantined = 1;
        return xh2rt_poison(context, result);
    }
    return xh2rt_make_result(XH2RT_STATUS_OK, "buffer_write");
}

xh2rt_result xh2rt_buffer_read(xh2rt_context *context, void *destination,
                               xh2rt_buffer_view *source,
                               uint64_t source_offset, uint64_t bytes)
{
    xh2rt_buffer_view *source_view = NULL;
    uint64_t global;
    xh2rt_result result = xh2rt_require_healthy(context, "buffer_read");
    int rc;
    int saved_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_checked_range(
        context, source, source_offset, bytes, "buffer_read", &source_view,
        NULL, &global, NULL);
    if (!xh2rt_result_is_ok(result))
        return result;
    result =
        xh2rt_check_host_range(context, destination, bytes, "buffer_read");
    if (!xh2rt_result_is_ok(result) || bytes == 0)
        return result;
    result = xh2rt_drain_group(context, "buffer_read.drain");
    if (!xh2rt_result_is_ok(result))
        return result;
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(context->transfer, global,
                        (uint64_t)(uintptr_t)destination, (size_t)bytes,
                        XH2A_MEMORY_TRANSFER_TYPE_DEVICE_TO_HOST);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_transfer_copy_buffer(D2H)", rc,
                                  saved_errno);
        source_view->allocation->quarantined = 1;
        return xh2rt_poison(context, result);
    }
    return xh2rt_make_result(XH2RT_STATUS_OK, "buffer_read");
}

static int xh2rt_ranges_overlap(uint64_t left, uint64_t right,
                                uint64_t bytes)
{
    if (left <= right)
        return right - left < bytes;
    return left - right < bytes;
}

xh2rt_result xh2rt_buffer_copy(xh2rt_context *context,
                               xh2rt_buffer_view *destination,
                               uint64_t destination_offset,
                               xh2rt_buffer_view *source,
                               uint64_t source_offset, uint64_t bytes)
{
    xh2rt_buffer_view *destination_view;
    xh2rt_buffer_view *source_view;
    uint64_t destination_absolute;
    uint64_t source_absolute;
    uint64_t destination_global;
    uint64_t source_global;
    xh2rt_result result;
    int rc;
    int saved_errno;

    result = xh2rt_require_healthy(context, "buffer_copy");
    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_checked_range(
        context, destination, destination_offset, bytes, "buffer_copy",
        &destination_view, &destination_absolute, &destination_global, NULL);
    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_checked_range(
        context, source, source_offset, bytes, "buffer_copy", &source_view,
        &source_absolute, &source_global, NULL);
    if (!xh2rt_result_is_ok(result))
        return result;
    if (bytes > (uint64_t)SIZE_MAX)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW, "buffer_copy"));
    if (bytes == 0)
        return result;

    if (destination_view->allocation == source_view->allocation) {
        if (destination_absolute == source_absolute)
            return result;
        if (xh2rt_ranges_overlap(destination_absolute, source_absolute,
                                 bytes)) {
            return xh2rt_record(
                context,
                xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                                  "buffer_copy.overlap"));
        }
    }

    result = xh2rt_drain_group(context, "buffer_copy.drain");
    if (!xh2rt_result_is_ok(result))
        return result;
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(context->transfer, source_global,
                        destination_global, (size_t)bytes,
                        XH2A_MEMORY_TRANSFER_TYPE_INNER_DEVICE);
    saved_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_transfer_copy_buffer(D2D)", rc,
                                  saved_errno);
        destination_view->allocation->quarantined = 1;
        source_view->allocation->quarantined = 1;
        return xh2rt_poison(context, result);
    }
    return xh2rt_make_result(XH2RT_STATUS_OK, "buffer_copy");
}

xh2rt_result xh2rt_buffer_iomap(xh2rt_context *context,
                                const xh2rt_buffer_view *buffer,
                                uint64_t offset, uint64_t bytes,
                                uint64_t alignment, uintptr_t *out_iomap)
{
    uint64_t iomap;
    xh2rt_result result;

    if (out_iomap == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_iomap"));
    *out_iomap = 0;
    if (!xh2rt_alignment_valid(alignment))
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_iomap.alignment"));
    result = xh2rt_checked_range(context, buffer, offset, bytes,
                                 "buffer_iomap", NULL, NULL, NULL, &iomap);
    if (!xh2rt_result_is_ok(result))
        return result;
    if ((iomap & (alignment - 1)) != 0)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_MISALIGNED, "buffer_iomap"));
    if (iomap > (uint64_t)UINTPTR_MAX ||
        bytes > (uint64_t)UINTPTR_MAX - iomap) {
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW, "buffer_iomap"));
    }
    *out_iomap = (uintptr_t)iomap;
    return result;
}

xh2rt_result xh2rt_buffer_validate_iomap(
    xh2rt_context *context, const xh2rt_buffer_view *buffer,
    uintptr_t iomap, uint64_t bytes, uint64_t alignment,
    uint64_t *out_offset)
{
    xh2rt_buffer_view *view = NULL;
    uint64_t view_iomap;
    uint64_t address = (uint64_t)iomap;
    uint64_t offset;
    xh2rt_result result;

    if (out_offset == NULL)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_validate_iomap"));
    *out_offset = 0;
    if (!xh2rt_alignment_valid(alignment))
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_INVALID_ARGUMENT,
                              "buffer_validate_iomap.alignment"));
    result = xh2rt_require_view(context, buffer, "buffer_validate_iomap",
                                &view);
    if (!xh2rt_result_is_ok(result))
        return result;
    if (!xh2rt_u64_add(view->allocation->iomap, view->offset, &view_iomap))
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW,
                              "buffer_validate_iomap"));
    if (address < view_iomap)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OUT_OF_RANGE,
                              "buffer_validate_iomap"));
    offset = address - view_iomap;
    if (!xh2rt_range_fits(view->bytes, offset, bytes))
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OUT_OF_RANGE,
                              "buffer_validate_iomap"));
    if ((address & (alignment - 1)) != 0)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_MISALIGNED,
                              "buffer_validate_iomap"));
    if (bytes > (uint64_t)UINTPTR_MAX - address)
        return xh2rt_record(
            context,
            xh2rt_make_result(XH2RT_STATUS_OVERFLOW,
                              "buffer_validate_iomap"));
    *out_offset = offset;
    return result;
}
