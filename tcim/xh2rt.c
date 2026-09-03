#include "xh2rt_kernel.h"

#include <ipu_api.h>
#include <memory_allocator.h>
#include <memory_transfer.h>

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define XH2RT_IOMAP_DELTA UINT64_C(0x1000000000)
#define XH2RT_GROUP_MAX_INVOCATIONS UINT32_C(255)
#define XH2RT_GROUP_SPM_LIMIT UINT64_C(1048576)
#define XH2RT_GROUP_SYNC_TIMEOUT_US UINT32_C(0)
#define XH2RT_KERNEL_ENVELOPE_BYTES(slot_count) \
    (16 + ((slot_count) + 1) * sizeof(uint64_t))
#define XH2RT_KERNEL_MAX_ENVELOPE_BYTES \
    XH2RT_KERNEL_ENVELOPE_BYTES(XH2RT_KERNEL_MAX_SLOTS)
#define XH2RT_COMPLETION_ARENA_BYTES \
    (XH2RT_GROUP_MAX_INVOCATIONS * XH2RT_KERNEL_COMPLETION_RECORD_BYTES)

typedef char xh2rt_group_envelopes_must_fit_spm[
    XH2RT_GROUP_MAX_INVOCATIONS * XH2RT_KERNEL_MAX_ENVELOPE_BYTES <=
        XH2RT_GROUP_SPM_LIMIT
        ? 1 : -1];

typedef struct xh2rt_kernel_image {
    const char *name;
    const uint8_t *code;
    size_t code_bytes;
} xh2rt_kernel_image;

#include "fill_f32.hex"
#include "add_f32.hex"

static const xh2rt_kernel_image xh2rt_kernels[] = {
    {"fill_f32", xh2rt_fill_f32_code, sizeof(xh2rt_fill_f32_code)},
    {"add_f32", xh2rt_add_f32_code, sizeof(xh2rt_add_f32_code)}
};

enum {
    XH2RT_KERNEL_COUNT = sizeof(xh2rt_kernels) / sizeof(xh2rt_kernels[0])
};

enum xh2rt_error_kind {
    XH2RT_ERROR_NONE = 0,
    XH2RT_ERROR_INVALID_ARGUMENT,
    XH2RT_ERROR_INVALID_STATE,
    XH2RT_ERROR_NO_MEMORY,
    XH2RT_ERROR_OVERFLOW,
    XH2RT_ERROR_OUT_OF_RANGE,
    XH2RT_ERROR_MISALIGNED,
    XH2RT_ERROR_STALE_BUFFER,
    XH2RT_ERROR_BUSY,
    XH2RT_ERROR_HAL_FAILURE,
    XH2RT_ERROR_HAL_PROTOCOL,
    XH2RT_ERROR_HAL_EXECUTION,
    XH2RT_ERROR_KERNEL_OUTCOME,
    XH2RT_ERROR_COMPLETION_MISSING
};

enum xh2rt_context_phase {
    XH2RT_CONTEXT_OPEN = 0,
    XH2RT_CONTEXT_CLOSING
};

enum xh2rt_group_phase {
    XH2RT_GROUP_ACTIVE = 0,
    XH2RT_GROUP_QUIESCENCE_UNKNOWN,
    XH2RT_GROUP_DESTROY_PENDING
};

/* Rich HAL and kernel diagnostics terminate at the public int/errno facade. */
typedef struct xh2rt_result {
    enum xh2rt_error_kind error;
    const char *operation;
    int hal_rc;
    int hal_errno;
    int64_t group_id;
    uint64_t group_state;
    uint64_t group_flags;
    uint32_t group_kernel_count;
    uint32_t group_core_count;
    uint32_t group_core_mask;
    uint32_t hal_sync_result;
    uint32_t kernel_outcome;
    uint32_t invocation_index;
    uint32_t hart_index;
} xh2rt_result;

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
    xh2rt_group *pinned_group;
    int bo_live;
    int quarantined;
    struct xh2rt_allocation *next;
};

struct xh2rt_group {
    group_id_t id;
    uint32_t invocation_count;
    xh2rt_allocation **pins;
    size_t pin_count;
    size_t pin_capacity;
    enum xh2rt_group_phase phase;
};

struct xh2rt_context {
    enum xh2rt_context_phase phase;
    fd_handle_t ipu;
    fd_handle_t ddr;
    fd_handle_t transfer;
    int ipu_live;
    int ddr_live;
    int transfer_live;
    xh2rt_allocation *allocations;
    xh2rt_buffer_view *views;
    /* Internal buffers keep resolved views; opaque handles are only for callers. */
    xh2rt_buffer_view *kernel_views[XH2RT_KERNEL_COUNT];
    xh2rt_buffer_view *completion_view;
    /* One bounded arena per context; records cannot overlap across invocations.
     * Keep transfer staging alive with quarantined device memory on failure. */
    uint8_t completion_host[XH2RT_COMPLETION_ARENA_BYTES];
    uint32_t completion_records_to_reset;
    /* group is either NULL or &group_storage. Pins retain their allocation
     * markers across unknown quiescence; capacity is reused across groups. */
    xh2rt_group group_storage;
    xh2rt_group *group;
    int batch_active;
    xh2rt_result poison_cause;
};

static xh2rt_result xh2rt_drain_group(xh2rt_context *context,
                                      const char *operation);
static int xh2rt_result_is_ok(xh2rt_result result);
static int xh2rt_publish(xh2rt_result result);
static void xh2rt_stamp_group(xh2rt_result *result,
                              const xh2rt_group *group);
static void xh2rt_group_release_host(xh2rt_context *context,
                                     xh2rt_group *group);
static int xh2rt_group_reserve_pins(xh2rt_group *group, size_t add);
static xh2rt_result xh2rt_completion_transfer(xh2rt_context *context,
                                              uint32_t records, int readback);
static xh2rt_result xh2rt_context_close_result(xh2rt_context **context);
static xh2rt_result xh2rt_buffer_alloc_result(xh2rt_context *context,
                                              uint64_t bytes,
                                              xh2rt_buffer_view **out_buffer);
static xh2rt_result xh2rt_buffer_write_result(
    xh2rt_context *context, xh2rt_buffer_view *destination,
    uint64_t destination_offset, const void *source, uint64_t bytes);

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

static xh2rt_result xh2rt_make_result(enum xh2rt_error_kind error,
                                      const char *operation)
{
    xh2rt_result result;

    memset(&result, 0, sizeof(result));
    result.error = error;
    result.operation = operation;
    result.group_id = -1;
    result.invocation_index = UINT32_MAX;
    result.hart_index = UINT32_MAX;
    return result;
}

static xh2rt_result xh2rt_hal_result(const char *operation, int hal_rc,
                                     int hal_errno)
{
    xh2rt_result result =
        xh2rt_make_result(XH2RT_ERROR_HAL_FAILURE, operation);

    result.hal_rc = hal_rc;
    result.hal_errno = hal_errno;
    return result;
}

/* Replace the failure cause while retaining the group's observed diagnostics. */
static void xh2rt_result_set_cause(xh2rt_result *details,
                                   xh2rt_result cause)
{
    details->error = cause.error;
    details->operation = cause.operation;
    details->hal_rc = cause.hal_rc;
    details->hal_errno = cause.hal_errno;
    details->kernel_outcome = cause.kernel_outcome;
    details->invocation_index = cause.invocation_index;
    details->hart_index = cause.hart_index;
}

static xh2rt_result xh2rt_validate_kernel_images(void)
{
    size_t index;

    for (index = 0; index < XH2RT_KERNEL_COUNT; index++) {
        const xh2rt_kernel_image *image = &xh2rt_kernels[index];
        size_t previous;

        if (image->name == NULL || image->name[0] == '\0')
            return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                     "kernel_image.name");
        if (image->code == NULL || image->code_bytes == 0 ||
            image->code_bytes > UINT16_MAX) {
            return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                     "kernel_image.code");
        }
        for (previous = 0; previous < index; previous++) {
            if (strcmp(xh2rt_kernels[previous].name, image->name) == 0)
                return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                         "kernel_image.duplicate");
        }
    }
    return xh2rt_make_result(XH2RT_ERROR_NONE, "kernel_image");
}

static xh2rt_result xh2rt_poison(xh2rt_context *context,
                                 xh2rt_result primary)
{
    if (context == NULL)
        return primary;
    if (!xh2rt_result_is_ok(context->poison_cause))
        return context->poison_cause;
    context->poison_cause = primary;
    context->batch_active = 0;
    return context->poison_cause;
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

static int xh2rt_view_is_live(const xh2rt_buffer_view *view)
{
    return view != NULL && view->active && view->references != 0 &&
           view->allocation != NULL && view->allocation->bo_live;
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
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT, operation);
    if (context->phase != XH2RT_CONTEXT_OPEN)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE, operation);
    return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
}

static xh2rt_result xh2rt_require_healthy(xh2rt_context *context,
                                          const char *operation)
{
    xh2rt_result result = xh2rt_require_open(context, operation);

    if (!xh2rt_result_is_ok(result))
        return result;
    if (!xh2rt_result_is_ok(context->poison_cause))
        return context->poison_cause;
    return result;
}

int xh2rt_kernel_reject(xh2rt_context *context,
                        enum xh2rt_kernel_rejection rejection,
                        const char *operation)
{
    xh2rt_result result = xh2rt_require_healthy(context, operation);
    enum xh2rt_error_kind error;

    if (!xh2rt_result_is_ok(result))
        return xh2rt_publish(result);
    switch (rejection) {
    case XH2RT_KERNEL_REJECT_OVERFLOW:
        error = XH2RT_ERROR_OVERFLOW;
        break;
    case XH2RT_KERNEL_REJECT_OUT_OF_RANGE:
        error = XH2RT_ERROR_OUT_OF_RANGE;
        break;
    default:
        error = XH2RT_ERROR_INVALID_ARGUMENT;
        break;
    }
    result = xh2rt_make_result(error, operation);
    return xh2rt_publish(result);
}

static size_t xh2rt_find_kernel_index(const char *name)
{
    size_t index;

    for (index = 0; index < XH2RT_KERNEL_COUNT; index++) {
        if (strcmp(xh2rt_kernels[index].name, name) == 0)
            return index;
    }
    return XH2RT_KERNEL_COUNT;
}

int xh2rt_context_is_healthy(const xh2rt_context *context)
{
    return context != NULL && context->phase == XH2RT_CONTEXT_OPEN &&
           xh2rt_result_is_ok(context->poison_cause);
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
    if (!xh2rt_view_is_live(view)) {
        return xh2rt_make_result(XH2RT_ERROR_STALE_BUFFER, operation);
    }
    *out_view = view;
    return result;
}

static xh2rt_result xh2rt_checked_view_range(
    xh2rt_buffer_view *view,
    uint64_t offset, uint64_t bytes, const char *operation,
    uint64_t *out_absolute, uint64_t *out_global, uint64_t *out_iomap)
{
    uint64_t absolute;
    xh2rt_result result = xh2rt_make_result(XH2RT_ERROR_NONE, operation);

    if (!xh2rt_view_is_live(view))
        return xh2rt_make_result(XH2RT_ERROR_STALE_BUFFER, operation);
    if (!xh2rt_range_fits(view->bytes, offset, bytes)) {
        return xh2rt_make_result(XH2RT_ERROR_OUT_OF_RANGE, operation);
    }
    if (!xh2rt_u64_add(view->offset, offset, &absolute)) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, operation);
    }
    if (absolute > view->allocation->accessible_bytes ||
        bytes > view->allocation->accessible_bytes - absolute) {
        return xh2rt_make_result(XH2RT_ERROR_OUT_OF_RANGE, operation);
    }
    if (out_global != NULL &&
        !xh2rt_u64_add(view->allocation->bo.start, absolute,
                       out_global)) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, operation);
    }
    if (out_iomap != NULL &&
        !xh2rt_u64_add(view->allocation->iomap, absolute, out_iomap)) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, operation);
    }
    if (out_absolute != NULL)
        *out_absolute = absolute;
    return result;
}

static xh2rt_result xh2rt_checked_range(
    xh2rt_context *context, const xh2rt_buffer_view *candidate,
    uint64_t offset, uint64_t bytes, const char *operation,
    xh2rt_buffer_view **out_view, uint64_t *out_absolute,
    uint64_t *out_global, uint64_t *out_iomap)
{
    xh2rt_buffer_view *view;
    xh2rt_result result = xh2rt_require_open(context, operation);

    if (!xh2rt_result_is_ok(result))
        return result;
    view = xh2rt_find_view(context, candidate);
    result = xh2rt_checked_view_range(
        view, offset, bytes, operation, out_absolute, out_global, out_iomap);
    if (xh2rt_result_is_ok(result) && out_view != NULL)
        *out_view = view;
    return result;
}

static int xh2rt_close_handle(int (*close_function)(fd_handle_t),
                               fd_handle_t handle, int *hal_errno)
{
    int rc;

    errno = 0;
    rc = close_function(handle);
    *hal_errno = errno;
    return rc;
}

static void xh2rt_preserve_first(xh2rt_result *first,
                                 xh2rt_result candidate)
{
    if (xh2rt_result_is_ok(*first) && !xh2rt_result_is_ok(candidate))
        *first = candidate;
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
    free(context->group_storage.pins);
    free(context);
}

static void xh2rt_cleanup_handles(xh2rt_context *context,
                                  xh2rt_result *first)
{
    int rc;
    int hal_errno;

    if (context->transfer_live) {
        rc = xh2rt_close_handle(xh2a_memory_transfer_close,
                                  context->transfer, &hal_errno);
        if (rc == 0)
            context->transfer_live = 0;
        else
            xh2rt_preserve_first(
                first,
                xh2rt_hal_result("xh2a_memory_transfer_close", rc,
                                 hal_errno));
    }
    if (context->ddr_live && !xh2rt_any_live_bo(context)) {
        rc = xh2rt_close_handle(xh2a_memory_allocator_close, context->ddr,
                                  &hal_errno);
        if (rc == 0)
            context->ddr_live = 0;
        else
            xh2rt_preserve_first(
                first,
                xh2rt_hal_result("xh2a_memory_allocator_close", rc,
                                 hal_errno));
    }
    if (context->ipu_live &&
        (context->group == NULL ||
         context->group->phase == XH2RT_GROUP_ACTIVE)) {
        rc = xh2rt_close_handle(xh2a_ipu_close, context->ipu,
                                  &hal_errno);
        if (rc == 0)
            context->ipu_live = 0;
        else
            xh2rt_preserve_first(
                first,
                xh2rt_hal_result("xh2a_ipu_close", rc,
                                 hal_errno));
    }
}

static int xh2rt_result_is_ok(xh2rt_result result)
{
    return result.error == XH2RT_ERROR_NONE;
}

static const char *xh2rt_error_string(enum xh2rt_error_kind error)
{
    switch (error) {
    case XH2RT_ERROR_NONE:
        return "ok";
    case XH2RT_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case XH2RT_ERROR_INVALID_STATE:
        return "invalid state";
    case XH2RT_ERROR_NO_MEMORY:
        return "out of memory";
    case XH2RT_ERROR_OVERFLOW:
        return "integer overflow";
    case XH2RT_ERROR_OUT_OF_RANGE:
        return "out of range";
    case XH2RT_ERROR_MISALIGNED:
        return "misaligned";
    case XH2RT_ERROR_STALE_BUFFER:
        return "stale or unknown buffer";
    case XH2RT_ERROR_BUSY:
        return "busy";
    case XH2RT_ERROR_HAL_FAILURE:
        return "HAL failure";
    case XH2RT_ERROR_HAL_PROTOCOL:
        return "invalid HAL response";
    case XH2RT_ERROR_HAL_EXECUTION:
        return "HAL execution failure";
    case XH2RT_ERROR_KERNEL_OUTCOME:
        return "kernel outcome failure";
    case XH2RT_ERROR_COMPLETION_MISSING:
        return "kernel outcome was not published";
    }
    return "unknown error";
}

static int xh2rt_publish(xh2rt_result result)
{
    const xh2rt_result *details = &result;
    int error;

    if (xh2rt_result_is_ok(result))
        return 1;

    error = details->hal_errno;
    if (error == 0) {
        switch (details->error) {
        case XH2RT_ERROR_NO_MEMORY:
            error = ENOMEM;
            break;
        case XH2RT_ERROR_OVERFLOW:
            error = EOVERFLOW;
            break;
        case XH2RT_ERROR_BUSY:
            error = EBUSY;
            break;
        case XH2RT_ERROR_INVALID_ARGUMENT:
        case XH2RT_ERROR_INVALID_STATE:
        case XH2RT_ERROR_OUT_OF_RANGE:
        case XH2RT_ERROR_MISALIGNED:
        case XH2RT_ERROR_STALE_BUFFER:
            error = EINVAL;
            break;
        default:
            error = EIO;
            break;
        }
    }

    fprintf(stderr, "xh2rt: %s: %s (hal_rc=%d hal_errno=%d",
            details->operation, xh2rt_error_string(details->error),
            details->hal_rc, details->hal_errno);
    if (details->group_id >= 0)
        fprintf(stderr,
                " group=%" PRId64 " hal_sync_result=0x%08" PRIx32
                " group_state=0x%016" PRIx64
                " group_flags=0x%016" PRIx64
                " group_kernels=%" PRIu32 " group_cores=%" PRIu32
                " group_core_mask=0x%08" PRIx32,
                details->group_id, details->hal_sync_result,
                details->group_state, details->group_flags,
                details->group_kernel_count, details->group_core_count,
                details->group_core_mask);
    if (details->invocation_index != UINT32_MAX)
        fprintf(stderr, " invocation=%" PRIu32 " hart=%" PRIu32
                " outcome=0x%08" PRIx32, details->invocation_index,
                details->hart_index, details->kernel_outcome);
    fputs(")\n", stderr);
    errno = error;
    return 0;
}

static xh2rt_result xh2rt_context_open_result(
    uint32_t logical_device, xh2rt_context **out_context)
{
    xh2rt_context *context;
    xh2rt_buffer_view *handle;
    xh2rt_result result;
    size_t index;
    int rc;
    int hal_errno;

    if (out_context == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                 "context_open");
    *out_context = NULL;
    if (logical_device != 0)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                 "context_open.logical_device");

    result = xh2rt_validate_kernel_images();
    if (!xh2rt_result_is_ok(result))
        return result;

    context = (xh2rt_context *)calloc(1, sizeof(*context));
    if (context == NULL)
        return xh2rt_make_result(XH2RT_ERROR_NO_MEMORY, "context_open");
    context->ipu = INVALID_FD_HANDLE_VAL;
    context->ddr = INVALID_FD_HANDLE_VAL;
    context->transfer = INVALID_FD_HANDLE_VAL;
    context->group_storage.id = INVALID_GROUP_ID;
    context->poison_cause = xh2rt_make_result(XH2RT_ERROR_NONE, "none");

    errno = 0;
    rc = xh2a_ipu_open(logical_device, &context->ipu);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_open", rc, hal_errno);
        xh2rt_destroy_context(context);
        return result;
    }
    if (context->ipu == INVALID_FD_HANDLE_VAL) {
        result = xh2rt_make_result(XH2RT_ERROR_HAL_PROTOCOL,
                                   "xh2a_ipu_open");
        xh2rt_destroy_context(context);
        return result;
    }
    context->ipu_live = 1;

    errno = 0;
    rc = xh2a_memory_allocator_open(
        (int)logical_device, XH2A_MEMORY_ALLOCATOR_MEMPOOL_DDR,
        &context->ddr);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_allocator_open", rc, hal_errno);
        goto fail;
    }
    if (context->ddr == INVALID_FD_HANDLE_VAL) {
        result = xh2rt_make_result(XH2RT_ERROR_HAL_PROTOCOL,
                                   "xh2a_memory_allocator_open");
        goto fail;
    }
    context->ddr_live = 1;

    errno = 0;
    rc = xh2a_memory_transfer_open((int)logical_device, &context->transfer);
    hal_errno = errno;
    if (rc != 0) {
        result =
            xh2rt_hal_result("xh2a_memory_transfer_open", rc, hal_errno);
        goto fail;
    }
    if (context->transfer == INVALID_FD_HANDLE_VAL) {
        result = xh2rt_make_result(XH2RT_ERROR_HAL_PROTOCOL,
                                   "xh2a_memory_transfer_open");
        goto fail;
    }
    context->transfer_live = 1;

    /* Upload once before any batch can start. Code uses the same BO pinning
     * and failure quarantine as data, and stays owned by this context. */
    for (index = 0; index < XH2RT_KERNEL_COUNT; index++) {
        result = xh2rt_buffer_alloc_result(
            context, xh2rt_kernels[index].code_bytes,
            &handle);
        if (!xh2rt_result_is_ok(result))
            goto fail;
        context->kernel_views[index] = xh2rt_find_view(context, handle);
        if (context->kernel_views[index] == NULL) {
            result = xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                       "context_open.kernel_view");
            goto fail;
        }
        result = xh2rt_buffer_write_result(
            context, handle, 0,
            xh2rt_kernels[index].code,
            xh2rt_kernels[index].code_bytes);
        if (!xh2rt_result_is_ok(result))
            goto fail;
    }
    result = xh2rt_buffer_alloc_result(context, XH2RT_COMPLETION_ARENA_BYTES,
                                       &handle);
    if (!xh2rt_result_is_ok(result))
        goto fail;
    context->completion_view = xh2rt_find_view(context, handle);
    if (context->completion_view == NULL) {
        result = xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                   "context_open.completion_view");
        goto fail;
    }
    result = xh2rt_completion_transfer(context, XH2RT_GROUP_MAX_INVOCATIONS, 0);
    if (!xh2rt_result_is_ok(result)) {
        result = xh2rt_poison(context, result);
        goto fail;
    }
    *out_context = context;
    return xh2rt_make_result(XH2RT_ERROR_NONE, "context_open");

fail:
    (void)xh2rt_context_close_result(&context);
    *out_context = context;
    return result;
}

static xh2rt_result xh2rt_context_close_result(
    xh2rt_context **context_pointer)
{
    xh2rt_context *context;
    xh2rt_allocation *allocation;
    xh2rt_buffer_view *view;
    xh2rt_result first = xh2rt_make_result(XH2RT_ERROR_NONE,
                                           "context_close");

    if (context_pointer == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                 "context_close");
    context = *context_pointer;
    if (context == NULL)
        return first;

    if (context->group != NULL &&
        context->group->phase == XH2RT_GROUP_ACTIVE) {
        xh2rt_result result =
            xh2rt_drain_group(context, "context_close.drain");

        if (!xh2rt_result_is_ok(result))
            xh2rt_preserve_first(&first, result);
    }
    if (context->group != NULL &&
        context->group->phase == XH2RT_GROUP_DESTROY_PENDING) {
        int rc;
        int hal_errno;

        errno = 0;
        rc = xh2a_ipu_destroy_group(context->ipu, context->group->id);
        hal_errno = errno;
        if (rc == 0) {
            xh2rt_group_release_host(context, context->group);
        } else {
            xh2rt_result result =
                xh2rt_hal_result("xh2a_ipu_destroy_group", rc,
                                 hal_errno);
            xh2rt_stamp_group(&result, context->group);
            xh2rt_preserve_first(&first, result);
        }
    }
    context->batch_active = 0;
    context->phase = XH2RT_CONTEXT_CLOSING;
    for (view = context->views; view != NULL; view = view->next) {
        view->active = 0;
        view->references = 0;
    }
    for (allocation = context->allocations; allocation != NULL;
         allocation = allocation->next) {
        int rc;
        int hal_errno;

        allocation->logical_references = 0;
        if (!allocation->bo_live)
            continue;
        if (allocation->inflight != 0 || allocation->quarantined) {
            xh2rt_preserve_first(
                &first,
                xh2rt_make_result(XH2RT_ERROR_BUSY,
                                  allocation->quarantined
                                      ? "context_close.quarantine"
                                      : "context_close.inflight"));
            continue;
        }
        errno = 0;
        rc = xh2a_memory_allocator_free_buffer_object(&allocation->bo);
        hal_errno = errno;
        (void)hal_errno; /* Primary metadata failure remains authoritative. */
        if (rc == 0)
            allocation->bo_live = 0;
        else
            xh2rt_preserve_first(
                &first,
                xh2rt_hal_result("xh2a_memory_allocator_free_buffer_object",
                                  rc, hal_errno));
    }

    xh2rt_cleanup_handles(context, &first);
    if (context->group == NULL && !xh2rt_any_live_bo(context) &&
        !context->transfer_live && !context->ddr_live &&
        !context->ipu_live) {
        xh2rt_destroy_context(context);
        *context_pointer = NULL;
        return first;
    }
    if (xh2rt_result_is_ok(first))
        first = xh2rt_make_result(XH2RT_ERROR_BUSY, "context_close");
    return first;
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
        return xh2rt_make_result(XH2RT_ERROR_HAL_PROTOCOL,
                                 "xh2a_memory_allocator_alloc_buffer_object.metadata");
    }
    *iomap = bo->start - XH2RT_IOMAP_DELTA;
    if (!xh2rt_u64_add(bo->start, bo->size, &global_end) ||
        !xh2rt_u64_add(*iomap, bo->size, &iomap_end) ||
        *iomap > (uint64_t)UINTPTR_MAX ||
        bo->size > (uint64_t)UINTPTR_MAX - *iomap) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW,
                                 "xh2a_memory_allocator_alloc_buffer_object.metadata");
    }
    (void)global_end;
    (void)iomap_end;
    if (xh2rt_allocations_overlap(context, bo->start, bo->size))
        return xh2rt_make_result(XH2RT_ERROR_HAL_PROTOCOL,
                                 "xh2a_memory_allocator_alloc_buffer_object.overlap");
    return xh2rt_make_result(XH2RT_ERROR_NONE, "buffer_alloc");
}

static xh2rt_result xh2rt_free_allocation(xh2rt_context *context,
                                          xh2rt_allocation *allocation,
                                          const char *operation)
{
    int rc;
    int hal_errno;

    if (allocation == NULL || !allocation->bo_live)
        return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
    assert((allocation->pinned_group == NULL) ==
           (allocation->inflight == 0));
    if (allocation->inflight != 0 || allocation->quarantined)
        return xh2rt_make_result(XH2RT_ERROR_BUSY, operation);
    errno = 0;
    rc = xh2a_memory_allocator_free_buffer_object(&allocation->bo);
    hal_errno = errno;
    if (rc != 0)
        return xh2rt_hal_result("xh2a_memory_allocator_free_buffer_object",
                              rc, hal_errno);
    xh2rt_forget_allocation(context, allocation);
    return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
}

static void xh2rt_stamp_group(xh2rt_result *result,
                              const xh2rt_group *group)
{
    result->group_id = group->id;
    result->group_kernel_count = group->invocation_count;
    result->group_core_count = XH2RT_CORE_COUNT;
}

/* Best-effort failure diagnostics; this query never defines the primary cause. */
static void xh2rt_capture_group_info(xh2rt_context *context,
                                     xh2rt_group *group,
                                     xh2rt_result *details)
{
    struct xh2a_group_info info = {.group_id = group->id};
    int rc;

    xh2rt_stamp_group(details, group);
    errno = 0;
    rc = xh2a_ipu_get_group_info(context->ipu, &info);
    if (rc != 0)
        return;
    details->group_id = info.group_id;
    details->group_state = info.state;
    details->group_flags = info.flags;
    details->group_kernel_count = info.kernel_num;
    details->group_core_count = info.core_num;
    details->group_core_mask = info.core_mask;
}

static void xh2rt_group_release_host(xh2rt_context *context,
                                     xh2rt_group *group)
{
    assert(group == &context->group_storage);
    assert(context->group == group);
    assert(group->pin_count == 0);
    context->group = NULL;
    group->id = INVALID_GROUP_ID;
    group->invocation_count = 0;
    group->phase = XH2RT_GROUP_ACTIVE;
}

static void xh2rt_group_unpin(xh2rt_context *context,
                              xh2rt_group *group,
                              xh2rt_result *first)
{
    size_t index;

    for (index = 0; index < group->pin_count; index++) {
        xh2rt_allocation *allocation = group->pins[index];
        xh2rt_result result;

        assert(allocation->pinned_group == group);
        assert(allocation->inflight == 1);
        group->pins[index] = NULL;
        allocation->pinned_group = NULL;
        allocation->inflight--;
        if (allocation->logical_references == 0 && allocation->bo_live &&
            !allocation->quarantined && allocation->inflight == 0) {
            result = xh2rt_free_allocation(context, allocation,
                                           "group_unpin");
            if (!xh2rt_result_is_ok(result))
                xh2rt_preserve_first(first, result);
        }
    }
    group->pin_count = 0;
}

/* Private transfers must not call the public transfer API, which drains a
 * group first. Reset only records consumed by the previous group. */
static xh2rt_result xh2rt_completion_transfer(xh2rt_context *context,
                                              uint32_t records, int readback)
{
    xh2rt_buffer_view *view = context->completion_view;
    uint64_t global = 0;
    uint64_t iomap = 0;
    size_t bytes = (size_t)records * XH2RT_KERNEL_COMPLETION_RECORD_BYTES;
    uint64_t host = (uint64_t)(uintptr_t)context->completion_host;
    const char *operation = readback ? "completion.read"
                                     : "completion.initialize";
    xh2rt_result result = xh2rt_checked_view_range(
        view, 0, bytes, operation, NULL, &global, &iomap);
    int rc;
    int hal_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    if ((iomap & (sizeof(uint32_t) - 1)) != 0)
        return xh2rt_make_result(XH2RT_ERROR_MISALIGNED, operation);
    if (!readback)
        /* UINT32_MAX is the endian-independent all-bytes-set sentinel. */
        memset(context->completion_host, 0xff, bytes);
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(
        context->transfer, readback ? global : host,
        readback ? host : global, bytes,
        readback ? XH2A_MEMORY_TRANSFER_TYPE_DEVICE_TO_HOST
                 : XH2A_MEMORY_TRANSFER_TYPE_HOST_TO_DEVICE);
    hal_errno = errno;
    if (rc != 0) {
        view->allocation->quarantined = 1;
        return xh2rt_hal_result(operation, rc, hal_errno);
    }
    return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
}

static xh2rt_result xh2rt_ensure_group(xh2rt_context *context,
                                       size_t pin_slots)
{
    xh2rt_group *group;
    xh2rt_result result;
    int rc;
    int hal_errno;

    if (context->group != NULL) {
        assert(context->group == &context->group_storage);
        assert(context->group->phase == XH2RT_GROUP_ACTIVE);
        if (!xh2rt_group_reserve_pins(context->group, pin_slots)) {
            return xh2rt_make_result(XH2RT_ERROR_NO_MEMORY,
                                  "launch.group_resources");
        }
        return xh2rt_make_result(XH2RT_ERROR_NONE, "group_reuse");
    }
    if (context->completion_records_to_reset != 0) {
        result = xh2rt_completion_transfer(
            context, context->completion_records_to_reset, 0);
        if (!xh2rt_result_is_ok(result))
            return xh2rt_poison(context, result);
        context->completion_records_to_reset = 0;
    }
    group = &context->group_storage;
    assert(group->pin_count == 0);
    group->id = INVALID_GROUP_ID;
    group->invocation_count = 0;
    group->phase = XH2RT_GROUP_ACTIVE;
    /* Reserve fallible host bookkeeping before creating a HAL group.  Once
     * the group exists, every error path must either submit or destroy it. */
    if (!xh2rt_group_reserve_pins(group, pin_slots)) {
        return xh2rt_make_result(XH2RT_ERROR_NO_MEMORY,
                              "launch.group_resources");
    }
    errno = 0;
    rc = xh2a_ipu_create_group(context->ipu, &group->id);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_create_group", rc,
                                  hal_errno);
        group->id = INVALID_GROUP_ID;
        return result;
    }
    if (group->id == INVALID_GROUP_ID || group->id < 0) {
        result = xh2rt_make_result(XH2RT_ERROR_HAL_PROTOCOL,
                                   "xh2a_ipu_create_group");
        group->id = INVALID_GROUP_ID;
        return result;
    }
    context->group = group;
    return xh2rt_make_result(XH2RT_ERROR_NONE, "group_create");
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
    if (allocation->pinned_group == group)
        return;
    assert(allocation->pinned_group == NULL);
    assert(allocation->inflight == 0);
    assert(group->pin_count < group->pin_capacity);
    group->pins[group->pin_count++] = allocation;
    allocation->pinned_group = group;
    allocation->inflight++;
}

static xh2rt_result xh2rt_drain_group(xh2rt_context *context,
                                      const char *operation)
{
    xh2rt_group *group = context->group;
    xh2rt_result primary = xh2rt_make_result(XH2RT_ERROR_NONE, operation);
    xh2rt_result cleanup = xh2rt_make_result(XH2RT_ERROR_NONE,
                                             "group_cleanup");
    uint32_t hal_sync_result = 0;
    int rc;
    int hal_errno;
    int destroy_rc;
    int destroy_errno;

    if (group == NULL)
        return primary;
    if (group->phase != XH2RT_GROUP_ACTIVE)
        return context->poison_cause;

    errno = 0;
    rc = xh2a_ipu_sync_group(context->ipu, group->id,
                              XH2RT_GROUP_SYNC_TIMEOUT_US,
                              &hal_sync_result);
    hal_errno = errno;
    if (rc != 0) {
        primary = xh2rt_hal_result("xh2a_ipu_sync_group", rc,
                                   hal_errno);
        primary.hal_sync_result = hal_sync_result;
        xh2rt_stamp_group(&primary, group);
        xh2rt_capture_group_info(context, group, &primary);
        group->phase = XH2RT_GROUP_QUIESCENCE_UNKNOWN;
        return xh2rt_poison(context, primary);
    }

    primary.hal_sync_result = hal_sync_result;
    xh2rt_stamp_group(&primary, group);
    if (hal_sync_result != 0) {
        xh2rt_result cause = xh2rt_make_result(
            XH2RT_ERROR_HAL_EXECUTION, "xh2a_ipu_sync_group.result");

        cause.hal_errno = hal_errno;
        xh2rt_result_set_cause(&primary, cause);
    } else if (group->invocation_count != 0) {
        xh2rt_result completion_result =
            xh2rt_completion_transfer(context, group->invocation_count, 1);
        uint32_t invocation;

        if (!xh2rt_result_is_ok(completion_result)) {
            xh2rt_result_set_cause(&primary, completion_result);
        } else {
            /* Submission order, then hart order, defines the first failure. */
            for (invocation = 0;
                 invocation < group->invocation_count &&
                     xh2rt_result_is_ok(primary);
                 invocation++) {
                uint32_t hart;
                const uint8_t *record = context->completion_host +
                    (size_t)invocation *
                        XH2RT_KERNEL_COMPLETION_RECORD_BYTES;

                for (hart = 0; hart < XH2RT_HART_COUNT; hart++) {
                    const uint8_t *slot = record + hart * sizeof(uint32_t);
                    uint32_t outcome = (uint32_t)slot[0] |
                        ((uint32_t)slot[1] << 8) | ((uint32_t)slot[2] << 16) |
                        ((uint32_t)slot[3] << 24);

                    if (outcome == XH2RT_KERNEL_OUTCOME_OK)
                        continue;
                    primary.error = outcome == XH2RT_COMPLETION_UNWRITTEN
                        ? XH2RT_ERROR_COMPLETION_MISSING
                        : XH2RT_ERROR_KERNEL_OUTCOME;
                    primary.operation = "completion.outcome";
                    primary.kernel_outcome = outcome;
                    primary.invocation_index = invocation;
                    primary.hart_index = hart;
                    break;
                }
            }
        }
    }
    context->completion_records_to_reset = group->invocation_count;

    /* sync_group already proved completion. Group info is diagnostics for an
     * observed execution failure, never an extra call on the success path. */
    if (!xh2rt_result_is_ok(primary))
        xh2rt_capture_group_info(context, group, &primary);

    errno = 0;
    destroy_rc = xh2a_ipu_destroy_group(context->ipu, group->id);
    destroy_errno = errno;
    if (destroy_rc != 0 && xh2rt_result_is_ok(primary)) {
        xh2rt_result cause = xh2rt_hal_result(
            "xh2a_ipu_destroy_group", destroy_rc, destroy_errno);

        /* Destruction is the first fatal cause; diagnostics cannot replace it. */
        xh2rt_result_set_cause(&primary, cause);
        /* The SDK does not promise the failed destroy preserved the handle;
         * query only as best-effort diagnostics and retain destroy as cause. */
        xh2rt_capture_group_info(context, group, &primary);
    }
    xh2rt_group_unpin(context, group, &cleanup);
    if (destroy_rc == 0) {
        xh2rt_group_release_host(context, group);
    } else {
        group->phase = XH2RT_GROUP_DESTROY_PENDING;
    }

    /* Fatal execution/lifecycle failures outrank recoverable observations;
     * within each class, preserve the first operation in execution order. */
    if (!xh2rt_result_is_ok(primary))
        return xh2rt_poison(context, primary);
    if (!xh2rt_result_is_ok(cleanup)) {
        xh2rt_result_set_cause(&primary, cleanup);
        return primary;
    }
    return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
}

/* Typed wrappers describe native words and complete buffer ranges. The runtime
 * resolves IOMAP words and owns the hidden completion slot and all pins. */
static xh2rt_result xh2rt_kernel_invoke_result(
    xh2rt_context *context, const char *name,
    const xh2rt_kernel_slot *slots, size_t slot_count, int submit)
{
    xh2rt_buffer_view *kernel_view;
    xh2rt_buffer_view *completion_view;
    xh2rt_allocation *slot_allocations[XH2RT_KERNEL_MAX_SLOTS];
    uint64_t slot_words[XH2RT_KERNEL_MAX_SLOTS];
    uint8_t envelope[XH2RT_KERNEL_MAX_ENVELOPE_BYTES];
    uint64_t completion_arena_iomap;
    uint64_t completion_iomap;
    size_t native_slot_count;
    size_t envelope_size;
    uint32_t envelope_bytes;
    size_t index;
    size_t kernel_index;
    xh2rt_group *group;
    struct kernel_launch_data launch;
    xh2rt_result result = xh2rt_require_healthy(context, "launch");
    int one_shot;
    int rc;
    int hal_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    if (name == NULL || name[0] == '\0')
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                 "launch.kernel");
    kernel_index = xh2rt_find_kernel_index(name);
    if (kernel_index == XH2RT_KERNEL_COUNT)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                 "launch.kernel");
    if (slot_count != 0 && slots == NULL) {
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                 "launch.slots");
    }
    if (slot_count > XH2RT_KERNEL_MAX_SLOTS) {
        return xh2rt_make_result(XH2RT_ERROR_OUT_OF_RANGE,
                                 "launch.slots");
    }
    native_slot_count = slot_count + 1;
    envelope_size = XH2RT_KERNEL_ENVELOPE_BYTES(slot_count);
    envelope_bytes = (uint32_t)envelope_size;

    kernel_view = context->kernel_views[kernel_index];
    if (kernel_view == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                 "launch.kernel");
    if (!xh2rt_view_is_live(kernel_view))
        return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                                 "launch.kernel");

    for (index = 0; index < slot_count; index++) {
        const xh2rt_kernel_slot *slot = &slots[index];

        slot_allocations[index] = NULL;
        switch (slot->kind) {
        case XH2RT_KERNEL_SLOT_WORD:
            slot_words[index] = slot->word;
            break;
        case XH2RT_KERNEL_SLOT_BUFFER: {
            xh2rt_buffer_view *view;

            if (!xh2rt_alignment_valid(slot->alignment)) {
                return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                         "launch.slot_alignment");
            }
            result = xh2rt_checked_range(
                context, slot->buffer, slot->offset, slot->bytes,
                "launch.slot", &view, NULL, NULL, &slot_words[index]);
            if (!xh2rt_result_is_ok(result))
                return result;
            if ((slot_words[index] & (slot->alignment - 1)) != 0)
                return xh2rt_make_result(XH2RT_ERROR_MISALIGNED,
                                         "launch.slot");
            slot_allocations[index] = view->allocation;
            break;
        }
        default:
            return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                     "launch.slot_kind");
        }
    }
    if (!submit)
        return xh2rt_make_result(XH2RT_ERROR_NONE, "launch");

    completion_view = context->completion_view;
    result = xh2rt_checked_view_range(
        completion_view, 0, XH2RT_COMPLETION_ARENA_BYTES,
        "launch.completion", NULL, NULL, &completion_arena_iomap);
    if (!xh2rt_result_is_ok(result))
        return result;
    if ((completion_arena_iomap & (sizeof(uint32_t) - 1)) != 0)
        return xh2rt_make_result(XH2RT_ERROR_MISALIGNED,
                                 "launch.completion");

    one_shot = !context->batch_active;
    group = context->group;
    if (group != NULL &&
        group->invocation_count == XH2RT_GROUP_MAX_INVOCATIONS) {
        result = xh2rt_drain_group(context, "launch.split");
        if (!xh2rt_result_is_ok(result))
            return result;
    }
    result = xh2rt_ensure_group(context, slot_count + 2);
    if (!xh2rt_result_is_ok(result))
        return result;
    group = context->group;
    xh2rt_group_pin_one(group, kernel_view->allocation);
    for (index = 0; index < slot_count; index++) {
        if (slot_allocations[index] != NULL)
            xh2rt_group_pin_one(group, slot_allocations[index]);
    }
    xh2rt_group_pin_one(group, completion_view->allocation);
    completion_iomap = completion_arena_iomap +
        (uint64_t)group->invocation_count * XH2RT_KERNEL_COMPLETION_RECORD_BYTES;
    memset(envelope, 0, 8);
    xh2rt_put_le64(envelope + 8, native_slot_count);
    xh2rt_put_le64(envelope + 16, completion_iomap);
    for (index = 0; index < slot_count; index++)
        xh2rt_put_le64(envelope + 24 + index * 8,
                       slot_words[index]);

    memset(&launch, 0, sizeof(launch));
    launch.kernel_addr =
        kernel_view->allocation->iomap + kernel_view->offset;
    launch.kernel_size = (uint32_t)kernel_view->bytes;
    launch.param_phy_addr = (uint64_t)(uintptr_t)envelope;
    launch.param_type = XH2A_GROUP_PARAM_SPM;
    launch.param_size = envelope_bytes;
    /* Each of the 32 launched harts owns one completion outcome. */
    launch.core_num = XH2RT_CORE_COUNT;
    launch.tile_num = XH2RT_TILE_COUNT;
    launch.ilm_mode = UINT32_C(1);
    launch.timeout_us = UINT32_C(1000000);
    errno = 0;
    /* SPM launch copies these host bytes during the call, not during sync. */
    rc = xh2a_ipu_launch_kernel(context->ipu, group->id, &launch);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_ipu_launch_kernel", rc,
                                   hal_errno);
        xh2rt_stamp_group(&result, group);
        xh2rt_capture_group_info(context, group, &result);
        group->phase = XH2RT_GROUP_QUIESCENCE_UNKNOWN;
        return xh2rt_poison(context, result);
    }
    group->invocation_count++;
    return one_shot
        ? xh2rt_drain_group(context, "launch.one_shot")
        : xh2rt_make_result(XH2RT_ERROR_NONE, "launch");
}

static xh2rt_result xh2rt_commands_begin_result(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "commands_begin");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (context->batch_active)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                              "commands_begin");
    context->batch_active = 1;
    return result;
}

int xh2rt_commands_active(const xh2rt_context *context)
{
    return context != NULL && context->phase == XH2RT_CONTEXT_OPEN &&
           context->batch_active;
}

static xh2rt_result xh2rt_commands_flush_result(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "commands_flush");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (!context->batch_active)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                              "commands_flush");
    return xh2rt_drain_group(context, "commands_flush");
}

static xh2rt_result xh2rt_commands_end_result(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "commands_end");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (!context->batch_active)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_STATE,
                              "commands_end");
    result = xh2rt_drain_group(context, "commands_end");
    context->batch_active = 0;
    return result;
}

static xh2rt_result xh2rt_synchronize_result(xh2rt_context *context)
{
    xh2rt_result result = xh2rt_require_healthy(context,
                                                "synchronize");

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_drain_group(context, "synchronize");
    context->batch_active = 0;
    return result;
}

static xh2rt_result xh2rt_buffer_alloc_result(
    xh2rt_context *context, uint64_t bytes,
    xh2rt_buffer_view **out_buffer)
{
    xh2rt_allocation *allocation;
    xh2rt_buffer_view *view;
    xh2rt_result result = xh2rt_require_healthy(context, "buffer_alloc");
    int rc;
    int hal_errno;

    if (out_buffer == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_alloc.out_buffer");
    *out_buffer = NULL;
    if (!xh2rt_result_is_ok(result))
        return result;
    if (bytes == 0)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_alloc.bytes");

    allocation = (xh2rt_allocation *)calloc(1, sizeof(*allocation));
    view = (xh2rt_buffer_view *)calloc(1, sizeof(*view));
    if (allocation == NULL || view == NULL || !xh2rt_assign_handle(view)) {
        free(allocation);
        free(view);
        return xh2rt_make_result(XH2RT_ERROR_NO_MEMORY, "buffer_alloc");
    }
    allocation->bo.memory_allocator_fd = INVALID_FD_HANDLE_VAL;
    errno = 0;
    rc = xh2a_memory_allocator_alloc_buffer_object(context->ddr, bytes,
                                                  &allocation->bo);
    hal_errno = errno;
    if (rc != 0) {
        free(allocation);
        free(view);
        return xh2rt_hal_result("xh2a_memory_allocator_alloc_buffer_object",
                              rc, hal_errno);
    }
    allocation->bo_live = 1;
    allocation->accessible_bytes = bytes;
    result = xh2rt_validate_bo(context, &allocation->bo, bytes,
                               &allocation->iomap);
    if (!xh2rt_result_is_ok(result)) {
        errno = 0;
        rc = xh2a_memory_allocator_free_buffer_object(&allocation->bo);
        hal_errno = errno;
        (void)hal_errno; /* Snapshot before retaining the primary error. */
        if (rc == 0) {
            free(allocation);
        } else {
            allocation->next = context->allocations;
            context->allocations = allocation;
            context->phase = XH2RT_CONTEXT_CLOSING;
        }
        free(view);
        return result;
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
    return xh2rt_make_result(XH2RT_ERROR_NONE, "buffer_alloc");
}

static xh2rt_result xh2rt_buffer_retain_result(
    xh2rt_context *context, xh2rt_buffer_view *buffer)
{
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result =
        xh2rt_require_view(context, buffer, "buffer_retain", &view);

    if (!xh2rt_result_is_ok(result))
        return result;
    if (view->references == UINT64_MAX ||
        view->allocation->logical_references == UINT64_MAX) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, "buffer_retain");
    }
    view->references++;
    view->allocation->logical_references++;
    return result;
}

static xh2rt_result xh2rt_buffer_release_result(
    xh2rt_context *context, xh2rt_buffer_view **buffer_pointer)
{
    xh2rt_allocation *allocation;
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result;

    if (buffer_pointer == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_release");
    if (*buffer_pointer == NULL)
        return xh2rt_make_result(XH2RT_ERROR_NONE, "buffer_release");
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
            result = xh2rt_make_result(XH2RT_ERROR_NONE,
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

static xh2rt_result xh2rt_buffer_view_create_result(
    xh2rt_context *context, xh2rt_buffer_view *buffer,
    uint64_t offset, uint64_t bytes, xh2rt_buffer_view **out_view)
{
    xh2rt_buffer_view *parent = NULL;
    xh2rt_buffer_view *view = NULL;
    uint64_t absolute;
    xh2rt_result result;

    if (out_view == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_view_create.out_view");
    *out_view = NULL;
    result = xh2rt_require_view(context, buffer, "buffer_view_create",
                                &parent);
    if (!xh2rt_result_is_ok(result))
        return result;
    if (!xh2rt_range_fits(parent->bytes, offset, bytes)) {
        return xh2rt_make_result(XH2RT_ERROR_OUT_OF_RANGE,
                              "buffer_view_create");
    }
    if (!xh2rt_u64_add(parent->offset, offset, &absolute)) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW,
                              "buffer_view_create");
    }
    if (parent->allocation->logical_references == UINT64_MAX) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW,
                              "buffer_view_create");
    }
    view = (xh2rt_buffer_view *)calloc(1, sizeof(*view));
    if (view == NULL || !xh2rt_assign_handle(view)) {
        free(view);
        return xh2rt_make_result(XH2RT_ERROR_NO_MEMORY,
                              "buffer_view_create");
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

static xh2rt_result xh2rt_buffer_bytes_result(
    xh2rt_context *context, const xh2rt_buffer_view *buffer,
    uint64_t *out_bytes)
{
    xh2rt_buffer_view *view = NULL;
    xh2rt_result result;

    if (out_bytes == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_bytes");
    *out_bytes = 0;
    result = xh2rt_require_view(context, buffer, "buffer_bytes", &view);
    if (!xh2rt_result_is_ok(result))
        return result;
    *out_bytes = view->bytes;
    return result;
}

static xh2rt_result xh2rt_check_host_range(const void *pointer,
                                           uint64_t bytes,
                                           const char *operation)
{
    uintptr_t address;

    if (bytes == 0)
        return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
    if (pointer == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT, operation);
    if (bytes > (uint64_t)SIZE_MAX)
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, operation);
    address = (uintptr_t)pointer;
    if (bytes > (uint64_t)UINTPTR_MAX - (uint64_t)address)
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, operation);
    return xh2rt_make_result(XH2RT_ERROR_NONE, operation);
}

static xh2rt_result xh2rt_buffer_write_result(
    xh2rt_context *context, xh2rt_buffer_view *destination,
    uint64_t destination_offset, const void *source, uint64_t bytes)
{
    xh2rt_buffer_view *destination_view = NULL;
    uint64_t global;
    xh2rt_result result = xh2rt_require_healthy(context, "buffer_write");
    int rc;
    int hal_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_checked_range(
        context, destination, destination_offset, bytes, "buffer_write",
        &destination_view, NULL, &global, NULL);

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_check_host_range(source, bytes, "buffer_write");
    if (!xh2rt_result_is_ok(result) || bytes == 0)
        return result;
    result = xh2rt_drain_group(context, "buffer_write.drain");
    if (!xh2rt_result_is_ok(result))
        return result;
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(context->transfer, (uint64_t)(uintptr_t)source,
                        global, (size_t)bytes,
                        XH2A_MEMORY_TRANSFER_TYPE_HOST_TO_DEVICE);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_transfer_copy_buffer(H2D)", rc,
                                  hal_errno);
        destination_view->allocation->quarantined = 1;
        return xh2rt_poison(context, result);
    }
    return xh2rt_make_result(XH2RT_ERROR_NONE, "buffer_write");
}

static xh2rt_result xh2rt_buffer_read_result(
    xh2rt_context *context, void *destination,
    xh2rt_buffer_view *source, uint64_t source_offset, uint64_t bytes)
{
    xh2rt_buffer_view *source_view = NULL;
    uint64_t global;
    xh2rt_result result = xh2rt_require_healthy(context, "buffer_read");
    int rc;
    int hal_errno;

    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_checked_range(
        context, source, source_offset, bytes, "buffer_read", &source_view,
        NULL, &global, NULL);
    if (!xh2rt_result_is_ok(result))
        return result;
    result = xh2rt_check_host_range(destination, bytes, "buffer_read");
    if (!xh2rt_result_is_ok(result) || bytes == 0)
        return result;
    result = xh2rt_drain_group(context, "buffer_read.drain");
    if (!xh2rt_result_is_ok(result))
        return result;
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(context->transfer, global,
                        (uint64_t)(uintptr_t)destination, (size_t)bytes,
                        XH2A_MEMORY_TRANSFER_TYPE_DEVICE_TO_HOST);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_transfer_copy_buffer(D2H)", rc,
                                  hal_errno);
        source_view->allocation->quarantined = 1;
        return xh2rt_poison(context, result);
    }
    return xh2rt_make_result(XH2RT_ERROR_NONE, "buffer_read");
}

static int xh2rt_ranges_overlap(uint64_t left, uint64_t right,
                                uint64_t bytes)
{
    if (left <= right)
        return right - left < bytes;
    return left - right < bytes;
}

static xh2rt_result xh2rt_buffer_copy_result(
    xh2rt_context *context, xh2rt_buffer_view *destination,
    uint64_t destination_offset, xh2rt_buffer_view *source,
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
    int hal_errno;

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
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, "buffer_copy");
    if (bytes == 0)
        return result;

    if (destination_view->allocation == source_view->allocation) {
        if (destination_absolute == source_absolute)
            return result;
        if (xh2rt_ranges_overlap(destination_absolute, source_absolute,
                                 bytes)) {
            return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                                  "buffer_copy.overlap");
        }
    }

    result = xh2rt_drain_group(context, "buffer_copy.drain");
    if (!xh2rt_result_is_ok(result))
        return result;
    errno = 0;
    rc = xh2a_memory_transfer_copy_buffer(context->transfer, source_global,
                        destination_global, (size_t)bytes,
                        XH2A_MEMORY_TRANSFER_TYPE_INNER_DEVICE);
    hal_errno = errno;
    if (rc != 0) {
        result = xh2rt_hal_result("xh2a_memory_transfer_copy_buffer(D2D)", rc,
                                  hal_errno);
        destination_view->allocation->quarantined = 1;
        source_view->allocation->quarantined = 1;
        return xh2rt_poison(context, result);
    }
    return xh2rt_make_result(XH2RT_ERROR_NONE, "buffer_copy");
}

static xh2rt_result xh2rt_buffer_iomap_result(xh2rt_context *context, const xh2rt_buffer_view *buffer, uint64_t offset, uint64_t bytes, uint64_t alignment, uintptr_t *out_iomap)
{
    uint64_t iomap;
    xh2rt_result result = xh2rt_require_healthy(
        context, "buffer_iomap");

    if (!xh2rt_result_is_ok(result))
        return result;
    if (out_iomap == NULL)
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_iomap");
    *out_iomap = 0;
    if (!xh2rt_alignment_valid(alignment))
        return xh2rt_make_result(XH2RT_ERROR_INVALID_ARGUMENT,
                              "buffer_iomap.alignment");
    result = xh2rt_checked_range(context, buffer, offset, bytes,
                                 "buffer_iomap", NULL, NULL, NULL, &iomap);
    if (!xh2rt_result_is_ok(result))
        return result;
    if ((iomap & (alignment - 1)) != 0)
        return xh2rt_make_result(XH2RT_ERROR_MISALIGNED, "buffer_iomap");
    if (iomap > (uint64_t)UINTPTR_MAX ||
        bytes > (uint64_t)UINTPTR_MAX - iomap) {
        return xh2rt_make_result(XH2RT_ERROR_OVERFLOW, "buffer_iomap");
    }
    *out_iomap = (uintptr_t)iomap;
    return result;
}

int xh2rt_context_open(uint32_t logical_device, xh2rt_context **out_context)
{
    xh2rt_result result =
        xh2rt_context_open_result(logical_device, out_context);

    return xh2rt_publish(result);
}

int xh2rt_context_close(xh2rt_context **context_pointer)
{
    xh2rt_result result = xh2rt_context_close_result(context_pointer);

    return xh2rt_publish(result);
}

int xh2rt_kernel_invoke(xh2rt_context *context,
                        const char *name,
                        const xh2rt_kernel_slot *slots, size_t slot_count,
                        int submit)
{
    return xh2rt_publish(xh2rt_kernel_invoke_result(
        context, name, slots, slot_count, submit));
}

int xh2rt_commands_begin(xh2rt_context *context)
{
    return xh2rt_publish(xh2rt_commands_begin_result(context));
}

int xh2rt_commands_flush(xh2rt_context *context)
{
    return xh2rt_publish(xh2rt_commands_flush_result(context));
}

int xh2rt_commands_end(xh2rt_context *context)
{
    return xh2rt_publish(xh2rt_commands_end_result(context));
}

int xh2rt_synchronize(xh2rt_context *context)
{
    return xh2rt_publish(xh2rt_synchronize_result(context));
}

int xh2rt_buffer_alloc(xh2rt_context *context, uint64_t bytes,
                       xh2rt_buffer_view **out_buffer)
{
    return xh2rt_publish(
        xh2rt_buffer_alloc_result(context, bytes, out_buffer));
}

int xh2rt_buffer_retain(xh2rt_context *context, xh2rt_buffer_view *buffer)
{
    return xh2rt_publish(xh2rt_buffer_retain_result(context, buffer));
}

int xh2rt_buffer_release(xh2rt_context *context, xh2rt_buffer_view **buffer)
{
    return xh2rt_publish(xh2rt_buffer_release_result(context, buffer));
}

int xh2rt_buffer_view_create(xh2rt_context *context,
                             xh2rt_buffer_view *buffer, uint64_t offset,
                             uint64_t bytes, xh2rt_buffer_view **out_view)
{
    return xh2rt_publish(
        xh2rt_buffer_view_create_result(context, buffer, offset, bytes,
                                        out_view));
}

int xh2rt_buffer_bytes(xh2rt_context *context,
                       const xh2rt_buffer_view *buffer,
                       uint64_t *out_bytes)
{
    return xh2rt_publish(
        xh2rt_buffer_bytes_result(context, buffer, out_bytes));
}

int xh2rt_buffer_write(xh2rt_context *context,
                       xh2rt_buffer_view *destination,
                       uint64_t destination_offset, const void *source,
                       uint64_t bytes)
{
    return xh2rt_publish(
        xh2rt_buffer_write_result(context, destination, destination_offset,
                                  source, bytes));
}

int xh2rt_buffer_read(xh2rt_context *context, void *destination,
                      xh2rt_buffer_view *source, uint64_t source_offset,
                      uint64_t bytes)
{
    return xh2rt_publish(
        xh2rt_buffer_read_result(context, destination, source, source_offset,
                                 bytes));
}

int xh2rt_buffer_copy(xh2rt_context *context,
                      xh2rt_buffer_view *destination,
                      uint64_t destination_offset, xh2rt_buffer_view *source,
                      uint64_t source_offset, uint64_t bytes)
{
    return xh2rt_publish(
        xh2rt_buffer_copy_result(context, destination, destination_offset,
                                 source, source_offset, bytes));
}

int xh2rt_buffer_iomap(xh2rt_context *context,
                       const xh2rt_buffer_view *buffer, uint64_t offset,
                       uint64_t bytes, uint64_t alignment,
                       uintptr_t *out_iomap)
{
    return xh2rt_publish(
        xh2rt_buffer_iomap_result(context, buffer, offset, bytes, alignment,
                                  out_iomap));
}
