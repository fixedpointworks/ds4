#ifndef DS4_TCIM_XH2RT_H
#define DS4_TCIM_XH2RT_H

#include <stddef.h>
#include <stdint.h>

typedef struct xh2rt_context xh2rt_context;
typedef struct xh2rt_buffer_view xh2rt_buffer_view;

enum xh2rt_status {
    XH2RT_STATUS_OK = 0,
    XH2RT_STATUS_INVALID_ARGUMENT,
    XH2RT_STATUS_INVALID_STATE,
    XH2RT_STATUS_NO_MEMORY,
    XH2RT_STATUS_OVERFLOW,
    XH2RT_STATUS_OUT_OF_RANGE,
    XH2RT_STATUS_MISALIGNED,
    XH2RT_STATUS_STALE_BUFFER,
    XH2RT_STATUS_BUSY,
    XH2RT_STATUS_HAL_FAILURE,
    XH2RT_STATUS_HAL_PROTOCOL,
    XH2RT_STATUS_POISONED,
    XH2RT_STATUS_EXECUTION_FAILURE
};

/*
 * Every operation returns its complete diagnostic by value.  The context also
 * retains the most recent failure so a higher layer can report it after its
 * own cleanup without depending on errno's lifetime.
 */
typedef struct xh2rt_result {
    enum xh2rt_status status;
    const char *operation;
    int raw_rc;
    int saved_errno;
    int64_t group_id;
    uint64_t group_state;
    uint64_t group_flags;
    uint32_t group_kernel_count;
    uint32_t group_core_count;
    uint32_t group_core_mask;
    uint32_t execution_result;
} xh2rt_result;

int xh2rt_result_is_ok(xh2rt_result result);
const char *xh2rt_status_string(enum xh2rt_status status);

/*
 * Only logical device 0 is accepted by the v1 runtime. Open uploads the built-in
 * kernels; close releases them along with the context's other buffers.
 * A failed open normally leaves *out_context NULL. If cleanup cannot release
 * everything, it returns the primary error with a non-NULL close-only context.
 * Retry close for cleanup errors; failed transfers quarantine their BOs, which
 * cannot be reclaimed without proof that the device has stopped using them.
 */
xh2rt_result xh2rt_context_open(uint32_t logical_device,
                                xh2rt_context **out_context);
xh2rt_result xh2rt_context_close(xh2rt_context **context);
xh2rt_result xh2rt_context_last_error(const xh2rt_context *context);

/*
 * Commands are blocking groups, not asynchronous streams.  A typed launch
 * outside an explicit batch is a one-shot operation and is complete when it
 * returns.  flush drains the current group while keeping the batch active;
 * end and synchronize drain it and leave batching inactive.
 */
xh2rt_result xh2rt_commands_begin(xh2rt_context *context);
xh2rt_result xh2rt_commands_flush(xh2rt_context *context);
xh2rt_result xh2rt_commands_end(xh2rt_context *context);
xh2rt_result xh2rt_synchronize(xh2rt_context *context);
int xh2rt_commands_active(const xh2rt_context *context);

xh2rt_result xh2rt_buffer_alloc(xh2rt_context *context, uint64_t bytes,
                                xh2rt_buffer_view **out_buffer);
xh2rt_result xh2rt_buffer_retain(xh2rt_context *context,
                                 xh2rt_buffer_view *buffer);
/* A failed final BO free keeps *buffer non-NULL so release can be retried. */
xh2rt_result xh2rt_buffer_release(xh2rt_context *context,
                                  xh2rt_buffer_view **buffer);
xh2rt_result xh2rt_buffer_view_create(xh2rt_context *context,
                                      xh2rt_buffer_view *buffer,
                                      uint64_t offset, uint64_t bytes,
                                      xh2rt_buffer_view **out_view);
xh2rt_result xh2rt_buffer_bytes(xh2rt_context *context,
                                const xh2rt_buffer_view *buffer,
                                uint64_t *out_bytes);

xh2rt_result xh2rt_buffer_write(xh2rt_context *context,
                                xh2rt_buffer_view *destination,
                                uint64_t destination_offset,
                                const void *source, uint64_t bytes);
xh2rt_result xh2rt_buffer_read(xh2rt_context *context, void *destination,
                               xh2rt_buffer_view *source,
                               uint64_t source_offset, uint64_t bytes);
xh2rt_result xh2rt_buffer_copy(xh2rt_context *context,
                               xh2rt_buffer_view *destination,
                               uint64_t destination_offset,
                               xh2rt_buffer_view *source,
                               uint64_t source_offset, uint64_t bytes);

/*
 * Transfers never use these values.  They are checked kernel-visible IOMAP
 * tokens and must never be dereferenced by the host.
 */
xh2rt_result xh2rt_buffer_iomap(xh2rt_context *context,
                                const xh2rt_buffer_view *buffer,
                                uint64_t offset, uint64_t bytes,
                                uint64_t alignment, uintptr_t *out_iomap);
xh2rt_result xh2rt_buffer_validate_iomap(
    xh2rt_context *context, const xh2rt_buffer_view *buffer,
    uintptr_t iomap, uint64_t bytes, uint64_t alignment,
    uint64_t *out_offset);

/*
 * These wrappers launch the built-in fill_f32.hdpl and add_f32.hdpl kernels,
 * using the SDK's native uint64_t arguments and 2-core x 4-tile execution.
 * Nonempty launches require count * 4 bytes in each view and 4-byte aligned
 * addresses. add accepts at most UINT32_MAX elements. A zero count validates
 * the handles but does not submit or drain work. value_bits is the float's
 * unchanged uint32_t representation, widened to a uint64_t device argument.
 */
xh2rt_result xh2rt_fill_f32(xh2rt_context *context,
                             xh2rt_buffer_view *destination,
                             uint32_t value_bits, uint64_t count);
xh2rt_result xh2rt_add_f32(xh2rt_context *context,
                            xh2rt_buffer_view *lhs,
                            xh2rt_buffer_view *rhs,
                            xh2rt_buffer_view *destination,
                            uint64_t count);

#endif
