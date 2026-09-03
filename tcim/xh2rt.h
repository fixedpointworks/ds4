#ifndef DS4_TCIM_XH2RT_H
#define DS4_TCIM_XH2RT_H

#include <stdint.h>

typedef struct xh2rt_context xh2rt_context;
typedef struct xh2rt_buffer_view xh2rt_buffer_view;

/* Fallible operations return nonzero on success. On failure they return 0,
 * report the runtime-owned diagnostic and set errno for the business layer.
 * The two state predicates below return only their current boolean value.
 * A context is thread-confined: callers must serialize every operation on it. */

/*
 * Only logical device 0 is accepted by the v1 runtime. Open uploads the built-in
 * kernels; close releases them along with the context's other buffers.
 * A failed open normally leaves *out_context NULL. If cleanup cannot release
 * everything, it leaves a non-NULL close-only context and reports the primary
 * error.
 * Retry close for cleanup errors; failed transfers quarantine their BOs, which
 * cannot be reclaimed without proof that the device has stopped using them.
 */
int xh2rt_context_open(uint32_t logical_device, xh2rt_context **out_context);
int xh2rt_context_close(xh2rt_context **context);
/* Host-only query for adapter init/device shims; recoverable preflight errors
 * do not make the context unhealthy. */
int xh2rt_context_is_healthy(const xh2rt_context *context);

/*
 * Commands are blocking groups, not asynchronous streams.  A typed launch
 * outside an explicit batch is a one-shot operation and is complete when it
 * returns, including checking all 32 hart outcomes. flush drains the
 * current group while keeping the batch active;
 * end and synchronize drain it and leave batching inactive.
 */
int xh2rt_commands_begin(xh2rt_context *context);
int xh2rt_commands_flush(xh2rt_context *context);
int xh2rt_commands_end(xh2rt_context *context);
int xh2rt_synchronize(xh2rt_context *context);
int xh2rt_commands_active(const xh2rt_context *context);

int xh2rt_buffer_alloc(xh2rt_context *context, uint64_t bytes, xh2rt_buffer_view **out_buffer);
int xh2rt_buffer_retain(xh2rt_context *context, xh2rt_buffer_view *buffer);
/* A failed final BO free keeps *buffer non-NULL so release can be retried. */
int xh2rt_buffer_release(xh2rt_context *context, xh2rt_buffer_view **buffer);
int xh2rt_buffer_view_create(xh2rt_context *context, xh2rt_buffer_view *buffer, uint64_t offset, uint64_t bytes, xh2rt_buffer_view **out_view);
int xh2rt_buffer_bytes(xh2rt_context *context, const xh2rt_buffer_view *buffer, uint64_t *out_bytes);

int xh2rt_buffer_write(xh2rt_context *context, xh2rt_buffer_view *destination, uint64_t destination_offset, const void *source, uint64_t bytes);
int xh2rt_buffer_read(xh2rt_context *context, void *destination, xh2rt_buffer_view *source, uint64_t source_offset, uint64_t bytes);
int xh2rt_buffer_copy(xh2rt_context *context, xh2rt_buffer_view *destination, uint64_t destination_offset, xh2rt_buffer_view *source, uint64_t source_offset, uint64_t bytes);
/* Returns a checked device token; the host must never dereference it. */
int xh2rt_buffer_iomap(xh2rt_context *context, const xh2rt_buffer_view *buffer, uint64_t offset, uint64_t bytes, uint64_t alignment, uintptr_t *out_iomap);

/* To add a kernel:
 * 1. Add <name>.hdpl with a typed host wrapper and device implementation.
 * 2. Include <name>.hex and add its catalog row in xh2rt.c.
 * 3. Declare the typed wrapper below.
 * The file stem, wrapper lookup name, device entry, xxd symbol and catalog
 * name must match. */
int xh2rt_fill_f32(xh2rt_context *context, xh2rt_buffer_view *destination, uint32_t value_bits, uint64_t count);
int xh2rt_add_f32(xh2rt_context *context, xh2rt_buffer_view *lhs, xh2rt_buffer_view *rhs, xh2rt_buffer_view *destination, uint64_t count);

#endif
