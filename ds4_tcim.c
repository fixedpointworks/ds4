/* Basic TCIM tensor/command adapter. XH2 owns device memory and command
 * lifetime; this file only binds DS4 wrappers to opaque runtime views. */
#ifndef DS4_TCIM_BUILD
#error "ds4_tcim.c is only for the DS4_TCIM_BUILD target"
#endif

#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"
#include "tcim/xh2rt.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory_allocator.h>

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {{0}};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

typedef struct tcim_tensor {
    ds4_gpu_tensor *tensor;
    xh2rt_buffer_view *view;
    int heap_owned;
    struct tcim_tensor *next;
} tcim_tensor;

static xh2rt_context *tcim_context;
static tcim_tensor *tcim_tensors;
static int tcim_initialized;

static int tcim_result_ok(xh2rt_result result) {
    if (xh2rt_result_is_ok(result)) return 1;
    int error = result.saved_errno;
    if (error == 0) {
        switch (result.status) {
        case XH2RT_STATUS_NO_MEMORY: error = ENOMEM; break;
        case XH2RT_STATUS_OVERFLOW: error = EOVERFLOW; break;
        case XH2RT_STATUS_BUSY: error = EBUSY; break;
        case XH2RT_STATUS_INVALID_ARGUMENT:
        case XH2RT_STATUS_INVALID_STATE:
        case XH2RT_STATUS_OUT_OF_RANGE:
        case XH2RT_STATUS_MISALIGNED:
        case XH2RT_STATUS_STALE_BUFFER: error = EINVAL; break;
        default: error = EIO; break;
        }
    }
    fprintf(stderr, "ds4: TCIM %s: %s (rc=%d errno=%d",
            result.operation, xh2rt_status_string(result.status),
            result.raw_rc, result.saved_errno);
    if (result.group_id >= 0)
        fprintf(stderr, " group=%" PRId64 " hal_sync_result=0x%08" PRIx32,
                result.group_id, result.execution_result);
    if (result.kernel_index != UINT32_MAX)
        fprintf(stderr, " kernel=%" PRIu32 " stripe=%" PRIu32
                " status=0x%08" PRIx32, result.kernel_index,
                result.kernel_stripe, result.kernel_status);
    fputs(")\n", stderr);
    errno = error;
    return 0;
}

static int tcim_ready(void) {
    if (tcim_initialized && xh2rt_context_is_healthy(tcim_context)) return 1;
    errno = tcim_context == NULL ? ENODEV : EIO;
    return 0;
}

static tcim_tensor *tcim_find_tensor(const ds4_gpu_tensor *tensor) {
    tcim_tensor *entry;
    for (entry = tcim_tensors; entry != NULL; entry = entry->next)
        if (entry->tensor == tensor) return entry;
    return NULL;
}

/* Tensor fields carry no ownership authority. Validate their token/extent
 * against the registered opaque view before the runtime can touch hardware. */
static xh2rt_buffer_view *tcim_tensor_view(const ds4_gpu_tensor *tensor) {
    tcim_tensor *entry;
    uintptr_t iomap;
    uint64_t bytes;
    if (!tcim_ready()) return NULL;
    entry = tcim_find_tensor(tensor);
    if (entry == NULL || entry->view == NULL) {
        errno = EINVAL;
        return NULL;
    }
    if (!tcim_result_ok(xh2rt_buffer_bytes(tcim_context, entry->view, &bytes)) ||
        !tcim_result_ok(xh2rt_buffer_iomap(tcim_context, entry->view,
                                          0, bytes, 1, &iomap))) return NULL;
    if ((uintptr_t)tensor->ptr != iomap || tensor->bytes != bytes ||
        tensor->device_id != 0) {
        errno = EINVAL;
        return NULL;
    }
    return entry->view;
}

int ds4_gpu_init_multi(const ds4_gpu_config *config) {
    if (config == NULL || config->n_gpus != 1 ||
        config->device_indices[0] != 0) {
        errno = EINVAL;
        return 0;
    }
    if (tcim_context != NULL) {
        if (tcim_ready()) return 1;
        /* A close-only/quarantined context must not be bypassed by opening
         * another allocator over device memory whose lifetime is unknown. */
        errno = EBUSY;
        return 0;
    }
    if (!tcim_result_ok(xh2rt_context_open(0, &tcim_context))) return 0;
    memset(g_gpu, 0, sizeof(g_gpu));
    memset(g_gpu_peer_ok, 0, sizeof(g_gpu_peer_ok));
    g_gpu[0].device_id = 0;
    g_gpu[0].budget_bytes = config->vram_bytes[0];
    /* used_bytes is a compatibility placeholder: physical BO ownership lives
     * in xh2rt, so freeing a base wrapper cannot undercount a surviving view. */
    g_n_gpus = 1;
    g_gpu_peer_ok[0][0] = 1;
    tcim_initialized = 1;
    return 1;
}

int ds4_gpu_init(void) {
    ds4_gpu_config config = {0};
    config.n_gpus = 1;
    return ds4_gpu_init_multi(&config);
}

void ds4_gpu_cleanup(void) {
    tcim_tensor *entry;
    const int primary_errno = errno;
    tcim_initialized = 0;
    (void)tcim_result_ok(xh2rt_context_close(&tcim_context));
    /* Callers still own their wrapper storage. Keep only that association so
     * free after cleanup is safe; no stale view can enter a later context. */
    for (entry = tcim_tensors; entry != NULL; entry = entry->next)
        entry->view = NULL;
    memset(g_gpu, 0, sizeof(g_gpu));
    memset(g_gpu_peer_ok, 0, sizeof(g_gpu_peer_ok));
    g_n_gpus = 1;
    g_gpu_peer_ok[0][0] = 1;
    if (primary_errno != 0) errno = primary_errno;
}

/* Allocate host bookkeeping first, so failure cannot strand a new device
 * view outside its context's ownership. Zero-size tensors follow CUDA's ABI. */
int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *tensor, int device_id,
                             uint64_t bytes) {
    tcim_tensor *entry;
    uintptr_t iomap;
    if (tensor == NULL || device_id != 0 || tcim_find_tensor(tensor) != NULL) {
        errno = EINVAL;
        return 1;
    }
    if (!tcim_ready()) return 1;
    if (bytes == 0) bytes = 1;
    entry = calloc(1, sizeof(*entry));
    if (entry == NULL) return 1;
    if (!tcim_result_ok(xh2rt_buffer_alloc(tcim_context, bytes, &entry->view))) {
        free(entry);
        return 1;
    }
    if (!tcim_result_ok(xh2rt_buffer_iomap(tcim_context, entry->view,
                                          0, bytes, 1, &iomap))) {
        int error = errno;
        (void)xh2rt_buffer_release(tcim_context, &entry->view);
        free(entry);
        errno = error;
        return 1;
    }
    *tensor = (ds4_gpu_tensor){(void *)(uintptr_t)iomap, bytes, 1, 0};
    entry->tensor = tensor;
    entry->next = tcim_tensors;
    tcim_tensors = entry;
    return 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    ds4_gpu_tensor *tensor;
    if (tier != 0) { errno = EINVAL; return NULL; }
    tensor = calloc(1, sizeof(*tensor));
    if (tensor == NULL) return NULL;
    if (ds4_gpu_tensor_alloc_on(tensor, tier, bytes) != 0) {
        free(tensor);
        return NULL;
    }
    tcim_find_tensor(tensor)->heap_owned = 1;
    return tensor;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    return ds4_gpu_tensor_alloc_ptr_on(0, bytes);
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base,
                                   uint64_t offset, uint64_t bytes) {
    xh2rt_buffer_view *base_view = tcim_tensor_view(base);
    tcim_tensor *entry;
    ds4_gpu_tensor *tensor;
    uintptr_t iomap;
    if (base_view == NULL) return NULL;
    entry = calloc(1, sizeof(*entry));
    tensor = calloc(1, sizeof(*tensor));
    if (entry == NULL || tensor == NULL) {
        free(entry);
        free(tensor);
        return NULL;
    }
    if (!tcim_result_ok(xh2rt_buffer_view_create(tcim_context, base_view,
                                                 offset, bytes, &entry->view)))
        goto fail;
    if (!tcim_result_ok(xh2rt_buffer_iomap(tcim_context, entry->view,
                                          0, bytes, 1, &iomap))) goto fail;
    *tensor = (ds4_gpu_tensor){(void *)(uintptr_t)iomap, bytes, 0, 0};
    entry->tensor = tensor;
    entry->heap_owned = 1;
    entry->next = tcim_tensors;
    tcim_tensors = entry;
    return tensor;

fail:
    {
        int error = errno;
        (void)xh2rt_buffer_release(tcim_context, &entry->view);
        free(entry);
        free(tensor);
        errno = error;
        return NULL;
    }
}

static void tcim_tensor_release(ds4_gpu_tensor *tensor, int free_wrapper) {
    tcim_tensor **link = &tcim_tensors;
    tcim_tensor *entry;
    const int primary_errno = errno;
    if (tensor == NULL) return;
    while (*link != NULL && (*link)->tensor != tensor) link = &(*link)->next;
    entry = *link;
    if (entry == NULL) {
        errno = primary_errno != 0 ? primary_errno : EINVAL;
        return;
    }
    /* Even when a void free cannot report failure, the runtime retains any
     * failed physical free in its registry for cleanup to retry safely. */
    if (entry->view != NULL)
        (void)tcim_result_ok(xh2rt_buffer_release(tcim_context, &entry->view));
    memset(tensor, 0, sizeof(*tensor));
    tensor->device_id = -1;
    if (!free_wrapper && entry->heap_owned) {
        entry->view = NULL;
        if (primary_errno != 0) errno = primary_errno;
        return;
    }
    *link = entry->next;
    if (free_wrapper && entry->heap_owned) free(tensor);
    free(entry);
    if (primary_errno != 0) errno = primary_errno;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    tcim_tensor_release(tensor, 1);
}

void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *tensor) {
    tcim_tensor_release(tensor, 0);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tcim_tensor_view(tensor) != NULL ? tensor->bytes : 0;
}

int ds4_gpu_tensor_device(const ds4_gpu_tensor *tensor) {
    return tcim_tensor_view(tensor) != NULL ? 0 : -1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset,
                           const void *data, uint64_t bytes) {
    xh2rt_buffer_view *view = tcim_tensor_view(tensor);
    if (view == NULL) return 0;
    return tcim_result_ok(xh2rt_buffer_write(tcim_context, view, offset,
                                             data, bytes));
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                          void *data, uint64_t bytes) {
    xh2rt_buffer_view *view = tcim_tensor_view(tensor);
    if (view == NULL) return 0;
    return tcim_result_ok(xh2rt_buffer_read(tcim_context, data, view,
                                            offset, bytes));
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    xh2rt_buffer_view *dst_view = tcim_tensor_view(dst);
    xh2rt_buffer_view *src_view = tcim_tensor_view(src);
    if (dst_view == NULL || src_view == NULL) return 0;
    return tcim_result_ok(xh2rt_buffer_copy(tcim_context, dst_view, dst_offset,
                                            src_view, src_offset, bytes));
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                             uint64_t count) {
    xh2rt_buffer_view *view = tcim_tensor_view(tensor);
    uint32_t value_bits;
    if (view == NULL) return 0;
    memcpy(&value_bits, &value, sizeof(value_bits));
    return tcim_result_ok(xh2rt_fill_f32(tcim_context, view, value_bits, count));
}

int ds4_gpu_begin_commands(void) {
    return tcim_ready() && tcim_result_ok(xh2rt_commands_begin(tcim_context));
}

int ds4_gpu_commands_active(void) {
    return tcim_ready() && xh2rt_commands_active(tcim_context);
}

int ds4_gpu_flush_commands(void) {
    return tcim_ready() && tcim_result_ok(xh2rt_commands_flush(tcim_context));
}

int ds4_gpu_flush_encoder(void) {
    return ds4_gpu_flush_commands();
}

int ds4_gpu_end_commands(void) {
    return tcim_ready() && tcim_result_ok(xh2rt_commands_end(tcim_context));
}

int ds4_gpu_synchronize(void) {
    return tcim_ready() && tcim_result_ok(xh2rt_synchronize(tcim_context));
}

int ds4_gpu_set_current_device(int tier) {
    if (tier != 0) { errno = EINVAL; return -1; }
    return tcim_ready() ? 0 : -1;
}

int ds4_gpu_set_current_device_fenced(int tier) {
    /* There is only one device, so no cross-device handoff is necessary. */
    return ds4_gpu_set_current_device(tier);
}

static void tcim_probe_error(char *errbuf, size_t errbuflen,
                             const char *fmt, ...) {
    if (!errbuf || errbuflen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(errbuf, errbuflen, fmt, ap);
    va_end(ap);
}

int ds4_gpu_args_probe_auto_cuda(const int      *device_filter,
                                  int             filter_len,
                                  ds4_gpu_config *out,
                                  size_t          safety_margin_bytes,
                                  char           *errbuf,
                                  size_t          errbuflen) {
    if (errbuf && errbuflen != 0) errbuf[0] = '\0';
    if (!out) {
        tcim_probe_error(errbuf, errbuflen,
                         "internal: NULL TCIM auto-probe output");
        return 1;
    }
    memset(out, 0, sizeof(*out));

    const int no_filter = device_filter == NULL && filter_len == 0;
    const int device_zero = device_filter != NULL && filter_len == 1 &&
                            device_filter[0] == 0;
    if (!no_filter && !device_zero) {
        tcim_probe_error(errbuf, errbuflen,
                         "TCIM supports only logical device 0");
        return 1;
    }

    fd_handle_t allocator_fd = INVALID_FD_HANDLE_VAL;
    errno = 0;
    const int open_rc = xh2a_memory_allocator_open(
            0, XH2A_MEMORY_ALLOCATOR_MEMPOOL_DDR, &allocator_fd);
    const int open_errno = errno;
    if (open_rc != 0) {
        tcim_probe_error(errbuf, errbuflen,
                         "TCIM logical device 0 DDR allocator open failed: "
                         "rc=%d errno=%d (%s)",
                         open_rc, open_errno, strerror(open_errno));
        return 1;
    }

    uint64_t start = 0;
    uint64_t total_size = 0;
    uint64_t free_size = 0;
    uint64_t max_free_buffer_size = 0;
    errno = 0;
    const int info_rc = xh2a_memory_allocator_get_mem_info(
            allocator_fd, &start, &total_size, &free_size,
            &max_free_buffer_size);
    const int info_errno = errno;

    errno = 0;
    const int close_rc = xh2a_memory_allocator_close(allocator_fd);
    const int close_errno = errno;

    if (info_rc != 0) {
        tcim_probe_error(errbuf, errbuflen,
                         "TCIM logical device 0 DDR memory query failed: "
                         "rc=%d errno=%d (%s)",
                         info_rc, info_errno, strerror(info_errno));
        return 1;
    }
    if (close_rc != 0) {
        tcim_probe_error(errbuf, errbuflen,
                         "TCIM logical device 0 DDR allocator close failed: "
                         "rc=%d errno=%d (%s)",
                         close_rc, close_errno, strerror(close_errno));
        return 1;
    }
    if (free_size > total_size || free_size > SIZE_MAX) {
        tcim_probe_error(errbuf, errbuflen,
                         "TCIM logical device 0 returned invalid DDR memory "
                         "information");
        return 1;
    }

    /* Match the shared CUDA/ROCm auto-budget policy.  The engine applies its
     * configured safety margin separately on top of this auto reserve. */
    const uint64_t reserve_floor = UINT64_C(2) * 1024u * 1024u * 1024u;
    const uint64_t reserve_percent = free_size / 20u;
    const uint64_t reserve = reserve_floor > reserve_percent
                           ? reserve_floor : reserve_percent;
    const uint64_t budget = free_size > reserve ? free_size - reserve : 0;

    (void)start;
    (void)max_free_buffer_size;
    out->device_indices[0] = 0;
    out->vram_bytes[0] = (size_t)budget;
    out->n_gpus = 1;
    out->safety_margin_bytes = safety_margin_bytes;
    return 0;
}

/* Remaining interfaces stay explicit and fail closed until their TCIM
 * implementations replace the corresponding marked function below. */
/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_add3_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_add_rms_norm_weight_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const void * arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7, float arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_add_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, uint32_t arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_argmax_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_decode_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, const ds4_gpu_tensor * arg9, uint32_t arg10, uint32_t arg11, const ds4_gpu_tensor * arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_decode_mixed_batch_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, const ds4_gpu_tensor * arg6, uint32_t arg7, const ds4_gpu_tensor * arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, uint32_t arg19)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_decode_raw_batch_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_decode_rows_rope_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_attention_decode_row * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, float arg11, float arg12, float arg13, float arg14, float arg15, float arg16)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, const ds4_gpu_tensor * arg6, uint32_t arg7, const ds4_gpu_tensor * arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, uint32_t arg19)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_output_low_q4_K_slice_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, uint32_t arg7, const ds4_gpu_tensor * arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_output_low_q8_rows_exact_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, const ds4_gpu_tensor * arg9, uint32_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, const ds4_gpu_tensor * arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_output_q4_K_batch_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, const void * arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint64_t arg9, uint64_t arg10, uint32_t arg11, uint64_t arg12, const ds4_gpu_tensor * arg13, uint32_t arg14)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_output_q8_batch_f16_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint64_t arg9, const ds4_gpu_tensor * arg10, uint32_t arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, const void * arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint64_t arg11, const ds4_gpu_tensor * arg12, uint32_t arg13)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_attention_output_q8_tp_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint64_t arg11, const ds4_gpu_tensor * arg12)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_prefill_raw_heads_range_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, const ds4_gpu_tensor * arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attention_prefill_static_mixed_heads_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, const ds4_gpu_tensor * arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, bool arg14, float arg15, float arg16, float arg17, float arg18, float arg19, float arg20, float arg21)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_build_derived_artifacts(const void * arg0, uint64_t arg1, const char * arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_cache_model_range(const void * arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3, const char * arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_cache_q8_f16_range(const void * arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const char * arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_commit_and_wait_selected_readback(uint64_t arg0, const char * arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_compressor_prefill_ratio4_replay_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, bool arg16, float arg17, float arg18, float arg19, float arg20, float arg21, float arg22, float arg23)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_compressor_prefill_state_ratio4_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const void * arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_compressor_prefill_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, uint32_t arg16, bool arg17, float arg18, float arg19, float arg20, float arg21, float arg22, float arg23, float arg24)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_compressor_update_tensor(const ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, uint32_t arg16, float arg17, float arg18, float arg19, float arg20, float arg21, float arg22, float arg23, bool arg24, bool arg25, bool arg26)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: No capture can begin: graph begin always returns -1 (eager); there is no graph resource to abort. */
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key * arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key * arg0)
{
    (void)arg0;
    return -1;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_decode_graph_end(const ds4_decode_graph_key * arg0)
{
    (void)arg0;
    return -1;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Graph support is always false and begin never captures; there are no graphs to invalidate. */
void ds4_gpu_decode_graphs_invalidate(void)
{
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_decode_graphs_supported(void)
{
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_device_cache_support_tensors(int arg0, int arg1, const ds4_tensor_range * arg2, int arg3, int arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 1;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_device_cache_tensors(int arg0, const ds4_tensor_range * arg1, int arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 1;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_directional_steering_project_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, float arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, ds4_gpu_tensor * arg6, const ds4_gpu_tensor * arg7, uint64_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, bool arg16, float arg17, float arg18, float arg19, float arg20, float arg21, float arg22, float arg23)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, ds4_gpu_tensor * arg6, const ds4_gpu_tensor * arg7, uint64_t arg8, uint32_t arg9, uint32_t arg10, float arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_embed_token_q8_0_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_embed_token_quant_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_embed_tokens_hc_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_embed_tokens_q8_0_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_embed_tokens_quant_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: No dequant/GEMM optimization is implemented; its operator APIs fail closed. */
void ds4_gpu_enable_q8_dequant_gemm(void)
{
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_embedding_bf16(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_expand_pool_selection_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_indexer_pool_update_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, float arg15, bool arg16)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_indexer_scores_batch_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, float arg10, bool arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_kda_decode(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, const ds4_gpu_tensor * arg6, const ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, const void * arg9, uint64_t arg10, uint64_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint64_t arg16, uint32_t arg17, uint32_t arg18, float arg19, float arg20)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_kda_prefill(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, ds4_gpu_tensor * arg4, ds4_gpu_tensor * arg5, ds4_gpu_tensor * arg6, const ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, const void * arg9, uint64_t arg10, uint64_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint64_t arg16, uint32_t arg17, uint32_t arg18, float arg19, float arg20)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_matmul_bf16(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint32_t arg5, const ds4_gpu_tensor * arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_scatter_image_hc(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm53_vision_encode(float * arg0, const float * arg1, uint32_t arg2, uint32_t arg3, const void * arg4, uint64_t arg5, const ds4_glm53_vision_weights * arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, bool arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_flash_staged_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, bool arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_flash_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, bool arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_full_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, bool arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, bool arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, float arg15, float arg16, float arg17, float arg18, float arg19, float arg20)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, bool arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, float arg15, float arg16, float arg17, float arg18, float arg19, float arg20)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_indexed_batch_typed_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, const ds4_gpu_tensor * arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, bool arg13, uint32_t arg14, uint32_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, uint32_t arg19, float arg20, float arg21, float arg22, float arg23, float arg24, float arg25)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_attention_indexed_decode_typed_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, const ds4_gpu_tensor * arg9, uint32_t arg10, uint32_t arg11, bool arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, float arg19, float arg20, float arg21, float arg22, float arg23, float arg24)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_build_kv_cache_flash_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, float arg15, float arg16, float arg17, float arg18, float arg19, float arg20, bool arg21)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_build_kv_cache_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, float arg15, float arg16, float arg17, float arg18, float arg19, float arg20, bool arg21)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_fill_selected_range_tensor(ds4_gpu_tensor * arg0, uint32_t arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_indexer_rope_tail_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, float arg7, float arg8, float arg9, float arg10, float arg11, float arg12)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_indexer_score_one_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, float arg7, bool arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_indexer_scores_batch_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, float arg9, bool arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_k_b_project_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_k_b_project_typed_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_kv_lora_rms_norm_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, float arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_qk_lowrank_typed_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, ds4_gpu_tensor * arg6, ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, uint32_t arg13, uint32_t arg14, uint32_t arg15, bool arg16, float arg17)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_rope_tail_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, float arg7, float arg8, float arg9, float arg10, float arg11, float arg12)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_routed_moe_batch_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint64_t arg10, uint64_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, const ds4_gpu_tensor * arg19, const ds4_gpu_tensor * arg20, uint32_t arg21, uint32_t arg22, float arg23, uint32_t arg24, const ds4_gpu_tensor * arg25, uint32_t arg26, uint32_t arg27, bool arg28)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    (void)arg27;
    (void)arg28;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_routed_moe_one_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint64_t arg10, uint64_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, const ds4_gpu_tensor * arg19, const ds4_gpu_tensor * arg20, uint32_t arg21, uint32_t arg22, float arg23, uint32_t arg24, const ds4_gpu_tensor * arg25, bool arg26)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_router_select_batch_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint32_t arg7, uint32_t arg8, float arg9, uint32_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_router_select_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint32_t arg7, uint32_t arg8, float arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_store_compact_kv_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, bool arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_store_indexer_k_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, uint32_t arg11, float arg12, float arg13, float arg14, float arg15, float arg16, float arg17, float arg18, bool arg19)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(const ds4_gpu_stream_expert_table * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: GLM model operators are outside this DeepSeek V4 TCIM backend. */
int ds4_gpu_glm_value_project_typed_batch_heads_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_expand_add_split_half_add_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_expand_add_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const ds4_gpu_tensor * arg5, uint32_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_expand_split_half_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint32_t arg5, uint32_t arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, uint32_t arg7, float arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_split_weighted_sum_norm_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint32_t arg12, float arg13, float arg14)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_split_weighted_sum_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, const void * arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, float arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, uint32_t arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, uint32_t arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, bool arg7, float arg8, float arg9, float arg10, float arg11, float arg12, float arg13, float arg14)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3, float arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_indexer_score_one_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, float arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_indexer_scores_decode_batch_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, float arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_indexer_scores_prefill_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, uint32_t arg7, uint32_t arg8, float arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_indexer_top1_value_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor *const * arg1, const uint32_t * arg2, const uint32_t * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, const void * arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint32_t arg9, uint64_t arg10, uint32_t arg11, const ds4_gpu_tensor * arg12, uint32_t arg13, uint32_t arg14)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_f16_pair_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, const ds4_gpu_tensor * arg8, uint64_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_f16_rms_fold_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint64_t arg7, float arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_f16_router_rows_exact_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, const ds4_gpu_tensor * arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint64_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint64_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_matmul_q4_K_pair_decode_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, const ds4_gpu_tensor * arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_f16_out_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint64_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_hc_expand_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, const ds4_gpu_tensor * arg9, uint32_t arg10, uint32_t arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, const ds4_gpu_tensor * arg10, const ds4_gpu_tensor * arg11, const ds4_gpu_tensor * arg12, uint32_t arg13, uint32_t arg14)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_matmul_q8_0_kslice_rows_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, const ds4_gpu_tensor * arg8, uint64_t arg9)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, uint32_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_pair_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, uint64_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, const ds4_gpu_tensor * arg6, uint64_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_q8_0_top1_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, uint32_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, uint64_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_matmul_quant_kslice_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, uint64_t arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_matmul_quant_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint32_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, uint64_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_model_range_replaced(const void * arg0, uint64_t arg1, uint64_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: No model residency is established by D4; all model registration/loading APIs fail closed until #7. */
void ds4_gpu_model_residency_skip(int arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_moe_handoff_pack_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_output_hc_weights_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint32_t arg6, float arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_pack_slot_rows_f32_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_preload_q4_expert_tables(const void * arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Optional diagnostic output only; omitting a report changes no computation or resource ownership. */
void ds4_gpu_print_memory_report(const char * arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_pro_q4_expert_table_auto_available(void)
{
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_q8_cache_suppressed(void)
{
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
uint64_t ds4_gpu_recommended_working_set_size(void)
{
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_register_model_map_no_copy(const void * arg0, uint64_t arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_register_support_map(const void * arg0, uint64_t arg1, uint64_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, float arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, float arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, uint32_t arg6, float arg7)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint32_t arg5, float arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_rope_tail_decode_rows_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_attention_decode_row * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, bool arg7, float arg8, float arg9, float arg10, float arg11, float arg12, float arg13)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor * arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6, bool arg7, float arg8, float arg9, float arg10, float arg11, float arg12, float arg13)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_batch_owned_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, ds4_gpu_tensor * arg19, ds4_gpu_tensor * arg20, uint32_t arg21, uint32_t arg22, uint32_t arg23, uint32_t arg24, float arg25, const ds4_gpu_tensor * arg26, uint32_t arg27, uint32_t arg28, bool * arg29)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    (void)arg27;
    (void)arg28;
    (void)arg29;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, const ds4_gpu_tensor * arg19, const ds4_gpu_tensor * arg20, uint32_t arg21, uint32_t arg22, float arg23, const ds4_gpu_tensor * arg24, uint32_t arg25, uint32_t arg26, bool * arg27, bool arg28)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    (void)arg27;
    (void)arg28;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_one_owned_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, const ds4_gpu_tensor * arg19, const ds4_gpu_tensor * arg20, uint32_t arg21, uint32_t arg22, uint32_t arg23, uint32_t arg24, float arg25, const ds4_gpu_tensor * arg26, ds4_gpu_tensor * arg27, bool arg28, ds4_gpu_tensor * arg29)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    (void)arg27;
    (void)arg28;
    (void)arg29;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, ds4_gpu_tensor * arg4, const void * arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint32_t arg10, uint32_t arg11, uint64_t arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15, uint32_t arg16, uint32_t arg17, uint32_t arg18, const ds4_gpu_tensor * arg19, const ds4_gpu_tensor * arg20, uint32_t arg21, uint32_t arg22, float arg23, const ds4_gpu_tensor * arg24, const ds4_gpu_tensor * arg25, uint32_t arg26, bool arg27)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    (void)arg18;
    (void)arg19;
    (void)arg20;
    (void)arg21;
    (void)arg22;
    (void)arg23;
    (void)arg24;
    (void)arg25;
    (void)arg26;
    (void)arg27;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_owned_packed_combine_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5, uint32_t arg6)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_owned_slots_combine_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, const ds4_gpu_tensor * arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_routed_moe_set_selected_override(const int32_t * arg0, uint32_t arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, bool arg10, bool arg11, const ds4_gpu_tensor * arg12, const ds4_gpu_tensor * arg13, uint32_t arg14, uint32_t arg15, float arg16, uint32_t arg17)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    (void)arg17;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_router_select_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint32_t arg7, uint32_t arg8, uint32_t arg9, uint32_t arg10, float arg11, uint32_t arg12, uint32_t arg13, bool arg14, bool arg15, const ds4_gpu_tensor * arg16)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    (void)arg16;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_set_aux_model_map_range(const void * arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_set_decode_fast_attention(int arg0)
{
    (void)arg0;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_set_decode_score_vec4(int arg0)
{
    (void)arg0;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: GLM kernels are unavailable; this tuning hint cannot enable them. */
void ds4_gpu_set_glm_model(bool arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: GLM streaming kernels are unavailable; this tuning hint cannot enable them. */
void ds4_gpu_set_glm_streaming_prefill_full_layer(bool arg0)
{
    (void)arg0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_set_model_fd(int arg0)
{
    (void)arg0;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_set_model_fd_for_map(int arg0, const void * arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_set_model_map_range(const void * arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_set_model_map_spans(const void * arg0, uint64_t arg1, const uint64_t * arg2, const uint64_t * arg3, uint32_t arg4, uint64_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: D4 has no Q8 weight cache; cache creation and dependent operators fail closed. */
void ds4_gpu_set_q8_cache_suppressed(int arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Only exact-bit fill is implemented; no approximate compute mode exists to configure. */
void ds4_gpu_set_quality(bool arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: D4 does not load model/expert storage; registration and load APIs fail closed until #7. */
void ds4_gpu_set_ssd_streaming(bool arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: D4 has no expert cache and all expert load APIs fail closed; budget cannot admit work. */
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: D4 has no expert cache and all expert load APIs fail closed; no allocation depends on this hint. */
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t arg0)
{
    (void)arg0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, const ds4_gpu_tensor * arg9, const ds4_gpu_tensor * arg10, const ds4_gpu_tensor * arg11, uint32_t arg12, uint32_t arg13)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, const ds4_gpu_tensor * arg9, const ds4_gpu_tensor * arg10, uint32_t arg11, const ds4_gpu_tensor * arg12, const ds4_gpu_tensor * arg13, uint32_t arg14, uint32_t arg15)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    (void)arg13;
    (void)arg14;
    (void)arg15;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_down_hc_expand_q8_0_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, const void * arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, const ds4_gpu_tensor * arg8, const ds4_gpu_tensor * arg9, const ds4_gpu_tensor * arg10, uint32_t arg11, uint32_t arg12)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, float arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, uint64_t arg10, float arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, uint64_t arg10, float arg11)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(ds4_gpu_tensor * arg0, ds4_gpu_tensor * arg1, ds4_gpu_tensor * arg2, const void * arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t arg8, const ds4_gpu_tensor * arg9, float arg10)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, float arg8, const ds4_gpu_tensor * arg9, const ds4_gpu_tensor * arg10, uint32_t arg11, bool arg12)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    (void)arg9;
    (void)arg10;
    (void)arg11;
    (void)arg12;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_shared_mid_swiglu_q8_0_tensor(ds4_gpu_tensor * arg0, const void * arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, const ds4_gpu_tensor * arg7, float arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
int ds4_gpu_should_use_managed_kv_cache(uint64_t arg0, uint64_t arg1)
{
    (void)arg0;
    (void)arg1;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_signal_selected_readback_ready(uint64_t * arg0)
{
    (void)arg0;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4, uint32_t arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint32_t arg2, uint32_t arg3, uint32_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_stream_expert_cache_begin_selected_load(const ds4_gpu_stream_expert_table * arg0, const int32_t * arg1, uint32_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(uint64_t arg0, uint64_t arg1)
{
    (void)arg0;
    (void)arg1;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
uint32_t ds4_gpu_stream_expert_cache_configured_count(void)
{
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: Existing graph capability probe or fused optimization fallback; caller retains its ordinary path. */
uint32_t ds4_gpu_stream_expert_cache_current_count(void)
{
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_stream_expert_cache_prepare_selected_batch(const ds4_gpu_stream_expert_table * arg0, const int32_t * arg1, uint32_t arg2, uint32_t arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: No resident expert cache or eviction heuristic exists before #7; no state needs resetting. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void)
{
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_stream_expert_cache_seed_experts(const ds4_gpu_stream_expert_table * arg0, const int32_t * arg1, const uint32_t * arg2, uint32_t arg3)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
int ds4_gpu_stream_expert_cache_seed_selected(const ds4_gpu_stream_expert_table * arg0, const int32_t * arg1, uint32_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, const ds4_gpu_tensor * arg2, uint32_t arg3, float arg4, float arg5)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: Managed memory is outside the explicit TCIM DDR BO ownership contract. */
ds4_gpu_tensor * ds4_gpu_tensor_alloc_managed(uint64_t arg0)
{
    (void)arg0;
    errno = ENOTSUP;
    return NULL;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: Managed memory is outside the explicit TCIM DDR BO ownership contract. */
ds4_gpu_tensor * ds4_gpu_tensor_alloc_managed_on(int arg0, uint64_t arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOTSUP;
    return NULL;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: The async-named compatibility API is not implemented by the blocking D4 adapter; fail closed. */
int ds4_gpu_tensor_copy_async(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint64_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint64_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint64_t arg2, ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint64_t arg5, ds4_gpu_tensor * arg6, const ds4_gpu_tensor * arg7, uint64_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_copy_xdev3_default_dst(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint64_t arg2, ds4_gpu_tensor * arg3, const ds4_gpu_tensor * arg4, uint64_t arg5, ds4_gpu_tensor * arg6, const ds4_gpu_tensor * arg7, uint64_t arg8)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    (void)arg7;
    (void)arg8;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint64_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor * arg0, const ds4_gpu_tensor * arg1, uint64_t arg2)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor * arg0, int arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor * arg0, int arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: Model/static-span storage, residency or transfer policy belongs to #7; fail closed until implemented. */
uint64_t ds4_gpu_tier_free_vram(int arg0)
{
    (void)arg0;
    errno = ENOSYS;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tp_batch_gate_encode(uint32_t arg0, uint32_t arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tp_big_gate_encode(uint32_t arg0, uint32_t arg1, const ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, uint64_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
uint64_t ds4_gpu_tp_big_gate_kick(uint32_t arg0, uint32_t arg1, const ds4_gpu_tensor * arg2, ds4_gpu_tensor * arg3, uint64_t arg4)
{
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tp_big_gate_wait(uint64_t arg0)
{
    (void)arg0;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OUT_OF_SCOPE_UNAVAILABLE: TP, cross-device handoff or sliced multi-device execution is outside the single-device TCIM scope. */
int ds4_gpu_tp_gate_encode(uint32_t arg0, uint32_t arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOTSUP;
    return 0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: TP is unavailable and no keepalive thread is started; there is nothing to pause. */
void ds4_gpu_tp_keepalive_pause(int arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: TP is unavailable; no split execution can be enabled by this hint. */
void ds4_gpu_tp_set_attn_head_split(int arg0)
{
    (void)arg0;
}

/* TCIM_STUB: OPTIONAL_FALLBACK: TP and sharded experts are unavailable; no ownership state exists to suspend. */
void ds4_gpu_tp_suspend_expert_sharding(int arg0)
{
    (void)arg0;
}

/* TCIM_STUB: PRODUCT_REQUIRED_TODO: DS4 compute/operator closure belongs to later Steps 2-3; native fill/add runtime tests do not implement this API. */
int ds4_gpu_wait_selected_readback_ready(uint64_t arg0, const char * arg1)
{
    (void)arg0;
    (void)arg1;
    errno = ENOSYS;
    return 0;
}
