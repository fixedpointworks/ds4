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

/* The generated inventory gives each unsupported symbol its exact public
 * signature and documented failure/fallback value. */
#include "tcim/ds4_tcim_stubs.inc"
