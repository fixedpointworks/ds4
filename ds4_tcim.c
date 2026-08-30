/* TCIM/XH2 adapter link contract.
 *
 * Step 1 deliberately has no operator runtime yet.  The support manifest
 * below gives every symbol used by the shared graph its exact C signature and
 * an explicit fail-closed or established-fallback result.  Implemented
 * entries move out of the manifest as the XH2 runtime grows; there is never a
 * weak, variadic, or catch-all success stub.
 */

#ifndef DS4_TCIM_BUILD
#error "ds4_tcim.c is only for the DS4_TCIM_BUILD target"
#endif

#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory_allocator.h>

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {{0}};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

void ds4_gpu_cleanup(void) {
    memset(g_gpu, 0, sizeof(g_gpu));
    memset(g_gpu_peer_ok, 0, sizeof(g_gpu_peer_ok));
    g_gpu_peer_ok[0][0] = 1;
    g_n_gpus = 1;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    /* Every Step 1 allocator fails before producing a tensor. Seeing one here
     * means a required allocation was incorrectly reported as successful. */
    fputs("ds4: TCIM received a tensor without an allocator implementation\n",
          stderr);
    abort();
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

/* A REQUIRED entry is part of the single-device production path and cannot
 * claim success before its XH2 implementation exists. OPTIONAL entries use a
 * fallback already defined by the shared graph. METAL_PRIVATE entries close
 * ds4.c's local cross-TU declarations without making them public TCIM hooks.
 * UNAVAILABLE entries exist only for excluded Step 1 features. */
#define DS4_TCIM_REQUIRED(signature, failure) \
    signature { errno = ENOSYS; return (failure); }
#define DS4_TCIM_OPTIONAL(signature, fallback) \
    signature { return (fallback); }
#define DS4_TCIM_OPTIONAL_VOID(signature) \
    signature { }
#define DS4_TCIM_METAL_PRIVATE(signature, failure) \
    signature { errno = ENOSYS; return (failure); }
#define DS4_TCIM_METAL_PRIVATE_VOID(signature) \
    signature { }
#define DS4_TCIM_UNAVAILABLE(signature, failure) \
    signature { errno = ENOTSUP; return (failure); }

#include "tcim/ds4_tcim_support.def"

#undef DS4_TCIM_REQUIRED
#undef DS4_TCIM_OPTIONAL
#undef DS4_TCIM_OPTIONAL_VOID
#undef DS4_TCIM_METAL_PRIVATE
#undef DS4_TCIM_METAL_PRIVATE_VOID
#undef DS4_TCIM_UNAVAILABLE
