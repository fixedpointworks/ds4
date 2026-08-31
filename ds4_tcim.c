/* TCIM/XH2 adapter link contract.
 *
 * The XH2 runtime is not yet wired into the shared graph. The remaining
 * stubs below retain exact C signatures and explicit failure or established
 * fallback results. Replace each stub here as its XH2 implementation is added.
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
    /* Until tensor allocation is wired in, no tensor can be returned.
     * Seeing one here means allocation was incorrectly reported as successful. */
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
 * UNAVAILABLE entries cover unsupported features. */
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

/* REQUIRED: 162 entries. */
DS4_TCIM_REQUIRED(int ds4_gpu_add3_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_add_rms_norm_weight_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_add_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_argmax_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_decode_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_decode_mixed_batch_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_decode_raw_batch_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_decode_rows_rope_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_attention_decode_row *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_output_low_q4_K_slice_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, const ds4_gpu_tensor *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_output_low_q8_rows_exact_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, const ds4_gpu_tensor *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_output_q4_K_batch_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, uint32_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_output_q8_batch_f16_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_prefill_raw_heads_range_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attention_prefill_static_mixed_heads_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_begin_commands(void), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_cache_model_range(const void *, uint64_t, uint64_t, uint64_t, const char *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_cache_q8_f16_range(const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const char *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_commit_and_wait_selected_readback(uint64_t, const char *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_compressor_prefill_ratio4_replay_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_compressor_prefill_state_ratio4_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_compressor_prefill_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_compressor_update_tensor(const ds4_gpu_tensor *, const ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, float, _Bool, _Bool, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_directional_steering_project_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_embed_token_q8_0_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_embed_token_quant_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_embed_tokens_hc_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_embed_tokens_q8_0_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_embed_tokens_quant_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_end_commands(void), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_flush_commands(void), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_flush_encoder(void), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_flash_staged_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_flash_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_full_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, _Bool, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_indexed_batch_typed_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, _Bool, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_attention_indexed_decode_typed_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, _Bool, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_build_kv_cache_flash_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_build_kv_cache_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_fill_selected_range_batch_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_fill_selected_range_tensor(ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_indexer_rope_tail_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_indexer_score_one_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, float, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_indexer_scores_batch_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_k_b_project_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_k_b_project_typed_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_kv_lora_rms_norm_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_qk_lowrank_typed_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_rope_tail_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_routed_moe_batch_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_routed_moe_one_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_router_select_batch_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t, uint32_t, float, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_router_select_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_store_compact_kv_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_store_indexer_k_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, float, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_glm_value_project_typed_batch_heads_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_expand_add_split_half_add_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_expand_add_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_expand_split_half_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_split_weighted_sum_norm_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_split_weighted_sum_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_indexer_score_one_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_indexer_scores_decode_batch_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_indexer_scores_prefill_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_indexer_top1_value_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_init(void), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_init_multi(const ds4_gpu_config *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *const *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_f16_pair_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_f16_rms_fold_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_f16_router_rows_exact_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_f16_out_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_hc_expand_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_pair_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_q8_0_top1_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_matmul_quant_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_output_hc_weights_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_pack_slot_rows_f32_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_register_model_map_no_copy(const void *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_rope_tail_decode_rows_tensor(ds4_gpu_tensor *, const ds4_gpu_attention_decode_row *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_batch_owned_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, ds4_gpu_tensor *, ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, float, const ds4_gpu_tensor *, uint32_t, uint32_t, _Bool *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, float, const ds4_gpu_tensor *, uint32_t, uint32_t, _Bool *, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_one_owned_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, float, const ds4_gpu_tensor *, ds4_gpu_tensor *, _Bool, ds4_gpu_tensor *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, float, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_owned_packed_combine_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_owned_slots_combine_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_routed_moe_set_selected_override(const int32_t *, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, _Bool, _Bool, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, float, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_router_select_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, uint32_t, _Bool, _Bool, const ds4_gpu_tensor *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_set_current_device(int), 1)
DS4_TCIM_REQUIRED(int ds4_gpu_set_current_device_fenced(int), 1)
DS4_TCIM_REQUIRED(int ds4_gpu_set_model_fd(int), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_set_model_fd_for_map(int, const void *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_set_model_map_range(const void *, uint64_t, uint64_t, uint64_t, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_set_model_map_spans(const void *, uint64_t, const uint64_t *, const uint64_t *, uint32_t, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_down_hc_expand_q8_0_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, float, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, _Bool), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_shared_mid_swiglu_q8_0_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_signal_selected_readback_ready(uint64_t *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, float, float), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_synchronize(void), 0)
DS4_TCIM_REQUIRED(ds4_gpu_tensor * ds4_gpu_tensor_alloc(uint64_t), NULL)
DS4_TCIM_REQUIRED(ds4_gpu_tensor * ds4_gpu_tensor_alloc_ptr_on(int, uint64_t), NULL)
DS4_TCIM_REQUIRED(uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_tensor_copy(ds4_gpu_tensor *, uint64_t, const ds4_gpu_tensor *, uint64_t, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_tensor_device(const ds4_gpu_tensor *), -1)
DS4_TCIM_REQUIRED(int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *, float, uint64_t), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_tensor_read(const ds4_gpu_tensor *, uint64_t, void *, uint64_t), 0)
DS4_TCIM_REQUIRED(ds4_gpu_tensor * ds4_gpu_tensor_view(const ds4_gpu_tensor *, uint64_t, uint64_t), NULL)
DS4_TCIM_REQUIRED(int ds4_gpu_tensor_write(ds4_gpu_tensor *, uint64_t, const void *, uint64_t), 0)
DS4_TCIM_REQUIRED(uint64_t ds4_gpu_tier_free_vram(int), 0)
DS4_TCIM_REQUIRED(int ds4_gpu_wait_selected_readback_ready(uint64_t, const char *), 0)

/* OPTIONAL_FALLBACK: 40 entries. */
DS4_TCIM_OPTIONAL(int ds4_gpu_build_derived_artifacts(const void *, uint64_t, const char *), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_commands_active(void), 0)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *))
DS4_TCIM_OPTIONAL(int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *), -1)
DS4_TCIM_OPTIONAL(int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *), -1)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_decode_graphs_invalidate(void))
DS4_TCIM_OPTIONAL(int ds4_gpu_decode_graphs_supported(void), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_dsv4_comp_row_finalize_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, uint32_t, uint64_t, ds4_gpu_tensor *, uint32_t, uint64_t, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, float), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, uint32_t, ds4_gpu_tensor *, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, float), 0)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_enable_q8_dequant_gemm(void))
DS4_TCIM_OPTIONAL(int ds4_gpu_hc_rms_norm_mix_f16_available(void), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_hc_rms_norm_mix_f16_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint32_t, float), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_matmul_f16_pair_compressor_store_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_matmul_f16_quad_compressor_store_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_model_range_replaced(const void *, uint64_t, uint64_t), 0)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_model_residency_skip(int))
DS4_TCIM_OPTIONAL(int ds4_gpu_preload_q4_expert_tables(const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t), 0)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_print_memory_report(const char *))
DS4_TCIM_OPTIONAL(int ds4_gpu_pro_q4_expert_table_auto_available(void), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_q8_cache_suppressed(void), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_qkv_pair_quad_compressor_store_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_OPTIONAL(uint64_t ds4_gpu_recommended_working_set_size(void), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_router_shared_gate_up_q8_0_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, float, _Bool), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_set_decode_fast_attention(int), 0)
DS4_TCIM_OPTIONAL(int ds4_gpu_set_decode_score_vec4(int), 0)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_glm_model(_Bool))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_glm_streaming_prefill_full_layer(_Bool))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_q8_cache_suppressed(int))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_quality(_Bool))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_ssd_streaming(_Bool))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_streaming_expert_cache_budget(uint32_t))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t))
DS4_TCIM_OPTIONAL(int ds4_gpu_should_use_managed_kv_cache(uint64_t, uint64_t), 0)
DS4_TCIM_OPTIONAL(uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(uint64_t, uint64_t), 0)
DS4_TCIM_OPTIONAL(uint32_t ds4_gpu_stream_expert_cache_configured_count(void), 0)
DS4_TCIM_OPTIONAL(uint32_t ds4_gpu_stream_expert_cache_current_count(void), 0)
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_stream_expert_cache_reset_route_hotness(void))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_tp_keepalive_pause(int))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_tp_set_attn_head_split(int))
DS4_TCIM_OPTIONAL_VOID(void ds4_gpu_tp_suspend_expert_sharding(int))

/* METAL_PRIVATE_LINK_FALLBACK: 5 entries.
 * These hooks are not part of ds4_gpu.h. Their availability probes remain
 * false, so the shared graph keeps the standalone portable operations. */
DS4_TCIM_METAL_PRIVATE(int ds4_gpu_decode_attn_rope_fuse_available(void), 0)
DS4_TCIM_METAL_PRIVATE(int ds4_gpu_decode_attn_rope_fuse_used(void), 0)
DS4_TCIM_METAL_PRIVATE(int ds4_gpu_kv_rope_fp8_fuse_available(void), 0)
DS4_TCIM_METAL_PRIVATE(int ds4_gpu_kv_rope_fp8_store_raw_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float), 0)
DS4_TCIM_METAL_PRIVATE_VOID(void ds4_gpu_set_decode_attn_rope_fuse(uint32_t, uint32_t, uint32_t, uint32_t, _Bool, float, float, float, float, float, float))

/* UNAVAILABLE: 31 entries. */
DS4_TCIM_UNAVAILABLE(int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_attention_output_q8_tp_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint64_t, const ds4_gpu_tensor *), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_device_cache_support_tensors(int, int, const ds4_tensor_range *, int, int), 1)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_device_cache_tensors(int, const ds4_tensor_range *, int), 1)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, const ds4_gpu_tensor *, uint32_t, _Bool, uint32_t, _Bool, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(const ds4_gpu_stream_expert_table *, const ds4_gpu_tensor *, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_matmul_q8_0_kslice_rows_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_matmul_quant_kslice_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_matmul_quant_rows_scalar_tensor(ds4_gpu_tensor *, const void *, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_moe_handoff_pack_tensor(ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_register_support_map(const void *, uint64_t, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_stream_expert_cache_begin_selected_load(const ds4_gpu_stream_expert_table *, const int32_t *, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_stream_expert_cache_prepare_selected_batch(const ds4_gpu_stream_expert_table *, const int32_t *, uint32_t, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_stream_expert_cache_seed_experts(const ds4_gpu_stream_expert_table *, const int32_t *, const uint32_t *, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_stream_expert_cache_seed_selected(const ds4_gpu_stream_expert_table *, const int32_t *, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(ds4_gpu_tensor * ds4_gpu_tensor_alloc_managed(uint64_t), NULL)
DS4_TCIM_UNAVAILABLE(ds4_gpu_tensor * ds4_gpu_tensor_alloc_managed_on(int, uint64_t), NULL)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_copy_xdev3_default_dst(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t, ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *, const ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *, int), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *, int), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tp_batch_gate_encode(uint32_t, uint32_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tp_big_gate_encode(uint32_t, uint32_t, const ds4_gpu_tensor *, ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(uint64_t ds4_gpu_tp_big_gate_kick(uint32_t, uint32_t, const ds4_gpu_tensor *, ds4_gpu_tensor *, uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tp_big_gate_wait(uint64_t), 0)
DS4_TCIM_UNAVAILABLE(int ds4_gpu_tp_gate_encode(uint32_t, uint32_t), 0)

#undef DS4_TCIM_REQUIRED
#undef DS4_TCIM_OPTIONAL
#undef DS4_TCIM_OPTIONAL_VOID
#undef DS4_TCIM_METAL_PRIVATE
#undef DS4_TCIM_METAL_PRIVATE_VOID
#undef DS4_TCIM_UNAVAILABLE
