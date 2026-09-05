/* Step 1 smoke for real XH2 logical device 0; no model file is required.
 * Keep this limited to the runtime/tensor/static-span substrate. */
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "tcim/xh2rt.h"
#include <ipu_api.h>

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s (errno=%d)\n", \
                __FILE__, __LINE__, #condition, errno); \
        exit(1); \
    } \
} while (0)

static void clean_context(void)
{
    ds4_gpu_cleanup();
    CHECK(ds4_gpu_commands_active() == 0);
}

static float from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void test_return_contracts(void)
{
    ds4_gpu_config config = {0};
    ds4_gpu_tensor stack = {0};
    ds4_gpu_tensor *heap;

    config.n_gpus = 1;
    config.device_indices[0] = 0;
    config.vram_bytes[0] = 1024 * 1024;
    CHECK(ds4_gpu_init_multi(NULL) == 0);
    CHECK(ds4_gpu_init_multi(&config) != 0);
    CHECK(ds4_gpu_init() != 0);
    CHECK(g_n_gpus == 1 && g_gpu[0].device_id == 0);
    CHECK(ds4_gpu_set_current_device(0) == 0);
    CHECK(ds4_gpu_set_current_device_fenced(0) == 0);
    CHECK(ds4_gpu_tensor_alloc_on(&stack, 0, 16) == 0);
    CHECK(ds4_gpu_tensor_bytes(&stack) == 16);
    CHECK(ds4_gpu_tensor_device(&stack) == 0);
    ds4_gpu_tensor_free_in_place(&stack);
    CHECK(stack.ptr == NULL && stack.bytes == 0);
    ds4_gpu_tensor_free_in_place(&stack);
    ds4_gpu_tensor_free_in_place(NULL);
    ds4_gpu_tensor_free(NULL);

    heap = ds4_gpu_tensor_alloc_ptr_on(0, 16);
    CHECK(heap != NULL);
    CHECK(ds4_gpu_tensor_device(heap) == 0);
    CHECK(ds4_gpu_tensor_bytes(heap) == 16);
    /* Unsupported native add must not become a public product operator. */
    CHECK(ds4_gpu_add_tensor(heap, heap, heap, 4) == 0);
    CHECK(ds4_gpu_tensor_fill_f32(heap, 1.0f, 4) != 0);
    errno = E2BIG;
    ds4_gpu_tensor_free(heap);
    CHECK(errno == E2BIG);
    heap = ds4_gpu_tensor_alloc(0);
    CHECK(heap != NULL && ds4_gpu_tensor_bytes(heap) == 1);
    ds4_gpu_tensor_free(heap);
    CHECK(ds4_gpu_tensor_bytes(NULL) == 0);
    CHECK(ds4_gpu_tensor_device(NULL) == -1);

    CHECK(ds4_gpu_synchronize() != 0);
    clean_context();
    puts("adapter return families PASS: nonzero-success, zero-success, pointer/void, device 0, required add fail-closed");
}

static void test_view_lifetime(unsigned int order)
{
    uint8_t expected[256], observed[256], replacement[13];
    ds4_gpu_tensor *base, *parent, *child;
    uintptr_t token;

    CHECK(ds4_gpu_init() != 0);
    base = ds4_gpu_tensor_alloc(sizeof(expected));
    CHECK(base != NULL);
    token = (uintptr_t)base->ptr;
    CHECK(sizeof(uintptr_t) == sizeof(uint64_t));
    CHECK((void *)(uintptr_t)(uint64_t)token == base->ptr);
    for (size_t i = 0; i < sizeof(expected); i++)
        expected[i] = (uint8_t)(i * 17u + 9u);
    memset(replacement, 0x5a, sizeof(replacement));
    CHECK(ds4_gpu_tensor_write(base, 0, expected, sizeof(expected)) != 0);
    parent = ds4_gpu_tensor_view(base, 17, 129);
    CHECK(parent != NULL);
    child = ds4_gpu_tensor_view(parent, 11, 57);
    CHECK(child != NULL);
    CHECK((uintptr_t)parent->ptr == token + 17);
    CHECK((uintptr_t)child->ptr == token + 28);
    CHECK(ds4_gpu_tensor_bytes(parent) == 129);
    CHECK(ds4_gpu_tensor_bytes(child) == 57);
    CHECK(ds4_gpu_tensor_device(child) == 0);
    CHECK(ds4_gpu_tensor_write(child, 5, replacement, sizeof(replacement)) != 0);
    memcpy(expected + 33, replacement, sizeof(replacement));

    if (order == 0) {
        ds4_gpu_tensor_free(base);
        CHECK(ds4_gpu_tensor_read(parent, 0, observed, 129) != 0);
        CHECK(memcmp(observed, expected + 17, 129) == 0);
        ds4_gpu_tensor_free(parent);
        CHECK(ds4_gpu_tensor_read(child, 0, observed, 57) != 0);
        CHECK(memcmp(observed, expected + 28, 57) == 0);
        ds4_gpu_tensor_free(child);
    } else if (order == 1) {
        ds4_gpu_tensor_free(child);
        ds4_gpu_tensor_free(parent);
        CHECK(ds4_gpu_tensor_read(base, 0, observed, sizeof(observed)) != 0);
        CHECK(memcmp(observed, expected, sizeof(expected)) == 0);
        ds4_gpu_tensor_free(base);
    } else {
        ds4_gpu_tensor_free(parent);
        ds4_gpu_tensor_free(base);
        CHECK(ds4_gpu_tensor_write(child, 0, replacement, sizeof(replacement)) != 0);
        memcpy(expected + 28, replacement, sizeof(replacement));
        CHECK(ds4_gpu_tensor_read(child, 0, observed, 57) != 0);
        CHECK(memcmp(observed, expected + 28, 57) == 0);
        ds4_gpu_tensor_free(child);
    }
    clean_context();
    printf("adapter nested-view lifetime order %u PASS: token=0x%016" PRIxPTR
           " never host-dereferenced; surviving views retain BO\n", order, token);
}

static void test_transfer_bounds(void)
{
    ds4_gpu_tensor unknown = {0};
    ds4_gpu_tensor *a, *b, *unaligned;
    uint8_t source[65], expected[65], observed[65], zero = 0;

    CHECK(ds4_gpu_init() != 0);
    a = ds4_gpu_tensor_alloc(sizeof(source));
    b = ds4_gpu_tensor_alloc(sizeof(source));
    CHECK(a != NULL && b != NULL);
    for (size_t i = 0; i < sizeof(source); i++) source[i] = (uint8_t)(i ^ 0x39);
    memset(expected, 0xa5, sizeof(expected));
    CHECK(ds4_gpu_tensor_write(a, 0, source, sizeof(source)) != 0);
    CHECK(ds4_gpu_tensor_write(b, 0, expected, sizeof(expected)) != 0);
    CHECK(ds4_gpu_tensor_copy(b, 3, a, 1, 61) != 0);
    memcpy(expected + 3, source + 1, 61);
    CHECK(ds4_gpu_tensor_read(b, 0, observed, sizeof(observed)) != 0);
    CHECK(memcmp(observed, expected, sizeof(expected)) == 0);
    CHECK(ds4_gpu_tensor_write(b, 65, &zero, 0) != 0);
    CHECK(ds4_gpu_tensor_read(b, 65, &zero, 0) != 0);
    unaligned = ds4_gpu_tensor_view(a, 1, 60);
    CHECK(unaligned != NULL);
    CHECK(ds4_gpu_tensor_view(a, UINT64_MAX, 1) == NULL);
    CHECK(ds4_gpu_tensor_view(a, 1, UINT64_MAX) == NULL);
    CHECK(ds4_gpu_tensor_view(a, 64, 2) == NULL);
    CHECK(ds4_gpu_tensor_write(a, UINT64_MAX, &zero, 1) == 0);
    CHECK(ds4_gpu_tensor_write(a, 65, &zero, 1) == 0);
    CHECK(ds4_gpu_tensor_read(a, 64, observed, 2) == 0);
    CHECK(ds4_gpu_tensor_read(a, 1, observed, UINT64_MAX) == 0);
    CHECK(ds4_gpu_tensor_copy(b, 64, a, 0, 2) == 0);
    CHECK(ds4_gpu_tensor_copy(b, 0, a, UINT64_MAX, 1) == 0);
    CHECK(ds4_gpu_tensor_copy(a, 1, a, 0, 16) == 0);
    CHECK(ds4_gpu_tensor_fill_f32(a, 1.0f, UINT64_MAX) == 0);
    CHECK(ds4_gpu_tensor_fill_f32(a, 1.0f, 17) == 0);
    CHECK(ds4_gpu_tensor_fill_f32(unaligned, 1.0f, 1) == 0);
    unknown = *a;
    CHECK(ds4_gpu_tensor_read(&unknown, 0, observed, 1) == 0);
    CHECK(ds4_gpu_tensor_bytes(&unknown) == 0);
    CHECK(ds4_gpu_tensor_device(&unknown) == -1);
    CHECK(ds4_gpu_init() != 0);
    CHECK(ds4_gpu_set_current_device(0) == 0);
    CHECK(ds4_gpu_set_current_device_fenced(0) == 0);
    CHECK(ds4_gpu_tensor_read(b, 0, observed, sizeof(observed)) != 0);
    CHECK(memcmp(observed, expected, sizeof(expected)) == 0);
    ds4_gpu_tensor_free(unaligned);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    clean_context();
    puts("adapter transfers/bounds PASS: odd offsets/tails/guards, overlap/range/overflow rejection, healthy retry");
}

static void test_fill_and_commands(void)
{
    enum { WORDS = 1047, GUARD = 8 };
    const uint32_t counts[] = {0, 1, 31, 32, 33, 37, 1031};
    const uint32_t bits[] = {UINT32_C(0x3f800000), UINT32_C(0x80000000),
                            UINT32_C(0x7fc12345)};
    uint32_t expected[WORDS], observed[WORDS];
    ds4_gpu_tensor *base, *view;

    CHECK(ds4_gpu_init() != 0);
    base = ds4_gpu_tensor_alloc(sizeof(expected));
    CHECK(base != NULL);
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
        const uint64_t count = counts[ci];
        view = ds4_gpu_tensor_view(base, GUARD * 4, count * 4);
        CHECK(view != NULL);
        for (size_t bi = 0; bi < sizeof(bits) / sizeof(bits[0]); bi++) {
            for (size_t i = 0; i < WORDS; i++) expected[i] = UINT32_C(0xdeadbeef);
            CHECK(ds4_gpu_tensor_write(base, 0, expected, sizeof(expected)) != 0);
            CHECK(ds4_gpu_tensor_fill_f32(view, from_bits(bits[bi]), count) != 0);
            for (size_t i = 0; i < count; i++) expected[GUARD + i] = bits[bi];
            CHECK(ds4_gpu_tensor_read(base, 0, observed, sizeof(observed)) != 0);
            CHECK(memcmp(observed, expected, sizeof(expected)) == 0);
        }
        ds4_gpu_tensor_free(view);
    }
    CHECK(ds4_gpu_commands_active() == 0);
    CHECK(ds4_gpu_begin_commands() != 0);
    CHECK(ds4_gpu_commands_active() != 0);
    CHECK(ds4_gpu_begin_commands() == 0);
    CHECK(ds4_gpu_tensor_fill_f32(base, 2.0f, WORDS) != 0);
    CHECK(ds4_gpu_flush_encoder() != 0);
    CHECK(ds4_gpu_commands_active() != 0);
    CHECK(ds4_gpu_flush_commands() != 0);
    CHECK(ds4_gpu_commands_active() != 0);
    CHECK(ds4_gpu_tensor_fill_f32(base, 3.0f, WORDS) != 0);
    CHECK(ds4_gpu_tensor_read(base, 0, observed, sizeof(observed)) != 0);
    CHECK(ds4_gpu_commands_active() != 0);
    for (size_t i = 0; i < WORDS; i++) CHECK(observed[i] == UINT32_C(0x40400000));
    CHECK(ds4_gpu_end_commands() != 0);
    CHECK(ds4_gpu_commands_active() == 0);
    CHECK(ds4_gpu_begin_commands() != 0);
    CHECK(ds4_gpu_tensor_fill_f32(base, 4.0f, WORDS) != 0);
    CHECK(ds4_gpu_synchronize() != 0);
    CHECK(ds4_gpu_commands_active() == 0);
    ds4_gpu_tensor_free(base);
    clean_context();
    puts("adapter fill/commands PASS: F32 raw bits/NaN/signed zero, tails/guards/zero count, active/flush/read/end/sync transitions");
}

static void test_batch_release(void)
{
    ds4_gpu_tensor *base, *view;
    uint32_t observed[8];

    CHECK(ds4_gpu_init() != 0);
    base = ds4_gpu_tensor_alloc(32);
    CHECK(base != NULL);
    view = ds4_gpu_tensor_view(base, 0, 32);
    CHECK(view != NULL);
    CHECK(ds4_gpu_begin_commands() != 0);
    CHECK(ds4_gpu_tensor_fill_f32(base, 5.0f, 8) != 0);
    ds4_gpu_tensor_free(base);
    CHECK(ds4_gpu_commands_active() != 0);
    CHECK(ds4_gpu_end_commands() != 0);
    CHECK(ds4_gpu_tensor_read(view, 0, observed, sizeof(observed)) != 0);
    for (size_t i = 0; i < 8; i++) CHECK(observed[i] == UINT32_C(0x40a00000));
    ds4_gpu_tensor_free(view);
    base = ds4_gpu_tensor_alloc(32);
    CHECK(base != NULL);
    CHECK(ds4_gpu_begin_commands() != 0);
    CHECK(ds4_gpu_tensor_fill_f32(base, 6.0f, 8) != 0);
    ds4_gpu_tensor_free(base);
    CHECK(ds4_gpu_end_commands() != 0);
    clean_context();
    puts("adapter batch release PASS: wrapper removal preserves pending BO and sibling view lifetime");
}

static void test_cleanup_wrappers(void)
{
    ds4_gpu_tensor stack = {0};
    ds4_gpu_tensor *heap;
    uint32_t word = 7;

    CHECK(ds4_gpu_init() != 0);
    heap = ds4_gpu_tensor_alloc(4);
    CHECK(heap != NULL);
    CHECK(ds4_gpu_tensor_alloc_on(&stack, 0, 4) == 0);
    clean_context();
    CHECK(ds4_gpu_tensor_write(heap, 0, &word, 4) == 0);
    CHECK(ds4_gpu_tensor_write(&stack, 0, &word, 4) == 0);
    CHECK(ds4_gpu_init() != 0);
    CHECK(ds4_gpu_tensor_write(heap, 0, &word, 4) == 0);
    CHECK(ds4_gpu_tensor_write(&stack, 0, &word, 4) == 0);
    ds4_gpu_tensor_free(heap);
    ds4_gpu_tensor_free_in_place(&stack);
    CHECK(stack.ptr == NULL && stack.bytes == 0);
    clean_context();
    puts("adapter cleanup wrapper lifetime PASS: old heap/stack tensors stay stale across reinit and release safely");
}

/* Observe only adapter H2D calls; all device operations use the real runtime.
 * This avoids dereferencing the opaque iomap token on the host. */
static struct {
    xh2rt_context *context;
    xh2rt_buffer_view *view;
    const void *source;
    uint64_t bytes;
} live_copies[8];
static unsigned live_copy_count;
static int live_observe;
static int live_fail_alloc;
int __real_xh2rt_buffer_write(xh2rt_context *, xh2rt_buffer_view *, uint64_t,
                              const void *, uint64_t);
int __real_xh2rt_buffer_alloc(xh2rt_context *, uint64_t, xh2rt_buffer_view **);
int __wrap_xh2rt_buffer_alloc(xh2rt_context *c, uint64_t bytes,
                              xh2rt_buffer_view **out)
{
    if (live_fail_alloc && --live_fail_alloc == 0) {
        errno = ENOSPC;
        return 0;
    }
    return __real_xh2rt_buffer_alloc(c, bytes, out);
}
int __wrap_xh2rt_buffer_write(xh2rt_context *c, xh2rt_buffer_view *v,
                              uint64_t off, const void *src, uint64_t bytes)
{
    int ok = __real_xh2rt_buffer_write(c, v, off, src, bytes);
    if (live_observe && ok) {
        CHECK(live_copy_count < 8 && off == 0);
        live_copies[live_copy_count].context = c;
        live_copies[live_copy_count].view = v;
        live_copies[live_copy_count].source = src;
        live_copies[live_copy_count++].bytes = bytes;
    }
    return ok;
}

static void live_stage(const uint8_t *model, uint64_t model_bytes,
                       const uint64_t *offsets, const uint64_t *sizes,
                       unsigned count)
{
    uint8_t observed[96];
    live_copy_count = 0;
    live_observe = 1;
    CHECK(ds4_gpu_set_model_map_spans(model, model_bytes, offsets, sizes,
                                      count, 96));
    live_observe = 0;
    CHECK(live_copy_count == count);
    for (unsigned i = 0; i < count; i++) {
        uintptr_t expected_token = 0;
        void *token = NULL;
        int device = -1;
        CHECK(sizes[i] <= sizeof(observed));
        CHECK(live_copies[i].source == model + offsets[i]);
        CHECK(live_copies[i].bytes == sizes[i]);
        CHECK(xh2rt_buffer_read(live_copies[i].context, observed,
                                live_copies[i].view, 0, sizes[i]));
        CHECK(memcmp(observed, model + offsets[i], sizes[i]) == 0);
        CHECK(xh2rt_buffer_iomap(live_copies[i].context, live_copies[i].view,
                                 0, sizes[i], 1, &expected_token));
        CHECK(ds4_gpu_lookup_cache(offsets[i], sizes[i], &device, &token));
        CHECK(device == 0 && (uintptr_t)token == expected_token);
        CHECK(ds4_gpu_lookup_cache(offsets[i] + 1, sizes[i] - 1,
                                    &device, &token));
        CHECK((uintptr_t)token == expected_token + 1);
    }
}

static void test_static_span_live(void)
{
    uint8_t model[512];
    const uint64_t offsets[] = {32, 128, 200, 320};
    const uint64_t sizes[] = {24, 31, 17, 43};
    const uint64_t bad_offsets[] = {500}, bad_sizes[] = {13};
    for (size_t i = 0; i < sizeof(model); i++)
        model[i] = (uint8_t)(i * 29u + (i >> 2));
    for (unsigned round = 0; round < 3; round++) {
        CHECK(ds4_gpu_init());
        live_stage(model, sizeof(model), offsets, sizes, 1);
        live_stage(model, sizeof(model), offsets + 1, sizes + 1, 2);
        CHECK(!ds4_gpu_lookup_cache(offsets[0], sizes[0], NULL, NULL));
        CHECK(!ds4_gpu_lookup_cache(159, 41, NULL, NULL));
        CHECK(!ds4_gpu_lookup_cache(128, 89, NULL, NULL));
        CHECK(ds4_gpu_cache_model_range(model, sizeof(model), 128, 89, "gap"));
        CHECK(!ds4_gpu_cache_q8_f16_range(model, sizeof(model), 128, 89,
                                          1, 1, "gap"));
        CHECK(!ds4_gpu_set_model_map_spans(model, sizeof(model), bad_offsets,
                                           bad_sizes, 1, 13));
        CHECK(errno == EINVAL);
        CHECK(ds4_gpu_lookup_cache(128, 31, NULL, NULL));
        CHECK(ds4_gpu_begin_commands());
        CHECK(!ds4_gpu_set_model_map_spans(model, sizeof(model), offsets,
                                           sizes, 4, 43));
        CHECK(errno == EBUSY);
        CHECK(ds4_gpu_end_commands());
        /* Fail the second allocation after one real H2D. The runtime and
         * first allocation/free remain real; failure injection is host-only. */
        live_fail_alloc = 2;
        CHECK(!ds4_gpu_set_model_map_spans(model, sizeof(model), offsets,
                                           sizes, 4, 43));
        CHECK(errno == ENOSPC);
        CHECK(!ds4_gpu_lookup_cache(32, 24, NULL, NULL));
        CHECK(!ds4_gpu_lookup_cache(128, 31, NULL, NULL));
        live_stage(model, sizeof(model), offsets, sizes, 4);
        live_copy_count = 0;
        live_observe = 1;
        CHECK(ds4_gpu_set_model_map_spans(model, sizeof(model), offsets,
                                          sizes, 4, 43));
        CHECK(ds4_gpu_cache_q8_f16_range(model, sizeof(model), 128, 31,
                                         1, 1, "raw"));
        live_observe = 0;
        CHECK(live_copy_count == 0);
        clean_context();
        CHECK(!ds4_gpu_lookup_cache(32, 24, NULL, NULL));
        printf("adapter static LIVE round %u PASS: token -> layer -> all-static; "
               "exact H2D/D2H and iomap; gaps/preflight/batch rejection; "
               "injected partial alloc failure/retry; reuse/cleanup\n", round);
    }
}

/* Only the explicitly injected open call is replaced. No device resources
 * are created for that attempt; every successful path uses the real HAL. */
static int fail_open;
static unsigned open_calls;
int32_t __real_xh2a_ipu_open(uint32_t device, fd_handle_t *fd);
int32_t __wrap_xh2a_ipu_open(uint32_t device, fd_handle_t *fd)
{
    open_calls++;
    if (fail_open) {
        errno = EACCES;
        return -17;
    }
    return __real_xh2a_ipu_open(device, fd);
}

static void test_init_failure(void)
{
    fail_open = 1;
    CHECK(!ds4_gpu_init());
    CHECK(errno == EACCES);
    CHECK(open_calls == 1);
    CHECK(!ds4_gpu_commands_active());
    clean_context();
    fail_open = 0;
    CHECK(ds4_gpu_init());
    CHECK(open_calls == 2);
    clean_context();
    puts("runtime init failure PASS: HAL error reaches adapter unchanged; no alternate init; retry succeeds");
}

static void test_typed_runtime(void)
{
    enum { COUNT = 1031, WORDS = COUNT + 2 };
    xh2rt_context *context = NULL;
    xh2rt_buffer_view *a = NULL, *b = NULL, *out = NULL, *view = NULL;
    uint32_t observed[WORDS] = {0};
    CHECK(xh2rt_context_open(0, &context));
    CHECK(xh2rt_buffer_alloc(context, sizeof(observed), &a));
    CHECK(xh2rt_buffer_alloc(context, sizeof(observed), &b));
    CHECK(xh2rt_buffer_alloc(context, sizeof(observed), &out));
    CHECK(xh2rt_buffer_write(context, out, 0, observed, sizeof(observed)));
    CHECK(xh2rt_buffer_view_create(context, out, sizeof(uint32_t),
                                   COUNT * sizeof(uint32_t), &view));
    CHECK(xh2rt_commands_begin(context));
    CHECK(xh2rt_fill_f32(context, a, UINT32_C(0x3f800000), WORDS));
    CHECK(xh2rt_fill_f32(context, b, UINT32_C(0x40000000), WORDS));
    CHECK(xh2rt_add_f32(context, a, b, view, COUNT));
    CHECK(xh2rt_commands_end(context));
    CHECK(!xh2rt_commands_active(context));
    CHECK(xh2rt_buffer_read(context, observed, out, 0, sizeof(observed)));
    CHECK(observed[0] == 0 && observed[WORDS - 1] == 0);
    for (unsigned i = 1; i <= COUNT; i++)
        CHECK(observed[i] == UINT32_C(0x40400000));
    CHECK(xh2rt_buffer_release(context, &view));
    CHECK(xh2rt_buffer_release(context, &out));
    CHECK(xh2rt_buffer_release(context, &b));
    CHECK(xh2rt_buffer_release(context, &a));
    CHECK(xh2rt_context_close(&context));
    CHECK(context == NULL);
    puts("typed runtime PASS: embedded fill/add, command-group completion, 1031-element tail/guards, release");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    test_init_failure();
    test_typed_runtime();
    test_return_contracts();
    for (unsigned int order = 0; order < 3; order++) test_view_lifetime(order);
    test_transfer_bounds();
    test_fill_and_commands();
    test_batch_release();
    test_cleanup_wrappers();
    test_static_span_live();
    puts("test_tcim_runtime: PASS");
    return 0;
}
