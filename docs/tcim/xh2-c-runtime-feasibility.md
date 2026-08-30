# XH2 纯 C 算子 runtime 可行性

日期：2026-08-29

## 结论

**Go：按 CUDA 式分层实现 TCIM/XH2 算子后端。** hq50 小 kernel prototype 已证明纯 C + `libhal_xh2a` 能装载 raw payload、提交 KLD，并按正确地址域读写 DDR。它证明的是执行底座可行，不代表 DS4 算子覆盖、数值和性能已经完成。

- 删除 `DS4_BACKEND_TCIM`；TCIM 独立构建与 ROCm 一样复用 `DS4_BACKEND_CUDA` identity 和 `ds4_gpu_*` 契约。
- 复用的不是 CUDA runtime、kernel 或 capability；TCIM 只链接自己的 C adapter/runtime。
- `.xh2k` 类似 `cubin`：一个算子/variant 的 payload 加最小 ABI/KLD metadata，不是 HMM，也不是模型容器。
- weights、KV、inputs、outputs、workspace、GGUF/SSD cache 均由 DS4 宿主代码管理。
- 官方 Dots TCIM 路径只保留为 ABI/数值 oracle；entry-only relink 不再是 production 主路线或硬门槛。

## 三层实现

```text
ds4.c / GGUF / SSD / KV
        -> ds4_tcim.c      ds4_gpu_*、shape/variant、weight residency
        -> xh2rt.c         BO、.xh2k、command batch、launch/sync/error
        -> libhal_xh2a     allocator、transfer、IPU group/KLD

build: operator source -> op.xh2k + generated typed C descriptor/stub
```

`xh2rt` 不知道 DS4、GGUF 或模型 slot。descriptor 与 `.xh2k` 由同一 canonical ABI IR 生成；runtime 在申请 kernel BO 前核对 payload/descriptor hash pair。

## 已定标的约束

- KLD 地址和 kernel 内 DDR pointer 使用 `IPU_IOMAP(global) = global - 0x1000000000`；HAL transfer 仍使用 global address。XH2A 1.4.0 driver 的 `xh2a_ipu_config.h` 定义了该映射，hq50 noop/store probe 也验证了 global pointer 超时而 IOMAP pointer 连续三次正确回读。
- `ds4_gpu_tensor.ptr` 保存可做 `ptr + offset` 的 IOMAP 地址；adapter registry 另持有 HAL BO/global address/size/lifetime。
- HAL group 是阻塞 command batch：最多 255 个 KLD、SPM 参数合计不超过 1 MiB，同组 core/tile/param type 必须一致。
- cache-mode hardware descriptor 只保留 16-bit `kernel_size`；首版 `.xh2k` 拒绝 payload `> 65535` bytes。
- HAL 返回值按 `rc != 0` 判错；sync timeout/device reset 后 poison context，KV/logits/output 不得继续使用或静默 fallback。

## 下一阶段门禁

1. `.xh2k` pack/hash、BO registry、2-core/4-tile fill/add、同组连续 launch 和负测；
2. 按目标模型实际可达的 `ds4_gpu_*` manifest 实现 required primitives，逐算子对 CPU/CUDA oracle；
3. 分段验证 attention/KV、MoE 和 output head，再做 raw GGUF + SSD cache 的连续 decode；
4. 最后关闭长稳态、OOM/timeout/reinit、性能及 CPU/Metal/CUDA/ROCm 回归。

实现规格与验收门见 [GitHub Issue #6](https://github.com/fixedpointworks/ds4/issues/6)，backend identity 见 [ADR-0004](../adr/0004-tcim-reuses-cuda-backend-identity.md)。
