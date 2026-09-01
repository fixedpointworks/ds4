# XH2 纯 C 算子 runtime 可行性

更新：2026-08-31

## 结论

**纯 C + `libhal_xh2a` 可作为 TCIM/XH2 算子后端的执行底座；当前采用 SDK 原生参数 ABI 与内嵌 kernel。** Issue #6 的 D0–D4 已通过；这不代表完整 DS4 算子覆盖、模型数值和性能验收已经完成。

- 2026-08-31 已实现 kernel 的计算结果与执行状态通过显式输出参数写回。新增 ABI、payload 与状态 BO 已重跑 host 和 hq50 验证，D0–D3 的逐项验收见 Issue #6；`d85602a` 的旧 ABI evidence 仅保留供追溯。
- D3 完成态故障已在 hq50 验证：安全、正常完成的 kernel 写出非零状态，即使 HAL sync rc/result 均为零，production runtime 仍拒收、poison 并保留原始错误；前错后成功、非 writer hart 的 stripe 错误和未写状态均被识别，正常完成后资源回到基线。HAL 的非零判错与 fake-HAL 测试仍保留；这些证据不证明 timeout/reset 后安全恢复或捕获全部静默计算故障。
- D4 基础 DS4 adapter 已接入 init/cleanup、tensor 分配/view/传输/fill、阻塞 command batch 与 logical tier 0；host 故障测试和 hq50 三轮生命周期/数值验证通过。`tensor.ptr` 仅为经 runtime 校验的 IOMAP token；base wrapper 释放不影响仍存活的 view，cleanup/free 保留原始 errno。未实现算子继续 fail closed，Primitive add 仍只用于 runtime 验证。
- 后续仍需完成目标模型所需算子及全模型集成，验证数值、GGUF/SSD 连续 decode、长稳态、故障恢复、性能，以及现有后端回归。

契约与阶段验收记录见 [Issue #6](https://github.com/fixedpointworks/ds4/issues/6)；环境验证脚本与结果保留在 ignored 的 `misc/`，正式 evidence 记录完整命令、版本、hash 与最终输出。

## 方案

- TCIM 独立构建，复用 DS4 的 GPU 接口和 CUDA backend identity，但不复用 CUDA runtime、kernel 或 capability，见 [ADR-0004](../adr/0004-tcim-reuses-cuda-backend-identity.md)。
- DS4 宿主代码管理模型、权重、KV、GGUF/SSD cache 和算子调度；`xh2rt` 只负责设备资源与执行，不承担模型策略。
- host 保持纯 C，直接调用 HAL；设备 kernel 使用 HDPL/Primitive 构建，由 `xxd` 生成完整 C byte array 后内嵌，payload 必须为 1..65535 bytes。每个 context 在 open 时分配并上传 code BO，后续 launch 复用；运行时不依赖 HDPL 编译器或外部 kernel 文件。
- TCIM 的接口实现和 fail-closed fallback 都直接维护在 `ds4_tcim.c`。尚未实现的函数逐个标记 `TCIM_STUB`；使用 `rg -n '^/\* TCIM_STUB:' ds4_tcim.c` 可查看剩余范围，实现时直接替换对应函数体并删除该标记。普通 C 编译检查共享 header 中的原型，最终链接和 `tests/test_tcim.sh` 检查当前 TCIM CLI 实际引用的缺失符号，以及重复、非函数、weak、错误 provider 与 unresolved symbol；不维护第二套 contract/inventory 或生成式 C source。
- host typed wrapper 与设备入口共同维护 SDK 原生参数契约，以编译、envelope known-vector 和实机测试约束漂移。目标 entry 为 `void fill_f32(dst_iomap, status_iomap, value_bits, count)` 和 `void add_f32(lhs_iomap, rhs_iomap, dst_iomap, status_iomap, count)`；SPM envelope 为 8-byte zero header、8-byte argc 与各 8-byte little-endian 参数，分别为 4 参数/48 bytes 和 5 参数/56 bytes。KLD 固定 2 cores × 4 tiles、SPM、`ilm_mode=1`，不引入自定义描述块、模块容器、签名匹配或动态 module cache。
- `status_iomap` 是 kernel 的显式输出地址参数，由 `xh2rt` 内部管理其 BO，公开 wrapper 与 `ds4_gpu_*` 返回约定不变。每 KLD 独占 160 bytes：前 32 bytes 是 8 个最终状态槽，后 128 bytes 是 32 harts 的归并 scratch；不占 SDK 保留 SPM。每 context 的有界 arena 容纳 255 条记录，各 stripe 单写者发布所有参与 harts 的归并状态。提交前置为 `UINT32_MAX`，kernel 写回 0 成功或其他非零错误；sync 后一次读回本 group 记录，未写和任一失败均拒收整个 batch。HAL rc、HAL sync result 与 kernel 输出状态分开处理。
- wrapper 在提交 HAL 前必须校验 buffer handle/lifetime、`count * sizeof(float)` overflow、全部 view 的访问范围和 4-byte alignment，并拒绝 add 超出 `UINT32_MAX` 的 count。buffer 地址只能由 registry 解析；非法参数不能依赖 caller 自查或 device early-return 后静默成功。
- command group 是阻塞 batch；SPM envelope 在 launch 调用内由 driver 复制，code/read/write/status BO 则 pin 到安全完成，状态记录还须保留到内部读回结束。正常与可报告错误路径遵循共同 completion epilogue。故障后 poison context，禁止复用无效结果或静默 fallback；未证明 quiescent 的 BO 与其 allocator 必须 quarantine，不能因为状态槽成功、destroy 成功或进程退出就假定安全释放。
- 官方 Dots TCIM 路径仅作为 ABI/数值参考，entry-only relink 不作为生产路线或验收前提。
