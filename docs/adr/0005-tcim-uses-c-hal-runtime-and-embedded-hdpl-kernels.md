---
status: accepted
---

# TCIM 使用纯 C HAL runtime 与内嵌 HDPL kernel

在 [ADR-0002](0002-build-tcim-as-independent-target.md) 的独立 TCIM build 内，XH2 operator runtime 在 host 侧保持纯 C 并直接调用 `libhal_xh2a`；device kernel 由 HDPL/Primitive 提前编译为内嵌的 C byte array，并通过 SDK 原生参数 ABI 启动。[Issue #6](https://github.com/fixedpointworks/ds4/issues/6) 的 D0–D4 与 hq50 验证表明，这条路径无需动态注册、HMM runtime、host C++ 或外部 kernel 文件即可完成资源管理、执行和错误传播，同时保持 `xh2rt` interface 狭窄、部署自包含。代价是新增或修改 kernel image 后必须重新构建 TCIM executable。

TCIM v1 的每个实际提交的 kernel invocation 固定使用 XH2 logical device 的完整启动拓扑：2 个 active cores、每 core 4 个 tiles、每 tile 4 个 RV harts；零工作调用不提交。拓扑由 `xh2rt` implementation 独占并写入 HAL launch descriptor，typed wrapper 不可选择；这以放弃 partial topology、逐 kernel 拓扑调优和同设备分核并发，换取只有一种合法状态的窄 interface，以及 work partition、全核 barrier 和每 hart outcome 发布的一致性。

Host 与 device 的 completion contract 只靠同一套源码与构建产物保持同步，不在 wire record 中增加版本 header，也不做 artifact 摘要校验。Runtime 在 native 参数前固定插入 completion IOMAP；completion record 仅包含按拓扑顺序排列的 32 个 `uint32_t` hart outcomes，共 128 bytes。`xh2rt_kernel.h` 是 host/device 共用的唯一 contract；它的改动通过 Make dependency 触发内嵌 kernel image 重建。Kernel image 大小取自完整 C array；Make 不对 0 或大于 65535 bytes 作构建期拒绝，`context_open` 在任何 HAL 调用前校验实际大小并失败关闭。因此不同构建产生的 host object 与 kernel image 不可混用。

公开 fallible runtime interface 统一以非零 `int` 表示成功、`0` 表示失败；失败时设置 `errno`，并由 runtime 将区分 HAL 调用返回值、HAL group sync result 与 kernel outcome 的诊断写入 `stderr`。Host preflight、group create 及可安全恢复的 bookkeeping/cleanup 失败不 poison；launch、sync、data/completion transfer 或 group destroy 失败、非零 HAL group sync result，以及任一非零或缺失的 kernel outcome，都会保存为 context 的首个 fatal poison cause 并禁止后续 work。Poison 表示本 batch 输出与 context 不再可信，并不等同于资源不可回收；只有无法证明 device 已静止或资源可安全回收时，相关 group、BO 与 allocator 才继续 pin/quarantine。Context 是串行对象，调用方必须串行化同一 context 上的所有操作。
