# TCIM 优先复用 DS4 基础设施

TCIM backend 在 incubation 与首发实现中复用 DS4 现有的 backend identity、CLI/help、backend/platform validation、GGUF 与 SSD 权重路径 orchestration、测试入口和 `ds4_gpu_*` seam，不为 TCIM 新增通用基础设施特性。若现有基础设施无法承载 Issue #1 已冻结的语义，只允许做最小、后端中立且不改变 Metal/CUDA/ROCm/CPU 既有行为的共享修改，并将 TCIM-specific 行为留在独立构建目标与 adapter 后；必要回归继续使用现有测试入口或无设备编译配置，不建立新的 capability、dispatch、validation 或测试框架。这一取舍要求 TCIM 适配既有接口，以减少最终 merge upstream 的 diff、冲突与跨 backend 回归风险。现有诊断入口按其依赖的 CPU/reference 或当前构建提供的 `ds4_gpu_*` seam 工作，不增加 TCIM-specific identity gate；TCIM runtime 尚不可用时由既有 availability validation 拒绝，就绪后直接复用相同入口。
