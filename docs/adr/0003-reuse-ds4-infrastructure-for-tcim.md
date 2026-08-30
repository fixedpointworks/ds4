# TCIM 优先复用 DS4 基础设施

TCIM backend 在 incubation 与首发实现中复用 DS4 现有的 CLI/help、backend/platform validation、GGUF 与 SSD 权重路径 orchestration、测试入口和 `ds4_gpu_*` seam，不为 TCIM 新增通用基础设施特性；backend identity 按 [ADR-0004](0004-tcim-reuses-cuda-backend-identity.md) 在独立 TCIM 构建中复用 `DS4_BACKEND_CUDA`。若现有基础设施无法承载 Issue #1 的目标语义，只允许做最小、后端中立且不改变 Metal/CUDA/ROCm/CPU 既有行为的共享修改，并将 TCIM-specific 行为留在独立构建目标与 adapter 后；必要回归继续使用现有测试入口或无设备编译配置，不建立新的 capability、dispatch、validation 或测试框架。现有诊断入口按其依赖的 CPU/reference 或当前构建提供的 `ds4_gpu_*` seam 工作；TCIM runtime 初始化失败时沿已链接 adapter 的既有错误路径传播，不增加 TCIM-specific availability validator。
