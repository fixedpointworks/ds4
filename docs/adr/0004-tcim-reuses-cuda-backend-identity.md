---
status: accepted
---

# TCIM 独立构建复用 CUDA backend identity

TCIM backend 不再增加 `DS4_BACKEND_TCIM`，而与 ROCm 一样在其独立构建中复用 `DS4_BACKEND_CUDA`；`DS4_TCIM_BUILD` 只在编译期选择 `tcim` 名称、CLI 入口和唯一一套 TCIM `ds4_gpu_*` 链接实现。这样保留 ADR-0002 的独立构建与 ADR-0003 的共享 GPU seam，同时取消 runtime backend dispatch：复用的是 public identity 和算子契约，不是 CUDA runtime 或 CUDA kernel 实现；非 TCIM 构建直接拒绝 `--tcim`，不得 alias 或 fallback。旧规格中关于独立 `DS4_BACKEND_TCIM` identity 和专用 availability validator 的条款均由本决定取代。
