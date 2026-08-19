# TCIM 使用独立构建目标

TCIM backend 与 Metal/CUDA/ROCm 一样，由独立构建目标在链接期提供唯一一套 `ds4_gpu_*` 实现，不与其他 GPU backend 同时链接，也不引入 runtime backend dispatch 或 ops table。Steps 1–5 均在 incubation branch 内开发，Step 6 前不向 upstream 提交中间 PR；独立构建可以直接复用现有全局 GPU seam，以较小的共享 runtime 改动建立 TCIM 开发入口，并避免为尚未公开支持的 backend 提前扩展架构。代价是同一构建产物不能在 TCIM 与其他 GPU backend 之间切换，跨 backend 回归与对比通过各自的构建产物完成；首发范围与验收规范仍只由 [Issue #1](https://github.com/fixedpointworks/ds4/issues/1) 定义。
